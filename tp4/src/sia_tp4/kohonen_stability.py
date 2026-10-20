"""Medir estabilidad temática y de shortlists del SOM final."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen import KohonenConfig
from .kohonen_analysis import KohonenRun, run_kohonen
from .kohonen_interpretation import dominant_topic_regions
from .pca_analysis import _load_pyplot, _save_figure


TOPICS = ("football", "makeup", "dogs", "art")
CAMPAIGNS = (
    ("futbol", "football"),
    ("belleza", "makeup"),
    ("perros", "dogs"),
    ("arte", "art"),
)
CAMPAIGN_LABELS = {
    "futbol": "Fútbol",
    "belleza": "Belleza",
    "perros": "Perros",
    "arte": "Arte",
}
CLASS_COLORS = {
    "estable": "#59A14F",
    "fronterizo": "#F28E2B",
    "inestable": "#E15759",
}
_KOHONEN_FIELDS = {
    "grid_size",
    "epochs",
    "learning_rate_start",
    "learning_rate_end",
    "radius_start",
    "radius_end",
    "initialization",
    "shuffle",
}


@dataclass(frozen=True)
class StabilityDesign:
    """Configuración fija, semillas y umbrales descriptivos."""

    name: str
    seeds: tuple[int, ...]
    stable_min_count: int
    frontier_min_count: int
    transition_threshold_pct: float
    kohonen_parameters: dict

    def run_for_seed(self, seed: int) -> KohonenRun:
        config = KohonenConfig(seed=seed, **self.kohonen_parameters)
        return KohonenRun(f"{self.name}-seed-{seed:02d}", config)


@dataclass(frozen=True)
class CampaignShortlist:
    """Neurona ocupada de mayor afinidad para una campaña y semilla."""

    campaign: str
    topic: str
    row: int
    column: int
    topic_pct: float
    dominant_region: str
    dominant_region_pct: float
    creator_indexes: tuple[int, ...]


@dataclass(frozen=True)
class SeedStabilityResult:
    """Información mínima agregable de una corrida, sin mapas por semilla."""

    seed: int
    run: str
    quantization_error: float
    occupied_units: int
    weak_topic_cells: int
    creator_regions: tuple[str, ...]
    creator_transition_flags: tuple[bool, ...]
    shortlists: tuple[CampaignShortlist, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datos/creadores.csv"),
        help="CSV fuente (por defecto, datos/creadores.csv).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kohonen/stability.json"),
        help="Diseño de estabilidad con configuración final y semillas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/08-stability"),
        help="Directorio único para el informe y resultados agregados.",
    )
    return parser.parse_args(argv)


def load_design(path: Path | str) -> StabilityDesign:
    """Leer y validar el diseño completo antes de entrenar."""

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON inválido en {path}: {error.msg}") from error
    if not isinstance(document, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {
        "name",
        "seeds",
        "stable_min_count",
        "frontier_min_count",
        "transition_threshold_pct",
        "kohonen",
    }
    unknown = set(document) - expected
    missing = expected - set(document)
    if unknown or missing:
        raise ValueError(
            "campos inválidos en la configuración: "
            f"faltan={sorted(missing)}, sobran={sorted(unknown)}"
        )

    name = document["name"]
    if not isinstance(name, str) or not name:
        raise ValueError("name debe ser un texto no vacío")
    seeds = document["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 10
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds debe contener exactamente 10 enteros únicos")

    stable = document["stable_min_count"]
    frontier = document["frontier_min_count"]
    if (
        isinstance(stable, bool)
        or isinstance(frontier, bool)
        or not isinstance(stable, int)
        or not isinstance(frontier, int)
        or not 1 <= frontier <= stable <= len(seeds)
    ):
        raise ValueError(
            "los umbrales deben cumplir 1 <= fronterizo <= estable <= semillas"
        )
    transition = document["transition_threshold_pct"]
    if isinstance(transition, bool) or not isinstance(transition, (int, float)):
        raise ValueError("transition_threshold_pct debe ser numérico")
    if not 0 <= transition <= 100:
        raise ValueError("transition_threshold_pct debe estar entre 0 y 100")

    parameters = document["kohonen"]
    if not isinstance(parameters, dict) or set(parameters) != _KOHONEN_FIELDS:
        raise ValueError(
            "kohonen debe contener exactamente la configuración final sin seed"
        )
    try:
        KohonenConfig(seed=seeds[0], **parameters)
    except (TypeError, ValueError) as error:
        raise ValueError(f"configuración de Kohonen inválida: {error}") from error

    return StabilityDesign(
        name=name,
        seeds=tuple(seeds),
        stable_min_count=stable,
        frontier_min_count=frontier,
        transition_threshold_pct=float(transition),
        kohonen_parameters=dict(parameters),
    )


def stability_class(count: int, design: StabilityDesign) -> str:
    """Clasificar una frecuencia con umbrales declarados antes de medir."""

    if count >= design.stable_min_count:
        return "estable"
    if count >= design.frontier_min_count:
        return "fronterizo"
    return "inestable"


def _topic_planes(dataset: CreatorDataset, prototypes: np.ndarray) -> np.ndarray:
    indexes = {name: index for index, name in enumerate(dataset.feature_names)}
    football = prototypes[:, :, indexes["topic_football_pct"]]
    makeup = prototypes[:, :, indexes["topic_makeup_pct"]]
    dogs = prototypes[:, :, indexes["topic_dogs_pct"]]
    art = 100 - football - makeup - dogs
    return np.stack((football, makeup, dogs, art), axis=-1)


def _campaign_shortlists(
    dataset: CreatorDataset,
    assignments: np.ndarray,
    prototypes: np.ndarray,
    region_map: np.ndarray,
    strengths: np.ndarray,
) -> tuple[CampaignShortlist, ...]:
    topic_planes = _topic_planes(dataset, prototypes)
    occupied = []
    for row in range(prototypes.shape[0]):
        for column in range(prototypes.shape[1]):
            indexes = tuple(
                int(index)
                for index in np.flatnonzero(
                    np.all(assignments == (row, column), axis=1)
                )
            )
            if indexes:
                occupied.append((row, column, indexes))

    shortlists = []
    for campaign, topic in CAMPAIGNS:
        topic_index = TOPICS.index(topic)
        row, column, indexes = max(
            occupied,
            key=lambda cell: topic_planes[cell[0], cell[1], topic_index],
        )
        shortlists.append(
            CampaignShortlist(
                campaign=campaign,
                topic=topic,
                row=row,
                column=column,
                topic_pct=float(topic_planes[row, column, topic_index]),
                dominant_region=str(region_map[row, column]),
                dominant_region_pct=float(strengths[row, column]),
                creator_indexes=indexes,
            )
        )
    return tuple(shortlists)


def evaluate_seeds(
    dataset: CreatorDataset, design: StabilityDesign
) -> list[SeedStabilityResult]:
    """Entrenar sólo la configuración final para las diez semillas."""

    results = []
    for seed in design.seeds:
        run = design.run_for_seed(seed)
        result = run_kohonen(dataset, run)
        region_map, strengths = dominant_topic_regions(result)
        creator_regions = tuple(
            str(region_map[row, column]) for row, column in result.assignments
        )
        transition_flags = tuple(
            bool(strengths[row, column] < design.transition_threshold_pct)
            for row, column in result.assignments
        )
        shortlists = _campaign_shortlists(
            dataset,
            result.assignments,
            result.model.prototypes(original_scale=True),
            region_map,
            strengths,
        )
        results.append(
            SeedStabilityResult(
                seed=seed,
                run=run.name,
                quantization_error=result.quantization_error,
                occupied_units=result.occupied_units,
                weak_topic_cells=int(
                    np.sum(strengths < design.transition_threshold_pct)
                ),
                creator_regions=creator_regions,
                creator_transition_flags=transition_flags,
                shortlists=shortlists,
            )
        )
    return results


def summarize_creator_stability(
    dataset: CreatorDataset,
    results: Sequence[SeedStabilityResult],
    design: StabilityDesign,
) -> list[dict]:
    """Resumir región modal, empates, transiciones y shortlists por creador."""

    rows = []
    for index, (creator_id, name) in enumerate(
        zip(dataset.creator_ids, dataset.names)
    ):
        counts = {
            topic: sum(result.creator_regions[index] == topic for result in results)
            for topic in TOPICS
        }
        maximum = max(counts.values())
        modal = tuple(topic for topic in TOPICS if counts[topic] == maximum)
        shortlist_counts = {
            campaign: sum(
                index in shortlist.creator_indexes
                for result in results
                for shortlist in result.shortlists
                if shortlist.campaign == campaign
            )
            for campaign, _ in CAMPAIGNS
        }
        transition_count = sum(
            result.creator_transition_flags[index] for result in results
        )
        rows.append(
            {
                "creator_id": creator_id,
                "name": name,
                "modal_region": ";".join(modal),
                "modal_region_tie": len(modal) > 1,
                **{f"region_{topic}_count": counts[topic] for topic in TOPICS},
                "region_stability_count": maximum,
                "region_stability_pct": 100 * maximum / len(results),
                "region_stability_class": stability_class(maximum, design),
                "transition_count": transition_count,
                "transition_pct": 100 * transition_count / len(results),
                **{
                    f"shortlist_{campaign}_count": shortlist_counts[campaign]
                    for campaign, _ in CAMPAIGNS
                },
            }
        )
    return rows


def summarize_shortlist_candidates(
    dataset: CreatorDataset,
    results: Sequence[SeedStabilityResult],
    design: StabilityDesign,
) -> list[dict]:
    """Clasificar por campaña a todo creador incluido al menos una vez."""

    rows = []
    for campaign, _ in CAMPAIGNS:
        for index, (creator_id, name) in enumerate(
            zip(dataset.creator_ids, dataset.names)
        ):
            count = sum(
                index in shortlist.creator_indexes
                for result in results
                for shortlist in result.shortlists
                if shortlist.campaign == campaign
            )
            if count == 0:
                continue
            rows.append(
                {
                    "campaign": campaign,
                    "creator_id": creator_id,
                    "name": name,
                    "inclusion_count": count,
                    "inclusion_pct": 100 * count / len(results),
                    "candidate_class": stability_class(count, design),
                }
            )
    return rows


def summarize_campaigns(
    results: Sequence[SeedStabilityResult], candidate_rows: Sequence[dict]
) -> list[dict]:
    """Agregar tamaños, afinidades y candidatos por campaña."""

    rows = []
    for campaign, _ in CAMPAIGNS:
        shortlists = [
            shortlist
            for result in results
            for shortlist in result.shortlists
            if shortlist.campaign == campaign
        ]
        candidates = [row for row in candidate_rows if row["campaign"] == campaign]
        sizes = np.asarray([len(item.creator_indexes) for item in shortlists])
        affinities = np.asarray([item.topic_pct for item in shortlists])
        record = {
            "campaign": campaign,
            "shortlist_size_mean": float(np.mean(sizes)),
            "shortlist_size_std": float(np.std(sizes, ddof=1)),
            "shortlist_size_min": int(np.min(sizes)),
            "shortlist_size_max": int(np.max(sizes)),
            "topic_pct_mean": float(np.mean(affinities)),
            "topic_pct_std": float(np.std(affinities, ddof=1)),
            "topic_pct_min": float(np.min(affinities)),
            "topic_pct_max": float(np.max(affinities)),
            "distinct_candidates": len(candidates),
            "distinct_shortlists": len(
                {
                    tuple(sorted(shortlist.creator_indexes))
                    for shortlist in shortlists
                }
            ),
            "dominant_topic_matches": sum(
                shortlist.dominant_region == shortlist.topic
                for shortlist in shortlists
            ),
        }
        for category in ("estable", "fronterizo", "inestable"):
            selected = [
                row["creator_id"]
                for row in candidates
                if row["candidate_class"] == category
            ]
            record[f"{category}_count"] = len(selected)
            record[f"{category}_creator_ids"] = ";".join(selected)
        rows.append(record)
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"no hay filas para escribir en {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    dataset: CreatorDataset,
    results: Sequence[SeedStabilityResult],
    design: StabilityDesign,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Escribir tablas únicas y agregadas; nunca una carpeta por semilla."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    creator_rows = summarize_creator_stability(dataset, results, design)
    candidate_rows = summarize_shortlist_candidates(dataset, results, design)
    campaign_rows = summarize_campaigns(results, candidate_rows)
    run_rows = [
        {
            "run": result.run,
            "seed": result.seed,
            "quantization_error": result.quantization_error,
            "occupied_units": result.occupied_units,
            "weak_topic_cells": result.weak_topic_cells,
        }
        for result in results
    ]
    shortlist_rows = [
        {
            "seed": result.seed,
            "campaign": shortlist.campaign,
            "row": shortlist.row,
            "column": shortlist.column,
            "hits": len(shortlist.creator_indexes),
            "creator_ids": ";".join(
                dataset.creator_ids[index] for index in shortlist.creator_indexes
            ),
            "topic_pct": shortlist.topic_pct,
            "dominant_region": shortlist.dominant_region,
            "dominant_region_pct": shortlist.dominant_region_pct,
        }
        for result in results
        for shortlist in result.shortlists
    ]
    _write_csv(generated / "corridas.csv", run_rows)
    _write_csv(generated / "shortlists_semillas.csv", shortlist_rows)
    _write_csv(generated / "estabilidad_creadores.csv", creator_rows)
    _write_csv(generated / "candidatos_shortlist.csv", candidate_rows)
    _write_csv(generated / "resumen_campanas.csv", campaign_rows)
    return creator_rows, candidate_rows, campaign_rows


def write_plots(
    output: Path,
    creator_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
    seed_count: int,
) -> None:
    """Generar dos figuras agregadas, sin mapas repetidos por semilla."""

    plt = _load_pyplot()
    generated = output / "generated"
    ordered = sorted(
        creator_rows,
        key=lambda row: (row["region_stability_count"], row["creator_id"]),
    )
    figure, axis = plt.subplots(figsize=(10, 12))
    y = np.arange(len(ordered))
    axis.barh(
        y,
        [row["region_stability_count"] for row in ordered],
        color=[CLASS_COLORS[row["region_stability_class"]] for row in ordered],
    )
    axis.set_yticks(y, [row["creator_id"] for row in ordered], fontsize=8)
    axis.set_xlim(0, seed_count)
    axis.set_xlabel(f"Semillas con la región temática modal (de {seed_count})")
    axis.set_title("Estabilidad temática por creador")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, generated / "estabilidad_regiones.svg")

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, (campaign, _) in zip(axes.flat, CAMPAIGNS):
        rows = sorted(
            (row for row in candidate_rows if row["campaign"] == campaign),
            key=lambda row: (row["inclusion_count"], row["creator_id"]),
        )
        positions = np.arange(len(rows))
        axis.barh(
            positions,
            [row["inclusion_count"] for row in rows],
            color=[CLASS_COLORS[row["candidate_class"]] for row in rows],
        )
        axis.set_yticks(positions, [row["creator_id"] for row in rows], fontsize=8)
        axis.set_xlim(0, seed_count)
        axis.set_title(CAMPAIGN_LABELS[campaign])
        axis.grid(axis="x", alpha=0.25)
    figure.supxlabel(f"Inclusiones en la shortlist (de {seed_count} semillas)")
    figure.suptitle("Frecuencia de candidatos por campaña")
    figure.tight_layout()
    _save_figure(figure, generated / "frecuencia_shortlists.svg")


def write_report(
    output: Path,
    design: StabilityDesign,
    results: Sequence[SeedStabilityResult],
    creator_rows: Sequence[dict],
    candidate_rows: Sequence[dict],
    campaign_rows: Sequence[dict],
) -> None:
    """Documentar el protocolo y dejar la interpretación trazable."""

    class_counts = {
        category: sum(
            row["region_stability_class"] == category for row in creator_rows
        )
        for category in ("estable", "fronterizo", "inestable")
    }
    campaign_table = "\n".join(
        "| {campaign} | {shortlist_size_mean:.1f} | {topic_pct_mean:.1f} | "
        "{distinct_shortlists} | {estable_count} | {fronterizo_count} | "
        "{inestable_count} |".format(
            **{**row, "campaign": CAMPAIGN_LABELS[row["campaign"]]}
        )
        for row in campaign_rows
    )
    candidate_table = "\n".join(
        "| {campaign} | `{stable}` | `{frontier}` |".format(
            campaign=CAMPAIGN_LABELS[campaign],
            stable=";".join(
                row["creator_id"]
                for row in candidate_rows
                if row["campaign"] == campaign
                and row["candidate_class"] == "estable"
            )
            or "ninguno",
            frontier=";".join(
                row["creator_id"]
                for row in candidate_rows
                if row["campaign"] == campaign
                and row["candidate_class"] == "fronterizo"
            )
            or "ninguno",
        )
        for campaign, _ in CAMPAIGNS
    )
    transition_assignments = sum(
        sum(result.creator_transition_flags) for result in results
    )
    total_assignments = len(creator_rows) * len(results)
    unstable_creator_ids = ";".join(
        row["creator_id"]
        for row in creator_rows
        if row["region_stability_class"] == "inestable"
    )
    matching_shortlists = sum(
        row["dominant_topic_matches"] for row in campaign_rows
    )
    total_shortlists = len(CAMPAIGNS) * len(results)
    distinct_shortlist_counts = [
        row["distinct_shortlists"] for row in campaign_rows
    ]
    parameters = design.kohonen_parameters
    configuration_summary = (
        f"grilla `{parameters['grid_size']} x {parameters['grid_size']}`, "
        f"tasa constante `{parameters['learning_rate_start']}`, radio "
        f"`{parameters['radius_start']:.3f} -> {parameters['radius_end']:.1f}`, "
        f"`{parameters['epochs']}` épocas e inicialización "
        f"`{parameters['initialization']}`"
    )
    classification_summary = (
        f"estable `{design.stable_min_count}-10`, fronterizo "
        f"`{design.frontier_min_count}-{design.stable_min_count - 1}` e "
        f"inestable `0-{design.frontier_min_count - 1}` apariciones"
    )
    report = f"""# Estabilidad de la configuración final de Kohonen

## Protocolo

- Configuración: {configuration_summary}.
- Semillas: `{', '.join(map(str, design.seeds))}`.
- Región temática: tema dominante del prototipo de la BMU de cada creador.
- Celda de transición: afinidad temática máxima menor que `{design.transition_threshold_pct:.0f} %`.
- Clasificación predefinida: {classification_summary}.

Las coordenadas de la grilla pueden rotarse o reflejarse entre semillas. Por eso
no se comparan coordenadas: se compara el significado temático del prototipo.
Este conteo es un diagnóstico descriptivo adicional, no una métrica usada para
entrenar o volver a elegir automáticamente los hiperparámetros.

## Estabilidad temática

De los {len(creator_rows)} creadores, **{class_counts['estable']}** son estables,
**{class_counts['fronterizo']}** fronterizos y **{class_counts['inestable']}**
inestables según su región temática modal. Esto equivale a
**{100 * class_counts['estable'] / len(creator_rows):.0f} %** estables y
**{100 * (class_counts['estable'] + class_counts['fronterizo']) / len(creator_rows):.0f} %**
entre estables y fronterizos. Los inestables son `{unstable_creator_ids}`.

Hubo {transition_assignments} asignaciones a celdas de transición sobre
{total_assignments} combinaciones creador-semilla
({100 * transition_assignments / total_assignments:.1f} %).

![Estabilidad regional](generated/estabilidad_regiones.svg)

## Estabilidad de las shortlists

En cada semilla se selecciona, para cada campaña, la neurona ocupada cuyo
prototipo tiene mayor afinidad con el tema correspondiente. La frecuencia indica
cuántas veces aparece cada creador; no es un ranking de calidad comercial.

| Campaña | Tamaño medio | Afinidad media (%) | Shortlists distintas | Estables | Fronterizos | Inestables |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{campaign_table}

![Frecuencia de shortlists](generated/frecuencia_shortlists.svg)

| Campaña | Candidatos estables | Candidatos fronterizos |
| --- | --- | --- |
{candidate_table}

Las {total_shortlists} celdas elegidas tienen como tema dominante el de su
campaña en {matching_shortlists} casos. Sin embargo, cada campaña produjo entre
{min(distinct_shortlist_counts)} y {max(distinct_shortlist_counts)} conjuntos
distintos en las {len(results)} semillas. La semántica general se conserva, pero
la partición interna de cada región entre neuronas no.

## Decisión metodológica

La inestabilidad de la shortlist estricta no alcanza para descartar la
configuración: sólo {class_counts['inestable']} de {len(creator_rows)} creadores
son regionalmente inestables y las celdas elegidas conservan el tema esperado.
El siguiente análisis no debe tratar una única neurona como selector definitivo.
Debe partir de la región temática completa y comparar dentro de ella las
variables originales y los objetivos de cada campaña.

No se reabre todavía el barrido de hiperparámetros. La comparación con las
configuraciones finalistas queda condicionada a que la selección por región
completa también resulte inestable o cambie las conclusiones comerciales.

## Archivos reproducibles

- `generated/corridas.csv`: calidad y ocupación de las diez corridas;
- `generated/shortlists_semillas.csv`: una fila por campaña y semilla;
- `generated/estabilidad_creadores.csv`: región y transición agregadas;
- `generated/candidatos_shortlist.csv`: frecuencia y clase por campaña;
- `generated/resumen_campanas.csv`: resumen agregado de las cuatro campañas.

Las {len(results)} redes se regeneran desde un único JSON. No se guarda una
carpeta ni un conjunto de mapas por semilla.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def run_analysis(
    data: Path, config: Path, output: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    """Ejecutar el diagnóstico completo y devolver sus tres agregados."""

    dataset = load_creator_dataset(data)
    design = load_design(config)
    results = evaluate_seeds(dataset, design)
    creator_rows, candidate_rows, campaign_rows = write_tables(
        output, dataset, results, design
    )
    write_plots(output, creator_rows, candidate_rows, len(design.seeds))
    write_report(
        output,
        design,
        results,
        creator_rows,
        candidate_rows,
        campaign_rows,
    )
    return creator_rows, candidate_rows, campaign_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    creator_rows, candidate_rows, _ = run_analysis(
        args.data, args.config, args.output
    )
    print(
        "Estabilidad de Kohonen completa: "
        f"{len(creator_rows)} creadores, {len(candidate_rows)} candidatos; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
