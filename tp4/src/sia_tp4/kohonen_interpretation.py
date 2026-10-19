"""Interpretar las cuatro configuraciones finales de Kohonen."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen_analysis import KohonenRun, KohonenRunResult, load_run, run_kohonen
from .kohonen_epoch_experiment import EPOCH_PROFILES, SELECTED_EPOCHS
from .pca_analysis import _load_pyplot, _save_figure


INTERPRETATION_PROFILES = EPOCH_PROFILES[:4]
FEATURE_GROUPS = (
    (
        "exposicion",
        "Exposición, actividad y audiencia",
        (
            "average_reach",
            "monthly_posts",
            "argentina_audience_pct",
            "non_follower_reach_pct",
        ),
    ),
    (
        "respuesta_comercial",
        "Respuesta y contexto comercial",
        (
            "total_engagement_per_1000",
            "profile_visits_per_1000",
            "price_usd",
            "sponsored_posts_pct",
        ),
    ),
    (
        "temas",
        "Afinidad temática",
        (
            "topic_football_pct",
            "topic_makeup_pct",
            "topic_dogs_pct",
            "topic_art_pct",
        ),
    ),
    (
        "contenido",
        "Tipo de contenido",
        (
            "content_tutorial_pct",
            "content_review_pct",
            "content_entertainment_pct",
            "content_showcase_pct",
        ),
    ),
    (
        "consumo_auxiliares",
        "Consumo y auxiliares de negocio",
        (
            "average_watch_time_seconds",
            "estimated_argentina_reach",
            "followers",
        ),
    ),
)


@dataclass(frozen=True)
class InterpretationMetrics:
    """Métricas usadas para elegir una semilla central."""

    run: str
    profile: str
    seed: int
    quantization_error: float
    dead_units_pct: float
    maximum_hits_pct: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datos/creadores.csv"),
        help="CSV fuente (por defecto, datos/creadores.csv).",
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("configs/kohonen/06-initialization"),
        help="Configuraciones que contienen las corridas sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/07-interpretation"),
        help="Directorio del estudio interpretativo.",
    )
    return parser.parse_args(argv)


def _radius_fraction(run: KohonenRun) -> float:
    diameter = np.sqrt(2) * (run.config.grid_size - 1)
    return float((run.config.radius_start - 1) / (diameter - 1))


def _profile_for_run(run: KohonenRun) -> str:
    config = run.config
    fraction = _radius_fraction(run)
    matches = [
        name
        for name, grid, rate_start, rate_end, radius_fraction
        in INTERPRETATION_PROFILES
        if config.grid_size == grid
        and np.isclose(config.learning_rate_start, rate_start)
        and np.isclose(config.learning_rate_end, rate_end)
        and np.isclose(fraction, radius_fraction)
    ]
    if len(matches) != 1:
        raise ValueError(f"'{run.name}' no pertenece a una candidata final")
    return matches[0]


def load_interpretation_runs(directory: Path | str) -> list[KohonenRun]:
    """Cargar cinco semillas sample para cada una de las cuatro candidatas."""

    paths = sorted(Path(directory).glob("*.json"))
    runs = []
    for path in paths:
        run = load_run(path)
        if run.config.initialization != "sample" or run.config.grid_size == 6:
            continue
        _profile_for_run(run)
        if run.config.epochs != SELECTED_EPOCHS:
            raise ValueError(f"'{run.name}' no usa 150 épocas")
        runs.append(run)
    combinations = {
        (_profile_for_run(run), run.config.seed)
        for run in runs
    }
    expected = {
        (profile[0], seed)
        for profile in INTERPRETATION_PROFILES
        for seed in range(5)
    }
    if combinations != expected or len(runs) != len(expected):
        raise ValueError("se necesitan cinco semillas sample por candidata")
    return runs


def evaluate_runs(
    dataset: CreatorDataset, runs: Sequence[KohonenRun]
) -> tuple[list[InterpretationMetrics], dict[str, KohonenRunResult]]:
    """Entrenar las 20 redes necesarias para elegir representantes."""

    metrics = []
    results = {}
    sample_count = len(dataset.creator_ids)
    for run in runs:
        result = run_kohonen(dataset, run)
        results[run.name] = result
        metrics.append(
            InterpretationMetrics(
                run=run.name,
                profile=_profile_for_run(run),
                seed=run.config.seed,
                quantization_error=result.quantization_error,
                dead_units_pct=100 * result.dead_units / result.hit_map.size,
                maximum_hits_pct=100 * int(result.hit_map.max()) / sample_count,
            )
        )
    return metrics, results


def select_representatives(
    metrics: Sequence[InterpretationMetrics],
) -> list[InterpretationMetrics]:
    """Seleccionar la semilla más central en tres métricas por perfil."""

    fields = ("quantization_error", "dead_units_pct", "maximum_hits_pct")
    selected = []
    for profile, *_ in INTERPRETATION_PROFILES:
        group = [metric for metric in metrics if metric.profile == profile]
        if len(group) != 5:
            raise ValueError(f"'{profile}' no tiene cinco semillas")
        values = np.asarray(
            [[getattr(metric, field) for field in fields] for metric in group]
        )
        medians = np.median(values, axis=0)
        scales = np.std(values, axis=0, ddof=1)
        scales[scales == 0] = 1.0
        distances = np.abs((values - medians) / scales).sum(axis=1)
        selected.append(
            min(
                zip(distances, group),
                key=lambda item: (item[0], item[1].run),
            )[1]
        )
    return selected


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _followers_map(
    dataset: CreatorDataset, result: KohonenRunResult
) -> np.ndarray:
    size = result.run.config.grid_size
    totals = np.zeros((size, size), dtype=float)
    counts = np.zeros((size, size), dtype=int)
    for followers, (row, column) in zip(dataset.followers, result.assignments):
        totals[row, column] += followers
        counts[row, column] += 1
    averages = np.full((size, size), np.nan)
    np.divide(totals, counts, out=averages, where=counts > 0)
    return averages


def _plane(
    dataset: CreatorDataset,
    result: KohonenRunResult,
    feature: str,
) -> tuple[np.ndarray, str]:
    """Obtener un plano comparable y la unidad usada para mostrarlo."""

    feature_names = result.model.feature_names
    standardized = result.model.weights
    original = result.model.prototypes(original_scale=True)
    if feature in feature_names:
        index = feature_names.index(feature)
        return standardized[:, :, index], "z-score del prototipo"
    indexes = {name: feature_names.index(name) for name in feature_names}
    if feature == "topic_art_pct":
        value = 100 - sum(
            original[:, :, indexes[name]]
            for name in (
                "topic_football_pct",
                "topic_makeup_pct",
                "topic_dogs_pct",
            )
        )
        return value, "porcentaje reconstruido"
    if feature == "content_showcase_pct":
        value = 100 - sum(
            original[:, :, indexes[name]]
            for name in (
                "content_tutorial_pct",
                "content_review_pct",
                "content_entertainment_pct",
            )
        )
        return value, "porcentaje reconstruido"
    if feature == "estimated_argentina_reach":
        value = (
            original[:, :, indexes["average_reach"]]
            * original[:, :, indexes["argentina_audience_pct"]]
            / 100
        )
        return value, "cuentas argentinas estimadas"
    if feature == "followers":
        return _followers_map(dataset, result), "promedio observado (auxiliar)"
    raise ValueError(f"variable desconocida: {feature}")


def write_tables(
    output: Path,
    dataset: CreatorDataset,
    metrics: Sequence[InterpretationMetrics],
    results: Sequence[KohonenRunResult],
) -> tuple[list[dict], list[dict]]:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    selected_names = {result.run.name for result in results}
    _write_csv(
        generated / "seleccion_representantes.csv",
        [*InterpretationMetrics.__dataclass_fields__, "selected"],
        (
            {**vars(metric), "selected": metric.run in selected_names}
            for metric in metrics
        ),
    )

    assignment_rows = []
    neuron_rows = []
    interpretation_rows = []
    campaign_rows = []
    for result in results:
        profile = _profile_for_run(result.run)
        size = result.run.config.grid_size
        original = result.model.prototypes(original_scale=True)
        standardized = result.model.weights
        feature_indexes = {
            name: index for index, name in enumerate(dataset.feature_names)
        }
        cell_details = []
        for creator_id, name, followers, position in zip(
            dataset.creator_ids,
            dataset.names,
            dataset.followers,
            result.assignments,
        ):
            assignment_rows.append(
                {
                    "profile": profile,
                    "run": result.run.name,
                    "creator_id": creator_id,
                    "name": name,
                    "followers": followers,
                    "row": int(position[0]),
                    "column": int(position[1]),
                }
            )
        for row in range(size):
            for column in range(size):
                indexes = np.flatnonzero(
                    np.all(result.assignments == (row, column), axis=1)
                )
                record = {
                    "profile": profile,
                    "run": result.run.name,
                    "row": row,
                    "column": column,
                    "hits": int(len(indexes)),
                    "u_distance": float(result.u_matrix[row, column]),
                    "creator_ids": ";".join(dataset.creator_ids[i] for i in indexes),
                    "creator_names": ";".join(dataset.names[i] for i in indexes),
                    "followers_mean": (
                        float(np.mean(dataset.followers[indexes]))
                        if len(indexes)
                        else ""
                    ),
                }
                for feature_index, feature in enumerate(dataset.feature_names):
                    record[f"prototype_z__{feature}"] = float(
                        standardized[row, column, feature_index]
                    )
                    record[f"prototype_original__{feature}"] = float(
                        original[row, column, feature_index]
                    )
                    record[f"assigned_mean__{feature}"] = (
                        float(np.mean(dataset.values[indexes, feature_index]))
                        if len(indexes)
                        else ""
                    )
                neuron_rows.append(record)
                if len(indexes):
                    topics = {
                        "football": float(
                            original[
                                row,
                                column,
                                feature_indexes["topic_football_pct"],
                            ]
                        ),
                        "makeup": float(
                            original[
                                row,
                                column,
                                feature_indexes["topic_makeup_pct"],
                            ]
                        ),
                        "dogs": float(
                            original[
                                row,
                                column,
                                feature_indexes["topic_dogs_pct"],
                            ]
                        ),
                    }
                    topics["art"] = 100 - sum(topics.values())
                    cell_details.append(
                        {
                            "row": row,
                            "column": column,
                            "indexes": indexes,
                            "topics": topics,
                            "dominant": max(topics, key=topics.get),
                        }
                    )
        dominant_values = [
            cell["topics"][cell["dominant"]] for cell in cell_details
        ]
        interpretation_rows.append(
            {
                "profile": profile,
                "run": result.run.name,
                "occupied_units": len(cell_details),
                "mean_dominant_topic_pct": float(np.mean(dominant_values)),
                **{
                    f"dominant_{topic}_units": sum(
                        cell["dominant"] == topic for cell in cell_details
                    )
                    for topic in ("football", "makeup", "dogs", "art")
                },
                **{
                    f"maximum_{topic}_pct": max(
                        cell["topics"][topic] for cell in cell_details
                    )
                    for topic in ("football", "makeup", "dogs", "art")
                },
            }
        )
        for campaign, topic in (
            ("futbol", "football"),
            ("belleza", "makeup"),
            ("perros", "dogs"),
            ("arte", "art"),
        ):
            cell = max(cell_details, key=lambda item: item["topics"][topic])
            row, column, indexes = (
                cell["row"],
                cell["column"],
                cell["indexes"],
            )
            prototype = original[row, column]
            reach = prototype[feature_indexes["average_reach"]]
            argentina_pct = prototype[feature_indexes["argentina_audience_pct"]]
            campaign_rows.append(
                {
                    "profile": profile,
                    "campaign": campaign,
                    "row": row,
                    "column": column,
                    "hits": len(indexes),
                    "creator_ids": ";".join(dataset.creator_ids[i] for i in indexes),
                    "topic_pct": cell["topics"][topic],
                    "average_reach": reach,
                    "argentina_audience_pct": argentina_pct,
                    "estimated_argentina_reach": reach * argentina_pct / 100,
                    "total_engagement_per_1000": prototype[
                        feature_indexes["total_engagement_per_1000"]
                    ],
                    "profile_visits_per_1000": prototype[
                        feature_indexes["profile_visits_per_1000"]
                    ],
                    "price_usd": prototype[feature_indexes["price_usd"]],
                    "content_tutorial_pct": prototype[
                        feature_indexes["content_tutorial_pct"]
                    ],
                    "content_review_pct": prototype[
                        feature_indexes["content_review_pct"]
                    ],
                    "content_entertainment_pct": prototype[
                        feature_indexes["content_entertainment_pct"]
                    ],
                }
            )
    _write_csv(
        generated / "asignaciones_finalistas.csv",
        list(assignment_rows[0]),
        assignment_rows,
    )
    _write_csv(
        generated / "perfiles_neuronas.csv",
        list(neuron_rows[0]),
        neuron_rows,
    )
    _write_csv(
        generated / "resumen_interpretabilidad.csv",
        list(interpretation_rows[0]),
        interpretation_rows,
    )
    _write_csv(
        generated / "regiones_campanas.csv",
        list(campaign_rows[0]),
        campaign_rows,
    )
    return interpretation_rows, campaign_rows


def _save_both(figure, path: Path) -> None:
    _save_figure(figure, path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")


def _profile_label(result: KohonenRunResult) -> str:
    return f"{_profile_for_run(result.run)}\nsemilla={result.run.config.seed}"


def write_hit_u_plot(output: Path, results: Sequence[KohonenRunResult]) -> None:
    plt = _load_pyplot()
    all_u = np.concatenate([result.u_matrix.ravel() for result in results])
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    colored = None
    for column, result in enumerate(results):
        for row, (matrix, title, vmin, vmax) in enumerate(
            (
                (result.hit_map, "Hits", 0, 6),
                (result.u_matrix, "Matriz U", float(all_u.min()), float(all_u.max())),
            )
        ):
            axis = axes[row, column]
            colored = axis.imshow(matrix, cmap="turbo", vmin=vmin, vmax=vmax)
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    value = int(matrix[y, x]) if row == 0 else f"{matrix[y, x]:.1f}"
                    axis.text(x, y, value, ha="center", va="center", fontsize=7)
            axis.set_title(f"{_profile_label(result)}\n{title}", fontsize=9)
            axis.set_xticks(range(matrix.shape[1]))
            axis.set_yticks(range(matrix.shape[0]))
    figure.suptitle("Ocupación y estructura de las cuatro candidatas")
    figure.tight_layout()
    _save_both(figure, output / "generated" / "hits_y_matrices_u")
    plt.close(figure)


def write_creator_plot(
    output: Path,
    dataset: CreatorDataset,
    results: Sequence[KohonenRunResult],
) -> None:
    plt = _load_pyplot()
    figure, axes = plt.subplots(2, 2, figsize=(16, 14), squeeze=False)
    for axis, result in zip(axes.flat, results):
        size = result.run.config.grid_size
        cells = [[[] for _ in range(size)] for _ in range(size)]
        for creator_id, (row, column) in zip(dataset.creator_ids, result.assignments):
            cells[row][column].append(creator_id)
        axis.imshow(result.hit_map, cmap="Blues", vmin=0, vmax=6)
        for row in range(size):
            for column in range(size):
                axis.text(
                    column,
                    row,
                    "\n".join(cells[row][column]) or "—",
                    ha="center",
                    va="center",
                    fontsize=6,
                )
        axis.set_title(_profile_label(result))
        axis.set_xticks(range(size))
        axis.set_yticks(range(size))
    figure.suptitle("Creadores asignados a las neuronas")
    figure.tight_layout()
    _save_both(figure, output / "generated" / "creadores_asignados")
    plt.close(figure)


def write_feature_groups(
    output: Path,
    dataset: CreatorDataset,
    results: Sequence[KohonenRunResult],
) -> None:
    plt = _load_pyplot()
    for filename, group_title, features in FEATURE_GROUPS:
        figure, axes = plt.subplots(
            len(features),
            len(results) + 1,
            figsize=(17, 3.5 * len(features)),
            squeeze=False,
            gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.06]},
        )
        for row, feature in enumerate(features):
            planes_and_units = [_plane(dataset, result, feature) for result in results]
            planes = [item[0] for item in planes_and_units]
            finite = np.concatenate(
                [plane[np.isfinite(plane)] for plane in planes]
            )
            if feature in dataset.feature_names:
                limit = max(abs(float(finite.min())), abs(float(finite.max())))
                vmin, vmax, cmap = -limit, limit, "RdBu_r"
            else:
                vmin, vmax, cmap = float(finite.min()), float(finite.max()), "viridis"
            colored = None
            for column, (result, plane) in enumerate(zip(results, planes)):
                axis = axes[row, column]
                colored = axis.imshow(plane, cmap=cmap, vmin=vmin, vmax=vmax)
                for y in range(plane.shape[0]):
                    for x in range(plane.shape[1]):
                        value = plane[y, x]
                        label = "—" if np.isnan(value) else f"{value:.1f}"
                        axis.text(x, y, label, ha="center", va="center", fontsize=6)
                if row == 0:
                    axis.set_title(_profile_label(result), fontsize=9)
                axis.set_xticks(range(plane.shape[1]))
                axis.set_yticks(range(plane.shape[0]))
                if column == 0:
                    axis.set_ylabel(feature, fontsize=9)
            figure.colorbar(
                colored,
                cax=axes[row, -1],
                label=planes_and_units[0][1],
            )
        figure.suptitle(group_title)
        figure.subplots_adjust(top=0.94, hspace=0.35, wspace=0.3)
        _save_both(figure, output / "generated" / f"mapas_{filename}")
        plt.close(figure)


def write_report(
    output: Path,
    results: Sequence[KohonenRunResult],
    interpretation: Sequence[dict],
    campaigns: Sequence[dict],
) -> None:
    rows = "\n".join(
        "| {} | {} | {:.4f} | {}/{} | {} |".format(
            _profile_for_run(result.run),
            result.run.config.seed,
            result.quantization_error,
            result.occupied_units,
            result.hit_map.size,
            int(result.hit_map.max()),
        )
        for result in results
    )
    interpretation_rows = "\n".join(
        "| {profile} | {occupied_units} | {mean_dominant_topic_pct:.1f} | "
        "{maximum_football_pct:.1f} | {maximum_makeup_pct:.1f} | "
        "{maximum_dogs_pct:.1f} | {maximum_art_pct:.1f} |".format(**row)
        for row in interpretation
    )
    selected_campaigns = [
        row for row in campaigns if row["profile"] == "k5-constant-global"
    ]
    campaign_rows = "\n".join(
        "| {campaign} | ({row}, {column}) | {hits} | `{creator_ids}` | "
        "{topic_pct:.1f} | {estimated_argentina_reach:.0f} | "
        "{total_engagement_per_1000:.1f} | {profile_visits_per_1000:.1f} | "
        "{price_usd:.0f} |".format(**row)
        for row in selected_campaigns
    )
    report = f"""# Experimento 7: interpretación de las candidatas

Se comparan las cuatro candidatas `k = 4` y `k = 5` con 150 épocas e
inicialización `sample`. Para cada configuración se eligió la semilla más
central respecto de error, unidades muertas y concentración máxima; no se
eligió la mejor corrida.

| Configuración | Semilla | Error | Ocupadas | Máximo de hits |
| --- | ---: | ---: | ---: | ---: |
{rows}

Todos los mapas de una misma variable comparten escala entre configuraciones.
Las 15 entradas se muestran como pesos estandarizados; arte, showcase y alcance
argentino se reconstruyen en unidades originales, mientras que followers es un
promedio observado y no participa del entrenamiento. Una rotación o reflexión
de la grilla no cambia su significado: se comparan regiones, no coordenadas.

![Hits y matrices U](generated/hits_y_matrices_u.png)

![Exposición](generated/mapas_exposicion.png)

![Respuesta comercial](generated/mapas_respuesta_comercial.png)

![Temas](generated/mapas_temas.png)

![Contenido](generated/mapas_contenido.png)

![Consumo y auxiliares](generated/mapas_consumo_auxiliares.png)

![Creadores asignados](generated/creadores_asignados.png)

## Resolución temática

| Configuración | Ocupadas | Tema dominante medio (%) | Máx. fútbol | Máx. belleza | Máx. perros | Máx. arte |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{interpretation_rows}

Las dos grillas `k = 5` generan prototipos temáticos más especializados: el
tema dominante ronda 49 %, frente a 41–45 % en `k = 4`. La resolución adicional
no se limita a bajar el error: separa zonas continuas de fútbol, belleza,
perros y arte, junto con gradientes de alcance, respuesta y contenido.

## Decisión

Se elige **`k5-constant-global`**: grilla `5 × 5`, tasa constante `0.2`, radio
inicial igual al diámetro y descenso hasta 1, 150 épocas e inicialización
`sample`. Frente a `k = 4`, reduce el error cerca de 9 % y conserva 24 de 25
neuronas ocupadas en la corrida representativa. Frente a `k5-decay-wide`, pierde
menos de 1 % en error, pero tiene menor concentración y variabilidad entre
semillas, una sola celda vacía en el mapa mostrado y regiones algo más limpias.

La semilla 2 sólo se usa para visualizar el caso central; la decisión se apoya
en las cinco semillas y no fija una “semilla óptima”.

### Regiones comerciales visibles en la configuración elegida

| Campaña | Neurona | Hits | Creadores | Afinidad temática (%) | Alcance argentino estimado | Engagement/1000 | Visitas/1000 | Precio USD |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
{campaign_rows}

Estas celdas son puntos de partida interpretativos, no un ranking automático:
para recomendar creadores concretos todavía deben leerse sus valores reales y
compararse con el baseline de followers.

Los valores completos están en `generated/perfiles_neuronas.csv`; las
asignaciones, en `generated/asignaciones_finalistas.csv`; y el criterio para
elegir las semillas, en `generated/seleccion_representantes.csv`. Los resúmenes
cuantitativos están en `generated/resumen_interpretabilidad.csv` y
`generated/regiones_campanas.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_creator_dataset(args.data)
    runs = load_interpretation_runs(args.configs)
    metrics, all_results = evaluate_runs(dataset, runs)
    selected = select_representatives(metrics)
    results = [all_results[metric.run] for metric in selected]
    args.output.mkdir(parents=True, exist_ok=True)
    interpretation, campaigns = write_tables(
        args.output, dataset, metrics, results
    )
    write_hit_u_plot(args.output, results)
    write_creator_plot(args.output, dataset, results)
    write_feature_groups(args.output, dataset, results)
    write_report(args.output, results, interpretation, campaigns)
    print(f"Estudio interpretativo generado en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
