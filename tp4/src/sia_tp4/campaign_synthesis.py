"""Integrar regiones de Kohonen, datos originales, PCA y Oja por campaña."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.lines import Line2D

from .data import BASE_FEATURES, CreatorDataset, load_creator_dataset
from .kohonen_stability import (
    TOPICS,
    evaluate_seeds,
    load_design as load_stability_design,
    summarize_creator_stability,
)
from .oja_analysis import (
    fit_final_runs,
    load_design as load_oja_design,
    select_representative,
)
from .pca import PCAResult, fit_pca
from .pca_analysis import _load_pyplot, _save_figure


STABILITY_CLASSES = ("estable", "fronterizo", "inestable")
STABILITY_MARKERS = {"estable": "o", "fronterizo": "^", "inestable": "X"}
CAMPAIGN_COLORS = {
    "futbol": "#4C78A8",
    "belleza": "#E45756",
    "perros": "#59A14F",
    "arte": "#F28E2B",
}
RAW_FEATURES = (
    "followers",
    "average_reach",
    "average_impressions",
    "average_video_views",
    "monthly_posts",
    "argentina_audience_pct",
    "estimated_argentina_reach",
    "non_follower_reach_pct",
    "likes_per_1000",
    "comments_per_1000",
    "saves_per_1000",
    "shares_per_1000",
    "total_engagement_per_1000",
    "profile_visits_per_1000",
    "price_usd",
    "price_ars",
    "sponsored_posts_pct",
    "organic_posts_pct",
    "topic_football_pct",
    "topic_makeup_pct",
    "topic_dogs_pct",
    "topic_art_pct",
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "content_showcase_pct",
    "average_caption_words",
    "average_watch_time_seconds",
    "video_posts_pct",
    "short_form_posts_pct",
    "trend_posts_pct",
)
SUMMARY_FEATURES = (
    "followers",
    "average_reach",
    "estimated_argentina_reach",
    "non_follower_reach_pct",
    "total_engagement_per_1000",
    "profile_visits_per_1000",
    "price_usd",
    "sponsored_posts_pct",
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
)


@dataclass(frozen=True)
class CampaignDefinition:
    """Relación explícita entre una campaña y una región temática."""

    name: str
    label: str
    region: str
    topic_feature: str


@dataclass(frozen=True)
class SynthesisDesign:
    """Fuentes reproducibles y regla de inclusión primaria."""

    name: str
    kohonen_stability_config: Path
    oja_config: Path
    primary_stability_classes: tuple[str, ...]
    campaigns: tuple[CampaignDefinition, ...]


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
        default=Path("configs/campaigns/synthesis.json"),
        help="Diseño de integración para las cuatro campañas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/campanas-creadores"),
        help="Directorio único del informe y resultados agregados.",
    )
    return parser.parse_args(argv)


def load_design(path: Path | str) -> SynthesisDesign:
    """Validar campañas, regiones, estabilidad y referencias de configuración."""

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON inválido en {path}: {error.msg}") from error
    if not isinstance(raw, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {
        "name",
        "kohonen_stability_config",
        "oja_config",
        "primary_stability_classes",
        "campaigns",
    }
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown or missing:
        raise ValueError(
            "campos inválidos en la configuración: "
            f"faltan={sorted(missing)}, sobran={sorted(unknown)}"
        )
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError("name debe ser un texto no vacío")

    primary = raw["primary_stability_classes"]
    if (
        not isinstance(primary, list)
        or not primary
        or len(primary) != len(set(primary))
        or not set(primary).issubset(STABILITY_CLASSES)
    ):
        raise ValueError("primary_stability_classes contiene valores inválidos")
    campaign_documents = raw["campaigns"]
    if not isinstance(campaign_documents, list) or len(campaign_documents) != 4:
        raise ValueError("se requieren exactamente cuatro campañas")
    campaigns = []
    for document in campaign_documents:
        if not isinstance(document, dict) or set(document) != {
            "name",
            "label",
            "region",
            "topic_feature",
        }:
            raise ValueError("cada campaña debe declarar nombre, etiqueta, región y tema")
        if any(not isinstance(document[key], str) or not document[key] for key in document):
            raise ValueError("los campos de campaña deben ser textos no vacíos")
        campaigns.append(CampaignDefinition(**document))
    if len({item.name for item in campaigns}) != 4:
        raise ValueError("los nombres de campaña deben ser únicos")
    if {item.region for item in campaigns} != set(TOPICS):
        raise ValueError("las campañas deben cubrir las cuatro regiones temáticas")
    if any(item.topic_feature not in RAW_FEATURES for item in campaigns):
        raise ValueError("cada topic_feature debe existir en los datos originales")

    base = path.parent
    stability_path = base / Path(raw["kohonen_stability_config"])
    oja_path = base / Path(raw["oja_config"])
    if not stability_path.is_file() or not oja_path.is_file():
        raise ValueError("no se encontraron las configuraciones referenciadas")
    return SynthesisDesign(
        name=raw["name"],
        kohonen_stability_config=stability_path,
        oja_config=oja_path,
        primary_stability_classes=tuple(primary),
        campaigns=tuple(campaigns),
    )


def load_original_rows(
    path: Path | str, dataset: CreatorDataset
) -> dict[str, dict[str, float]]:
    """Leer variables originales sin alterar el CSV fuente ni sus unidades."""

    path = Path(path)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("el CSV no tiene encabezado")
        required = {"creator_id", *RAW_FEATURES}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"faltan variables originales: {missing}")
        raw_rows = list(reader)
    if tuple(row["creator_id"].strip() for row in raw_rows) != dataset.creator_ids:
        raise ValueError("el orden o los identificadores no coinciden con el dataset")
    result = {}
    for row in raw_rows:
        creator_id = row["creator_id"].strip()
        try:
            values = {feature: float(row[feature]) for feature in RAW_FEATURES}
        except ValueError as error:
            raise ValueError("las variables originales deben ser numéricas") from error
        if not np.isfinite(tuple(values.values())).all():
            raise ValueError("las variables originales contienen valores no finitos")
        result[creator_id] = values
    return result


def build_creator_rows(
    dataset: CreatorDataset,
    original_rows: dict[str, dict[str, float]],
    stability_rows: Sequence[dict],
    campaigns: Sequence[CampaignDefinition],
    primary_classes: Sequence[str],
    pca_scores: np.ndarray,
    oja_indices: np.ndarray,
) -> list[dict]:
    """Unir todas las evidencias sin producir un ranking compuesto."""

    if pca_scores.shape[0] != len(dataset.creator_ids) or len(oja_indices) != len(
        dataset.creator_ids
    ):
        raise ValueError("PCA, Oja y dataset deben contener los mismos creadores")
    stability_by_id = {row["creator_id"]: row for row in stability_rows}
    campaign_by_region = {campaign.region: campaign for campaign in campaigns}
    rows = []
    for index, (creator_id, name) in enumerate(
        zip(dataset.creator_ids, dataset.names)
    ):
        stability = stability_by_id[creator_id]
        modal_regions = stability["modal_region"].split(";")
        for region in modal_regions:
            campaign = campaign_by_region[region]
            rows.append(
                {
                    "campaign": campaign.name,
                    "campaign_label": campaign.label,
                    "region": region,
                    "creator_id": creator_id,
                    "name": name,
                    "modal_region_tie": stability["modal_region_tie"],
                    **{
                        f"region_{topic}_count": stability[
                            f"region_{topic}_count"
                        ]
                        for topic in TOPICS
                    },
                    "region_stability_class": stability[
                        "region_stability_class"
                    ],
                    "region_stability_count": stability[
                        "region_stability_count"
                    ],
                    "region_stability_pct": stability["region_stability_pct"],
                    "transition_count": stability["transition_count"],
                    "transition_pct": stability["transition_pct"],
                    "primary_candidate": stability[
                        "region_stability_class"
                    ]
                    in primary_classes,
                    "pc1_score": float(pca_scores[index, 0]),
                    "pc2_score": float(pca_scores[index, 1]),
                    "oja_index": float(oja_indices[index]),
                    "oja_pc1_difference": float(
                        oja_indices[index] - pca_scores[index, 0]
                    ),
                    **original_rows[creator_id],
                }
            )
    class_order = {name: index for index, name in enumerate(STABILITY_CLASSES)}
    campaign_order = {
        campaign.name: index for index, campaign in enumerate(campaigns)
    }
    return sorted(
        rows,
        key=lambda row: (
            campaign_order[row["campaign"]],
            class_order[row["region_stability_class"]],
            row["creator_id"],
        ),
    )


def build_profile_rows(
    dataset: CreatorDataset,
    creator_rows: Sequence[dict],
    campaigns: Sequence[CampaignDefinition],
    standardized_values: np.ndarray,
) -> list[dict]:
    """Calcular perfiles regionales en el espacio z-score usado por los modelos."""

    index_by_id = {
        creator_id: index for index, creator_id in enumerate(dataset.creator_ids)
    }
    rows = []
    for campaign in campaigns:
        members = [row for row in creator_rows if row["campaign"] == campaign.name]
        indexes = [index_by_id[row["creator_id"]] for row in members]
        region_values = standardized_values[indexes]
        for feature_index, feature in enumerate(dataset.feature_names):
            rows.append(
                {
                    "campaign": campaign.name,
                    "campaign_label": campaign.label,
                    "region": campaign.region,
                    "feature": feature,
                    "mean_z": float(np.mean(region_values[:, feature_index])),
                    "std_z": _sample_std(region_values[:, feature_index]),
                }
            )
    return rows


def _sample_std(values: Sequence[float] | np.ndarray) -> float:
    """Desvío muestral; cero cuando un grupo contiene una sola observación."""

    array = np.asarray(values, dtype=float)
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def build_region_summaries(
    creator_rows: Sequence[dict],
    profile_rows: Sequence[dict],
    campaigns: Sequence[CampaignDefinition],
) -> list[dict]:
    """Resumir tamaño, estabilidad, ejes y variables originales por región."""

    rows = []
    for campaign in campaigns:
        members = [row for row in creator_rows if row["campaign"] == campaign.name]
        profiles = [row for row in profile_rows if row["campaign"] == campaign.name]
        ordered_high = sorted(profiles, key=lambda row: row["mean_z"], reverse=True)
        ordered_low = sorted(profiles, key=lambda row: row["mean_z"])
        record = {
            "campaign": campaign.name,
            "campaign_label": campaign.label,
            "region": campaign.region,
            "creator_count": len(members),
            **{
                f"{category}_count": sum(
                    row["region_stability_class"] == category for row in members
                )
                for category in STABILITY_CLASSES
            },
            "primary_candidate_count": sum(row["primary_candidate"] for row in members),
            "stable_creator_ids": ";".join(
                row["creator_id"]
                for row in members
                if row["region_stability_class"] == "estable"
            ),
            "frontier_creator_ids": ";".join(
                row["creator_id"]
                for row in members
                if row["region_stability_class"] == "fronterizo"
            ),
            "unstable_creator_ids": ";".join(
                row["creator_id"]
                for row in members
                if row["region_stability_class"] == "inestable"
            ),
            "pc1_mean": float(np.mean([row["pc1_score"] for row in members])),
            "pc1_std": _sample_std([row["pc1_score"] for row in members]),
            "pc2_mean": float(np.mean([row["pc2_score"] for row in members])),
            "pc2_std": _sample_std([row["pc2_score"] for row in members]),
            "oja_mean": float(np.mean([row["oja_index"] for row in members])),
            "oja_std": _sample_std([row["oja_index"] for row in members]),
            "highest_z_features": ";".join(
                f"{row['feature']}:{row['mean_z']:.3f}" for row in ordered_high[:3]
            ),
            "lowest_z_features": ";".join(
                f"{row['feature']}:{row['mean_z']:.3f}" for row in ordered_low[:3]
            ),
        }
        for feature in SUMMARY_FEATURES:
            record[f"mean__{feature}"] = float(
                np.mean([row[feature] for row in members])
            )
            record[f"median__{feature}"] = float(
                np.median([row[feature] for row in members])
            )
        record["topic_feature"] = campaign.topic_feature
        record["mean_topic_affinity"] = float(
            np.mean([row[campaign.topic_feature] for row in members])
        )
        record["median_topic_affinity"] = float(
            np.median([row[campaign.topic_feature] for row in members])
        )
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
    creator_rows: Sequence[dict],
    profile_rows: Sequence[dict],
    summary_rows: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "creadores_por_campana.csv", creator_rows)
    _write_csv(generated / "perfiles_estandarizados.csv", profile_rows)
    _write_csv(generated / "resumen_regiones.csv", summary_rows)


def write_plots(
    output: Path,
    dataset: CreatorDataset,
    creator_rows: Sequence[dict],
    profile_rows: Sequence[dict],
    campaigns: Sequence[CampaignDefinition],
) -> None:
    """Mostrar perfiles regionales y su relación con PC1-PC2."""

    plt = _load_pyplot()
    generated = output / "generated"
    profile_lookup = {
        (row["campaign"], row["feature"]): row["mean_z"]
        for row in profile_rows
    }
    matrix = np.asarray(
        [
            [profile_lookup[(campaign.name, feature)] for feature in dataset.feature_names]
            for campaign in campaigns
        ]
    )
    limit = float(np.max(np.abs(matrix)))
    figure, axis = plt.subplots(figsize=(13, 4.8))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(
        np.arange(len(dataset.feature_names)), dataset.feature_names, rotation=55, ha="right"
    )
    axis.set_yticks(np.arange(len(campaigns)), [item.label for item in campaigns])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(matrix[row, column]) > limit * 0.55 else "black",
            )
    axis.set_title("Perfil medio de las regiones temáticas")
    figure.colorbar(image, ax=axis, label="Media estandarizada (z-score)")
    figure.tight_layout()
    _save_figure(figure, generated / "perfiles_regiones.svg")

    figure, axis = plt.subplots(figsize=(9, 7))
    for campaign in campaigns:
        for stability_class in STABILITY_CLASSES:
            members = [
                row
                for row in creator_rows
                if row["campaign"] == campaign.name
                and row["region_stability_class"] == stability_class
            ]
            if not members:
                continue
            axis.scatter(
                [row["pc1_score"] for row in members],
                [row["pc2_score"] for row in members],
                color=CAMPAIGN_COLORS[campaign.name],
                marker=STABILITY_MARKERS[stability_class],
                s=58,
                alpha=0.8,
                edgecolors="black" if stability_class != "inestable" else None,
                linewidths=0.4,
            )
            if stability_class == "inestable":
                for row in members:
                    axis.annotate(
                        row["creator_id"],
                        (row["pc1_score"], row["pc2_score"]),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=7,
                    )
    campaign_legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CAMPAIGN_COLORS[item.name],
            markeredgecolor="none",
            label=item.label,
            markersize=8,
        )
        for item in campaigns
    ]
    stability_legend = [
        Line2D(
            [0],
            [0],
            marker=STABILITY_MARKERS[item],
            color="none",
            markerfacecolor="#777777",
            markeredgecolor="black",
            label=item.capitalize(),
            markersize=8,
        )
        for item in STABILITY_CLASSES
    ]
    first_legend = axis.legend(handles=campaign_legend, title="Región", loc="upper left")
    axis.add_artist(first_legend)
    axis.legend(handles=stability_legend, title="Estabilidad", loc="lower right")
    axis.axhline(0, color="#999999", linewidth=0.7)
    axis.axvline(0, color="#999999", linewidth=0.7)
    axis.set(
        title="Regiones temáticas y estabilidad sobre PC1-PC2",
        xlabel="PC1",
        ylabel="PC2",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    _save_figure(figure, generated / "regiones_pca_estabilidad.svg")


def _format_candidates(rows: Sequence[dict], category: str) -> str:
    selected = [
        f"{row['creator_id']} ({row['name']})"
        for row in rows
        if row["region_stability_class"] == category
    ]
    return ", ".join(selected) if selected else "ninguno"


def write_report(
    output: Path,
    design: SynthesisDesign,
    creator_rows: Sequence[dict],
    summary_rows: Sequence[dict],
    pca: PCAResult,
    oja_correlation: float,
) -> None:
    """Explicar cómo se integran las evidencias sin crear un ranking."""

    summary_table = "\n".join(
        "| {campaign_label} | {creator_count} | {estable_count} | "
        "{fronterizo_count} | {inestable_count} | {pc1_mean:+.2f} | "
        "{pc2_mean:+.2f} | {oja_mean:+.2f} |".format(**row)
        for row in summary_rows
    )
    campaign_sections = []
    for campaign in design.campaigns:
        members = [row for row in creator_rows if row["campaign"] == campaign.name]
        summary = next(row for row in summary_rows if row["campaign"] == campaign.name)
        campaign_sections.append(
            f"""### {campaign.label}

- Afinidad temática media: **{summary['mean_topic_affinity']:.1f} %**.
- Variables relativamente altas: `{summary['highest_z_features']}`.
- Variables relativamente bajas: `{summary['lowest_z_features']}`.
- Candidatos regionalmente estables: {_format_candidates(members, 'estable')}.
- Candidatos fronterizos: {_format_candidates(members, 'fronterizo')}.
- Creadores inestables, conservados con advertencia: {_format_candidates(members, 'inestable')}.
"""
        )
    explained = 100 * pca.cumulative_explained_variance[1]
    report = f"""# Síntesis regional para las cuatro campañas

## Criterio

Cada creador se asigna a su región temática modal en las diez semillas de la
configuración final de Kohonen. Se usa la región completa, no la neurona de
máxima afinidad. Los casos estables y fronterizos forman el conjunto primario;
los inestables permanecen visibles, pero con una advertencia explícita.

La tabla maestra conserva las 31 variables numéricas originales en sus unidades,
los conteos de región de las diez semillas, los scores de PC1-PC2 y el índice de
Oja. No se calcula un puntaje compuesto ni se ordena a los creadores como
mejores o peores. `followers` queda como dato auxiliar para la comparación
posterior con la estrategia actual de la agencia.

## Integración de modelos

| Campaña | Creadores | Estables | Fronterizos | Inestables | PC1 medio | PC2 medio | Oja medio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{summary_table}

PC1 y PC2 explican juntas **{explained:.2f} %** de la varianza. El índice de
Oja de la corrida representativa tiene correlación **{oja_correlation:.9f}**
con PC1. Por eso Oja confirma ese eje, pero no se cuenta como una segunda señal
independiente para seleccionar candidatos.

![Regiones sobre PCA](generated/regiones_pca_estabilidad.svg)

## Perfiles regionales

El siguiente gráfico usa las mismas 15 variables estandarizadas de PCA,
Kohonen y Oja. Cada valor es la media z-score de una región; no es una tasa ni
un score comercial.

![Perfiles de regiones](generated/perfiles_regiones.svg)

{''.join(campaign_sections)}
## Alcance de esta etapa

Esta síntesis define conjuntos de candidatos y hace visibles sus perfiles y su
estabilidad. Todavía no elige un ganador ni compara formalmente con el top por
`followers`. La etapa siguiente debe usar estos mismos conjuntos para una
comparación de igual cantidad de creadores y mostrar los trade-offs de alcance,
afinidad, respuesta, formato y costo.

## Archivos reproducibles

- `generated/creadores_por_campana.csv`: unión por creador de estabilidad,
  variables originales, PCA y Oja;
- `generated/resumen_regiones.csv`: tamaños, estabilidad, ejes y medias/medianas;
- `generated/perfiles_estandarizados.csv`: perfil z-score completo de cada región.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def run_analysis(data: Path, config: Path, output: Path) -> list[dict]:
    """Regenerar toda la síntesis desde datos y configuraciones versionadas."""

    dataset = load_creator_dataset(data)
    design = load_design(config)
    original_rows = load_original_rows(data, dataset)

    stability_design = load_stability_design(design.kohonen_stability_config)
    stability_results = evaluate_seeds(dataset, stability_design)
    stability_rows = summarize_creator_stability(
        dataset, stability_results, stability_design
    )

    oja_design = load_oja_design(design.oja_config)
    oja_results, pca = fit_final_runs(dataset, oja_design)
    representative = select_representative(oja_results)
    creator_rows = build_creator_rows(
        dataset,
        original_rows,
        stability_rows,
        design.campaigns,
        design.primary_stability_classes,
        pca.scores,
        representative.indices,
    )
    profile_rows = build_profile_rows(
        dataset, creator_rows, design.campaigns, pca.standardized_values
    )
    summary_rows = build_region_summaries(
        creator_rows, profile_rows, design.campaigns
    )
    write_tables(output, creator_rows, profile_rows, summary_rows)
    write_plots(output, dataset, creator_rows, profile_rows, design.campaigns)
    write_report(
        output,
        design,
        creator_rows,
        summary_rows,
        pca,
        representative.score_correlation,
    )
    return summary_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary_rows = run_analysis(args.data, args.config, args.output)
    print(
        f"Síntesis de campañas completa: {len(summary_rows)} regiones; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
