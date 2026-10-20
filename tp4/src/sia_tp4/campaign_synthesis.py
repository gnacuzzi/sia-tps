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
STRATEGY_LABELS = {
    "regional_kohonen": "Región Kohonen",
    "baseline_followers": "Top followers",
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
class BaselineDefinition:
    """Regla observable y reproducible para la estrategia de referencia."""

    feature: str
    scope: str
    order: str


@dataclass(frozen=True)
class SynthesisDesign:
    """Fuentes reproducibles y regla de inclusión primaria."""

    name: str
    kohonen_stability_config: Path
    oja_config: Path
    primary_stability_classes: tuple[str, ...]
    baseline: BaselineDefinition
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
        "baseline",
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
    baseline = raw["baseline"]
    if not isinstance(baseline, dict) or baseline != {
        "feature": "followers",
        "scope": "global",
        "order": "descending",
    }:
        raise ValueError(
            "el baseline debe ser el top global descendente por followers"
        )
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
        baseline=BaselineDefinition(**baseline),
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


def build_shortlist_comparison(
    dataset: CreatorDataset,
    original_rows: dict[str, dict[str, float]],
    stability_rows: Sequence[dict],
    creator_rows: Sequence[dict],
    campaigns: Sequence[CampaignDefinition],
    pca_scores: np.ndarray,
    oja_indices: np.ndarray,
) -> list[dict]:
    """Comparar cada conjunto regional con un top global de igual tamaño."""

    stability_by_id = {row["creator_id"]: row for row in stability_rows}
    index_by_id = {
        creator_id: index for index, creator_id in enumerate(dataset.creator_ids)
    }
    names_by_id = dict(zip(dataset.creator_ids, dataset.names))
    followers_order = sorted(
        dataset.creator_ids,
        key=lambda creator_id: (
            -original_rows[creator_id]["followers"],
            creator_id,
        ),
    )
    followers_position = {
        creator_id: position
        for position, creator_id in enumerate(followers_order, start=1)
    }
    rows = []
    for campaign in campaigns:
        regional = [
            row
            for row in creator_rows
            if row["campaign"] == campaign.name and row["primary_candidate"]
        ]
        regional_ids = {row["creator_id"] for row in regional}
        k = len(regional_ids)
        if k == 0:
            raise ValueError(f"la campaña {campaign.name} no tiene candidatos primarios")
        baseline_ids = set(followers_order[:k])
        selections = (
            (
                "regional_kohonen",
                sorted(
                    regional_ids,
                    key=lambda creator_id: (
                        STABILITY_CLASSES.index(
                            stability_by_id[creator_id]["region_stability_class"]
                        ),
                        creator_id,
                    ),
                ),
            ),
            ("baseline_followers", followers_order[:k]),
        )
        for strategy, selected_ids in selections:
            other_ids = baseline_ids if strategy == "regional_kohonen" else regional_ids
            for creator_id in selected_ids:
                stability = stability_by_id[creator_id]
                index = index_by_id[creator_id]
                modal_regions = stability["modal_region"].split(";")
                rows.append(
                    {
                        "campaign": campaign.name,
                        "campaign_label": campaign.label,
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "k": k,
                        "creator_id": creator_id,
                        "name": names_by_id[creator_id],
                        "in_both_strategies": creator_id in other_ids,
                        "global_followers_position": followers_position[creator_id],
                        "campaign_region_match": campaign.region in modal_regions,
                        "modal_region": stability["modal_region"],
                        "region_stability_class": stability[
                            "region_stability_class"
                        ],
                        "region_stability_count": stability[
                            "region_stability_count"
                        ],
                        "region_stability_pct": stability["region_stability_pct"],
                        "topic_feature": campaign.topic_feature,
                        "topic_affinity_pct": original_rows[creator_id][
                            campaign.topic_feature
                        ],
                        "pc1_score": float(pca_scores[index, 0]),
                        "pc2_score": float(pca_scores[index, 1]),
                        "oja_index": float(oja_indices[index]),
                        **original_rows[creator_id],
                    }
                )
    return rows


def build_strategy_summaries(
    comparison_rows: Sequence[dict],
    campaigns: Sequence[CampaignDefinition],
) -> list[dict]:
    """Agregar las dos estrategias conservando métricas y costos observados."""

    rows = []
    for campaign in campaigns:
        campaign_rows = [
            row for row in comparison_rows if row["campaign"] == campaign.name
        ]
        for strategy in STRATEGY_LABELS:
            members = [row for row in campaign_rows if row["strategy"] == strategy]
            k = len(members)
            overlap_count = sum(row["in_both_strategies"] for row in members)
            rows.append(
                {
                    "campaign": campaign.name,
                    "campaign_label": campaign.label,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "k": k,
                    "overlap_count": overlap_count,
                    "overlap_pct": 100 * overlap_count / k,
                    "campaign_region_match_count": sum(
                        row["campaign_region_match"] for row in members
                    ),
                    "stable_count": sum(
                        row["region_stability_class"] == "estable"
                        for row in members
                    ),
                    "frontier_count": sum(
                        row["region_stability_class"] == "fronterizo"
                        for row in members
                    ),
                    "unstable_count": sum(
                        row["region_stability_class"] == "inestable"
                        for row in members
                    ),
                    "sum_followers": float(sum(row["followers"] for row in members)),
                    "sum_average_reach": float(
                        sum(row["average_reach"] for row in members)
                    ),
                    "sum_estimated_argentina_reach": float(
                        sum(row["estimated_argentina_reach"] for row in members)
                    ),
                    "mean_argentina_audience_pct": float(
                        np.mean([row["argentina_audience_pct"] for row in members])
                    ),
                    "mean_non_follower_reach_pct": float(
                        np.mean([row["non_follower_reach_pct"] for row in members])
                    ),
                    "mean_total_engagement_per_1000": float(
                        np.mean(
                            [row["total_engagement_per_1000"] for row in members]
                        )
                    ),
                    "mean_profile_visits_per_1000": float(
                        np.mean([row["profile_visits_per_1000"] for row in members])
                    ),
                    "total_price_usd": float(sum(row["price_usd"] for row in members)),
                    "mean_price_usd": float(
                        np.mean([row["price_usd"] for row in members])
                    ),
                    "mean_sponsored_posts_pct": float(
                        np.mean([row["sponsored_posts_pct"] for row in members])
                    ),
                    "mean_topic_affinity_pct": float(
                        np.mean([row["topic_affinity_pct"] for row in members])
                    ),
                    "mean_content_tutorial_pct": float(
                        np.mean([row["content_tutorial_pct"] for row in members])
                    ),
                    "mean_content_review_pct": float(
                        np.mean([row["content_review_pct"] for row in members])
                    ),
                    "mean_content_entertainment_pct": float(
                        np.mean(
                            [row["content_entertainment_pct"] for row in members]
                        )
                    ),
                    "mean_content_showcase_pct": float(
                        np.mean([row["content_showcase_pct"] for row in members])
                    ),
                    "mean_pc1_score": float(
                        np.mean([row["pc1_score"] for row in members])
                    ),
                    "mean_oja_index": float(
                        np.mean([row["oja_index"] for row in members])
                    ),
                }
            )
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
    comparison_rows: Sequence[dict],
    strategy_summaries: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "creadores_por_campana.csv", creator_rows)
    _write_csv(generated / "perfiles_estandarizados.csv", profile_rows)
    _write_csv(generated / "resumen_regiones.csv", summary_rows)
    _write_csv(generated / "shortlists_comparadas.csv", comparison_rows)
    _write_csv(generated / "comparacion_followers.csv", strategy_summaries)


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


def write_strategy_plots(
    output: Path,
    strategy_summaries: Sequence[dict],
    campaigns: Sequence[CampaignDefinition],
) -> None:
    """Comparar estrategias en unidades observadas y mostrar su solapamiento."""

    plt = _load_pyplot()
    generated = output / "generated"
    lookup = {
        (row["campaign"], row["strategy"]): row for row in strategy_summaries
    }
    campaign_names = [campaign.name for campaign in campaigns]
    campaign_labels = [campaign.label for campaign in campaigns]
    positions = np.arange(len(campaigns))
    width = 0.36
    chart_metrics = (
        ("sum_followers", "Suma de followers", 1_000_000, "millones"),
        (
            "sum_estimated_argentina_reach",
            "Suma de alcance argentino estimado",
            1_000,
            "miles",
        ),
        ("mean_topic_affinity_pct", "Afinidad temática media", 1, "%"),
        (
            "mean_total_engagement_per_1000",
            "Engagement medio",
            1,
            "por 1000",
        ),
        (
            "mean_profile_visits_per_1000",
            "Visitas al perfil medias",
            1,
            "por 1000",
        ),
        ("total_price_usd", "Costo total", 1_000, "miles de USD"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 11))
    for axis, (metric, title, divisor, unit) in zip(axes.flat, chart_metrics):
        regional_values = [
            lookup[(campaign, "regional_kohonen")][metric] / divisor
            for campaign in campaign_names
        ]
        baseline_values = [
            lookup[(campaign, "baseline_followers")][metric] / divisor
            for campaign in campaign_names
        ]
        axis.bar(
            positions - width / 2,
            regional_values,
            width,
            color="#4C78A8",
            label=STRATEGY_LABELS["regional_kohonen"],
        )
        axis.bar(
            positions + width / 2,
            baseline_values,
            width,
            color="#B9B9B9",
            label=STRATEGY_LABELS["baseline_followers"],
        )
        axis.set_xticks(positions, campaign_labels)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Regiones temáticas frente al baseline por followers", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(figure, generated / "comparacion_estrategias.svg")

    regional_summaries = [
        lookup[(campaign.name, "regional_kohonen")] for campaign in campaigns
    ]
    overlap_pct = [row["overlap_pct"] for row in regional_summaries]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    bars = axis.barh(campaign_labels, overlap_pct, color="#4C78A8")
    axis.set_xlim(0, 100)
    axis.set_xlabel("Porcentaje de creadores presentes en ambas estrategias")
    axis.set_title("Solapamiento de las shortlists con igual K")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    for bar, row in zip(bars, regional_summaries):
        axis.text(
            min(bar.get_width() + 2, 96),
            bar.get_y() + bar.get_height() / 2,
            f"{row['overlap_count']}/{row['k']}",
            va="center",
            fontsize=9,
        )
    figure.tight_layout()
    _save_figure(figure, generated / "solapamiento_shortlists.svg")


def _format_candidates(rows: Sequence[dict], category: str) -> str:
    selected = [
        f"{row['creator_id']} ({row['name']})"
        for row in rows
        if row["region_stability_class"] == category
    ]
    return ", ".join(selected) if selected else "ninguno"


def _format_ids(rows: Sequence[dict]) -> str:
    return ", ".join(f"{row['creator_id']} ({row['name']})" for row in rows) or "ninguno"


def _percent_change(value: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("no se puede calcular variación porcentual con baseline cero")
    return 100 * (value - baseline) / baseline


def write_report(
    output: Path,
    design: SynthesisDesign,
    creator_rows: Sequence[dict],
    summary_rows: Sequence[dict],
    comparison_rows: Sequence[dict],
    strategy_summaries: Sequence[dict],
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
    strategy_lookup = {
        (row["campaign"], row["strategy"]): row for row in strategy_summaries
    }
    comparison_table_rows = []
    comparison_sections = []
    for campaign in design.campaigns:
        regional = strategy_lookup[(campaign.name, "regional_kohonen")]
        baseline = strategy_lookup[(campaign.name, "baseline_followers")]
        comparison_table_rows.append(
            "| {label} | {k} | {overlap}/{k} ({overlap_pct:.1f} %) | "
            "{regional_affinity:.1f} / {baseline_affinity:.1f} | "
            "{regional_reach:,.0f} / {baseline_reach:,.0f} | "
            "{regional_engagement:.1f} / {baseline_engagement:.1f} | "
            "{regional_cost:,.0f} / {baseline_cost:,.0f} |".format(
                label=campaign.label,
                k=regional["k"],
                overlap=regional["overlap_count"],
                overlap_pct=regional["overlap_pct"],
                regional_affinity=regional["mean_topic_affinity_pct"],
                baseline_affinity=baseline["mean_topic_affinity_pct"],
                regional_reach=regional["sum_estimated_argentina_reach"],
                baseline_reach=baseline["sum_estimated_argentina_reach"],
                regional_engagement=regional["mean_total_engagement_per_1000"],
                baseline_engagement=baseline["mean_total_engagement_per_1000"],
                regional_cost=regional["total_price_usd"],
                baseline_cost=baseline["total_price_usd"],
            )
        )
        campaign_comparison = [
            row for row in comparison_rows if row["campaign"] == campaign.name
        ]
        regional_only = [
            row
            for row in campaign_comparison
            if row["strategy"] == "regional_kohonen"
            and not row["in_both_strategies"]
        ]
        baseline_only = [
            row
            for row in campaign_comparison
            if row["strategy"] == "baseline_followers"
            and not row["in_both_strategies"]
        ]
        comparison_sections.append(
            f"""### {campaign.label}

- Afinidad temática: **{regional['mean_topic_affinity_pct']:.1f} %** regional frente a **{baseline['mean_topic_affinity_pct']:.1f} %** por followers ({regional['mean_topic_affinity_pct'] - baseline['mean_topic_affinity_pct']:+.1f} puntos porcentuales).
- Suma de alcance argentino estimado: **{regional['sum_estimated_argentina_reach']:,.0f}** frente a **{baseline['sum_estimated_argentina_reach']:,.0f}** ({_percent_change(regional['sum_estimated_argentina_reach'], baseline['sum_estimated_argentina_reach']):+.1f} %).
- Engagement medio: **{regional['mean_total_engagement_per_1000']:.1f}** frente a **{baseline['mean_total_engagement_per_1000']:.1f}** por 1000 ({_percent_change(regional['mean_total_engagement_per_1000'], baseline['mean_total_engagement_per_1000']):+.1f} %).
- Costo total: **USD {regional['total_price_usd']:,.0f}** frente a **USD {baseline['total_price_usd']:,.0f}** ({_percent_change(regional['total_price_usd'], baseline['total_price_usd']):+.1f} %).
- Sólo en la shortlist regional: {_format_ids(regional_only)}.
- Sólo en el baseline: {_format_ids(baseline_only)}.
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
mejores o peores. `followers` se usa únicamente para construir el baseline
global de igual tamaño que cada conjunto regional.

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
## Comparación con el baseline por followers

Para cada campaña, `K` es la cantidad de candidatos regionales estables o
fronterizos. El baseline toma los `K` creadores con más followers de todo el
catálogo. De este modo ambas estrategias tienen el mismo tamaño y la diferencia
no se explica por comparar conjuntos de distinta cantidad.

| Campaña | K | Solapamiento | Afinidad regional / baseline (%) | Alcance argentino regional / baseline | Engagement regional / baseline (por 1000) | Costo regional / baseline (USD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_table_rows)}

Las sumas de followers, alcance y alcance argentino son volúmenes descriptivos
no deduplicados: el dataset no permite conocer cuánta audiencia comparten dos
creadores. El costo es la suma de `price_usd` de los integrantes de cada
shortlist.

![Comparación de estrategias](generated/comparacion_estrategias.svg)

![Solapamiento](generated/solapamiento_shortlists.svg)

{''.join(comparison_sections)}
## Alcance de esta etapa

Esta síntesis define conjuntos de candidatos y hace visibles sus perfiles y su
estabilidad. La comparación muestra diferencias observadas de composición,
alcance, afinidad, interacción, formato y costo. No estima ventas, adopciones ni
retorno de inversión porque el dataset no contiene esas variables objetivo.
Tampoco elige automáticamente un ganador: la estrategia apropiada depende del
objetivo declarado para cada campaña.

## Archivos reproducibles

- `generated/creadores_por_campana.csv`: unión por creador de estabilidad,
  variables originales, PCA y Oja;
- `generated/resumen_regiones.csv`: tamaños, estabilidad, ejes y medias/medianas;
- `generated/perfiles_estandarizados.csv`: perfil z-score completo de cada región.
- `generated/shortlists_comparadas.csv`: comparación individual con igual `K`;
- `generated/comparacion_followers.csv`: resultados agregados por estrategia.
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
    comparison_rows = build_shortlist_comparison(
        dataset,
        original_rows,
        stability_rows,
        creator_rows,
        design.campaigns,
        pca.scores,
        representative.indices,
    )
    strategy_summaries = build_strategy_summaries(
        comparison_rows, design.campaigns
    )
    write_tables(
        output,
        creator_rows,
        profile_rows,
        summary_rows,
        comparison_rows,
        strategy_summaries,
    )
    write_plots(output, dataset, creator_rows, profile_rows, design.campaigns)
    write_strategy_plots(output, strategy_summaries, design.campaigns)
    write_report(
        output,
        design,
        creator_rows,
        summary_rows,
        comparison_rows,
        strategy_summaries,
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
