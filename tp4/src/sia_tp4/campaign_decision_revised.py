"""Selección de creadores guiada por el objetivo de campaña, no por regiones."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen_stability import (
    evaluate_seeds,
    load_design as load_stability_design,
    summarize_creator_stability,
)


STRATEGY_LABELS = {
    "propuesta": "Afinidad + descubrimiento",
    "followers_elegibles": "Followers entre afines",
    "followers_global": "Followers global",
}


@dataclass(frozen=True)
class Campaign:
    name: str
    label: str
    client: str
    business_goal: str
    topic_feature: str


@dataclass(frozen=True)
class DecisionDesign:
    name: str
    affinity_threshold_pct: float
    shortlist_size: int
    kohonen_stability_config: Path
    campaigns: tuple[Campaign, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datos/creadores.csv"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/campaigns/decision_revised.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/campanas-creadores-revisado"),
    )
    return parser.parse_args(argv)


def load_design(path: Path | str) -> DecisionDesign:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "name",
        "affinity_threshold_pct",
        "shortlist_size",
        "kohonen_stability_config",
        "campaigns",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("la configuración de campañas tiene campos inválidos")
    threshold = float(raw["affinity_threshold_pct"])
    if not 0 < threshold <= 100:
        raise ValueError("affinity_threshold_pct debe estar entre 0 y 100")
    shortlist_size = int(raw["shortlist_size"])
    if shortlist_size <= 0:
        raise ValueError("shortlist_size debe ser positivo")
    documents = raw["campaigns"]
    if not isinstance(documents, list) or len(documents) != 4:
        raise ValueError("se requieren exactamente cuatro campañas")
    campaigns = tuple(Campaign(**document) for document in documents)
    if len({campaign.name for campaign in campaigns}) != len(campaigns):
        raise ValueError("los nombres de campaña deben ser únicos")
    stability = path.parent / Path(raw["kohonen_stability_config"])
    if not stability.is_file():
        raise ValueError("no se encontró la configuración de estabilidad")
    return DecisionDesign(
        name=str(raw["name"]),
        affinity_threshold_pct=threshold,
        shortlist_size=shortlist_size,
        kohonen_stability_config=stability,
        campaigns=campaigns,
    )


def load_raw_rows(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("el catálogo está vacío")
    numeric = [column for column in rows[0] if column not in {"creator_id", "name"}]
    parsed = []
    for row in rows:
        parsed.append(
            {
                "creator_id": row["creator_id"],
                "name": row["name"],
                **{column: float(row[column]) for column in numeric},
            }
        )
    return parsed


def campaign_rows(
    raw_rows: Sequence[dict],
    stability_rows: Sequence[dict],
    campaign: Campaign,
    affinity_threshold_pct: float,
) -> list[dict]:
    """Calcular proxies transparentes para todos los creadores elegibles."""

    stability_by_id = {row["creator_id"]: row for row in stability_rows}
    rows = []
    for source in raw_rows:
        affinity = float(source[campaign.topic_feature])
        if affinity < affinity_threshold_pct:
            continue
        relevant_reach = source["estimated_argentina_reach"] * affinity / 100
        new_relevant_reach = (
            relevant_reach * source["non_follower_reach_pct"] / 100
        )
        relevant_profile_visits = (
            relevant_reach * source["profile_visits_per_1000"] / 1000
        )
        new_relevant_profile_visits = (
            new_relevant_reach * source["profile_visits_per_1000"] / 1000
        )
        stability = stability_by_id[source["creator_id"]]
        modal_regions = stability["modal_region"].split(";")
        topic_region = campaign.topic_feature.removeprefix("topic_").removesuffix(
            "_pct"
        )
        rows.append(
            {
                "campaign": campaign.name,
                "campaign_label": campaign.label,
                "client": campaign.client,
                "business_goal": campaign.business_goal,
                "creator_id": source["creator_id"],
                "name": source["name"],
                "eligible_by_affinity": True,
                "affinity_threshold_pct": affinity_threshold_pct,
                "topic_feature": campaign.topic_feature,
                "topic_affinity_pct": affinity,
                "estimated_relevant_reach": relevant_reach,
                "estimated_new_relevant_reach": new_relevant_reach,
                "estimated_relevant_profile_visits": relevant_profile_visits,
                "estimated_new_relevant_profile_visits": new_relevant_profile_visits,
                "cost_per_1000_new_relevant_reach": (
                    1000 * source["price_usd"] / new_relevant_reach
                    if new_relevant_reach > 0
                    else float("inf")
                ),
                "new_relevant_reach_per_usd": (
                    new_relevant_reach / source["price_usd"]
                    if source["price_usd"] > 0
                    else float("inf")
                ),
                "modal_region": stability["modal_region"],
                "kohonen_region_matches_topic": topic_region in modal_regions,
                "region_stability_class": stability["region_stability_class"],
                "region_stability_pct": stability["region_stability_pct"],
                **source,
            }
        )
    if len(rows) < 3:
        raise ValueError(f"la campaña {campaign.name} tiene menos de tres elegibles")
    return sorted(
        rows,
        key=lambda row: (
            -row["estimated_new_relevant_reach"],
            -row["estimated_new_relevant_profile_visits"],
            row["price_usd"],
            row["creator_id"],
        ),
    )


def build_all_campaign_rows(
    raw_rows: Sequence[dict],
    stability_rows: Sequence[dict],
    design: DecisionDesign,
) -> list[dict]:
    rows = []
    for campaign in design.campaigns:
        rows.extend(
            campaign_rows(
                raw_rows,
                stability_rows,
                campaign,
                design.affinity_threshold_pct,
            )
        )
    return rows


def select_shortlists(
    raw_rows: Sequence[dict],
    eligible_rows: Sequence[dict],
    design: DecisionDesign,
) -> list[dict]:
    """Comparar la propuesta con dos baselines de followers usando el mismo K."""

    raw_by_id = {row["creator_id"]: row for row in raw_rows}
    global_followers = sorted(
        raw_rows, key=lambda row: (-row["followers"], row["creator_id"])
    )
    output = []
    for campaign in design.campaigns:
        pool = [row for row in eligible_rows if row["campaign"] == campaign.name]
        pool_by_id = {row["creator_id"]: row for row in pool}
        proposal = pool[: design.shortlist_size]
        eligible_followers = sorted(
            pool, key=lambda row: (-row["followers"], row["creator_id"])
        )[: design.shortlist_size]
        global_ids = [
            row["creator_id"] for row in global_followers[: design.shortlist_size]
        ]
        strategies = {
            "propuesta": proposal,
            "followers_elegibles": eligible_followers,
            "followers_global": [
                _metrics_for_any_creator(
                    raw_by_id[creator_id], campaign, design.affinity_threshold_pct
                )
                for creator_id in global_ids
            ],
        }
        proposal_ids = {row["creator_id"] for row in proposal}
        for strategy, members in strategies.items():
            for rank, row in enumerate(members, start=1):
                enriched = dict(row)
                if row["creator_id"] in pool_by_id:
                    enriched.update(
                        {
                            key: pool_by_id[row["creator_id"]][key]
                            for key in (
                                "modal_region",
                                "kohonen_region_matches_topic",
                                "region_stability_class",
                                "region_stability_pct",
                            )
                        }
                    )
                output.append(
                    {
                        "strategy": strategy,
                        "strategy_label": STRATEGY_LABELS[strategy],
                        "rank": rank,
                        "in_proposed_shortlist": row["creator_id"] in proposal_ids,
                        **enriched,
                    }
                )
    return output


def build_threshold_sensitivity(
    raw_rows: Sequence[dict],
    stability_rows: Sequence[dict],
    design: DecisionDesign,
    thresholds: Sequence[float] = (40.0, 50.0, 60.0),
) -> list[dict]:
    """Comprobar si la shortlist depende del corte de afinidad elegido."""

    rows = []
    for campaign in design.campaigns:
        for threshold in thresholds:
            ranked = campaign_rows(
                raw_rows, stability_rows, campaign, float(threshold)
            )
            for rank, candidate in enumerate(
                ranked[: design.shortlist_size], start=1
            ):
                rows.append(
                    {
                        "campaign": campaign.name,
                        "campaign_label": campaign.label,
                        "affinity_threshold_pct": float(threshold),
                        "eligible_count": len(ranked),
                        "rank": rank,
                        "creator_id": candidate["creator_id"],
                        "name": candidate["name"],
                        "topic_affinity_pct": candidate["topic_affinity_pct"],
                        "estimated_new_relevant_reach": candidate[
                            "estimated_new_relevant_reach"
                        ],
                    }
                )
    return rows


def _metrics_for_any_creator(
    source: dict, campaign: Campaign, affinity_threshold_pct: float
) -> dict:
    affinity = float(source[campaign.topic_feature])
    relevant_reach = source["estimated_argentina_reach"] * affinity / 100
    new_relevant_reach = relevant_reach * source["non_follower_reach_pct"] / 100
    return {
        "campaign": campaign.name,
        "campaign_label": campaign.label,
        "client": campaign.client,
        "business_goal": campaign.business_goal,
        "creator_id": source["creator_id"],
        "name": source["name"],
        "eligible_by_affinity": affinity >= affinity_threshold_pct,
        "affinity_threshold_pct": affinity_threshold_pct,
        "topic_feature": campaign.topic_feature,
        "topic_affinity_pct": affinity,
        "estimated_relevant_reach": relevant_reach,
        "estimated_new_relevant_reach": new_relevant_reach,
        "estimated_relevant_profile_visits": (
            relevant_reach * source["profile_visits_per_1000"] / 1000
        ),
        "estimated_new_relevant_profile_visits": (
            new_relevant_reach * source["profile_visits_per_1000"] / 1000
        ),
        "cost_per_1000_new_relevant_reach": (
            1000 * source["price_usd"] / new_relevant_reach
            if new_relevant_reach > 0
            else float("inf")
        ),
        "new_relevant_reach_per_usd": (
            new_relevant_reach / source["price_usd"]
            if source["price_usd"] > 0
            else float("inf")
        ),
        **source,
    }


def summarize_strategies(
    shortlist_rows: Sequence[dict], design: DecisionDesign
) -> list[dict]:
    summaries = []
    for campaign in design.campaigns:
        for strategy in STRATEGY_LABELS:
            members = [
                row
                for row in shortlist_rows
                if row["campaign"] == campaign.name and row["strategy"] == strategy
            ]
            summaries.append(
                {
                    "campaign": campaign.name,
                    "campaign_label": campaign.label,
                    "strategy": strategy,
                    "strategy_label": STRATEGY_LABELS[strategy],
                    "k": len(members),
                    "eligible_count": sum(
                        row["eligible_by_affinity"] for row in members
                    ),
                    "mean_topic_affinity_pct": float(
                        np.mean([row["topic_affinity_pct"] for row in members])
                    ),
                    "sum_estimated_new_relevant_reach": float(
                        sum(row["estimated_new_relevant_reach"] for row in members)
                    ),
                    "sum_estimated_new_relevant_profile_visits": float(
                        sum(
                            row["estimated_new_relevant_profile_visits"]
                            for row in members
                        )
                    ),
                    "total_price_usd": float(sum(row["price_usd"] for row in members)),
                    "mean_profile_visits_per_1000": float(
                        np.mean([row["profile_visits_per_1000"] for row in members])
                    ),
                }
            )
    return summaries


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"no hay filas para {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output: Path,
    eligible_rows: Sequence[dict],
    shortlist_rows: Sequence[dict],
    summaries: Sequence[dict],
    sensitivity_rows: Sequence[dict],
    design: DecisionDesign,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "candidatos_elegibles.csv", eligible_rows)
    _write_csv(generated / "shortlists_comparadas.csv", shortlist_rows)
    _write_csv(generated / "comparacion_estrategias.csv", summaries)
    _write_csv(generated / "sensibilidad_umbral.csv", sensitivity_rows)
    _write_report(
        output,
        eligible_rows,
        shortlist_rows,
        summaries,
        sensitivity_rows,
        design,
    )
    _write_plots(generated, eligible_rows, summaries, design)


def _write_report(
    output: Path,
    eligible_rows: Sequence[dict],
    shortlist_rows: Sequence[dict],
    summaries: Sequence[dict],
    sensitivity_rows: Sequence[dict],
    design: DecisionDesign,
) -> None:
    sensitivity_table = []
    for campaign in design.campaigns:
        cells = []
        for threshold in (40.0, 50.0, 60.0):
            names = [
                row["name"]
                for row in sensitivity_rows
                if row["campaign"] == campaign.name
                and row["affinity_threshold_pct"] == threshold
            ]
            cells.append(", ".join(names))
        sensitivity_table.append(
            f"| {campaign.label} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    sections = []
    for campaign in design.campaigns:
        candidates = [
            row
            for row in shortlist_rows
            if row["campaign"] == campaign.name and row["strategy"] == "propuesta"
        ]
        candidate_table = "\n".join(
            "| {rank} | {name} | {affinity:.1f} % | {reach:,.0f} | {visits:,.1f} | "
            "USD {price:,.0f} | {region} |".format(
                rank=row["rank"],
                name=row["name"],
                affinity=row["topic_affinity_pct"],
                reach=row["estimated_new_relevant_reach"],
                visits=row["estimated_new_relevant_profile_visits"],
                price=row["price_usd"],
                region=(
                    "coincide"
                    if row.get("kohonen_region_matches_topic")
                    else "no coincide"
                ),
            )
            for row in candidates
        )
        strategy_table = []
        for strategy in STRATEGY_LABELS:
            row = next(
                item
                for item in summaries
                if item["campaign"] == campaign.name
                and item["strategy"] == strategy
            )
            strategy_table.append(
                "| {label} | {eligible}/{k} | {affinity:.1f} % | {reach:,.0f} | "
                "{visits:,.1f} | USD {price:,.0f} |".format(
                    label=row["strategy_label"],
                    eligible=row["eligible_count"],
                    k=row["k"],
                    affinity=row["mean_topic_affinity_pct"],
                    reach=row["sum_estimated_new_relevant_reach"],
                    visits=row["sum_estimated_new_relevant_profile_visits"],
                    price=row["total_price_usd"],
                )
            )
        sections.append(
            f"""## {campaign.label}: {campaign.client}

Objetivo del cliente: **{campaign.business_goal}**. Como el dataset no contiene
ventas, consultas ni adopciones, la selección maximiza el alcance nuevo y afín
como indicador principal. Las visitas al perfil se muestran como señal secundaria
de intención, no como una conversión observada.

| Puesto | Creador | Afinidad | Alcance nuevo y afín estimado | Visitas nuevas y afines estimadas | Precio | Región Kohonen |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
{candidate_table}

| Estrategia | Afines | Afinidad media | Alcance nuevo y afín | Visitas nuevas y afines | Costo total |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(strategy_table)}

La región de Kohonen se informa para interpretar el perfil. No decide la
elegibilidad ni el orden de los candidatos.
"""
        )
    report = f"""# Decisión revisada para las cuatro campañas

## Qué se corrigió

La selección anterior convertía la región temática modal de Kohonen en un filtro
obligatorio. Eso podía excluir a un creador con alta afinidad individual cuando
las otras variables lo ubicaban en otra zona del mapa. La revisión separa las
dos preguntas:

1. Kohonen describe perfiles multivariados y ayuda a interpretar similitudes.
2. La campaña selecciona directamente según el tema y el objetivo del cliente.

## Regla de selección

1. Un creador es elegible si al menos **{design.affinity_threshold_pct:.0f} %** de
   su contenido corresponde al tema de la campaña.
2. Se estima el alcance argentino afín como `alcance argentino × afinidad`.
3. Se estima el alcance nuevo y afín como el valor anterior multiplicado por el
   porcentaje de alcance a no seguidores.
4. Se ordenan los elegibles por alcance nuevo y afín y se conservan los primeros
   **{design.shortlist_size}**. Ante empate se priorizan más visitas al perfil y
   luego menor precio.

Los productos entre porcentajes son proxies descriptivos. Suponen que la mezcla
temática y las tasas generales del creador se mantienen en la publicación de la
campaña. No predicen ventas ni adopciones.

## Sensibilidad al umbral temático

El corte de 50 % significa que el tema ocupa la mayoría del contenido. Para
comprobar que la conclusión no dependa de ese único número, se repitió la
selección con cortes de 40 %, 50 % y 60 %. Los tres candidatos se conservan en
las cuatro campañas.

| Campaña | Umbral 40 % | Umbral 50 % | Umbral 60 % |
| --- | --- | --- | --- |
{chr(10).join(sensitivity_table)}

## Comparaciones

Se usan tres listas del mismo tamaño:

- **Afinidad + descubrimiento:** la propuesta revisada.
- **Followers entre afines:** controla el tema y luego ordena por followers.
- **Followers global:** representa la práctica actual de la agencia, sin filtro
  temático.

![Comparación de estrategias](generated/comparacion_estrategias.png)

![Candidatos y trade-offs](generated/candidatos_tradeoffs.png)

{''.join(sections)}
## Lectura correcta de Kohonen

La región modal y su estabilidad siguen siendo evidencia útil para describir
perfiles híbridos y comprobar que la interpretación del mapa no depende de una
sola semilla. No se usan como etiqueta verdadera ni como condición de entrada a
una campaña. Una discrepancia entre afinidad individual y región es información
para explicar, no un motivo automático de exclusión.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def _write_plots(
    generated: Path,
    eligible_rows: Sequence[dict],
    summaries: Sequence[dict],
    design: DecisionDesign,
) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("mean_topic_affinity_pct", "Afinidad temática media (%)"),
        ("sum_estimated_new_relevant_reach", "Alcance nuevo y afín estimado"),
        (
            "sum_estimated_new_relevant_profile_visits",
            "Visitas nuevas y afines estimadas",
        ),
        ("total_price_usd", "Costo total (USD)"),
    )
    colors = {
        "propuesta": "#25639A",
        "followers_elegibles": "#2A9D8F",
        "followers_global": "#E67E22",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    x = np.arange(len(design.campaigns))
    width = 0.24
    for axis, (metric, title) in zip(axes.flat, metrics):
        for index, strategy in enumerate(STRATEGY_LABELS):
            values = [
                next(
                    row[metric]
                    for row in summaries
                    if row["campaign"] == campaign.name
                    and row["strategy"] == strategy
                )
                for campaign in design.campaigns
            ]
            axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=STRATEGY_LABELS[strategy],
                color=colors[strategy],
            )
        axis.set_title(title)
        axis.set_xticks(x, [campaign.label for campaign in design.campaigns])
        axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(generated / "comparacion_estrategias.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, campaign in zip(axes.flat, design.campaigns):
        rows = [row for row in eligible_rows if row["campaign"] == campaign.name]
        selected_ids = {row["creator_id"] for row in rows[: design.shortlist_size]}
        for row in rows:
            selected = row["creator_id"] in selected_ids
            axis.scatter(
                row["price_usd"],
                row["estimated_new_relevant_reach"],
                s=70 if selected else 28,
                color="#25639A" if selected else "#B8C2CC",
                alpha=0.9 if selected else 0.6,
            )
            if selected:
                right_side = row["price_usd"] > 0.72 * max(
                    member["price_usd"] for member in rows
                )
                axis.annotate(
                    row["name"],
                    (row["price_usd"], row["estimated_new_relevant_reach"]),
                    xytext=(-4 if right_side else 4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    ha="right" if right_side else "left",
                )
        axis.set_title(campaign.label)
        axis.set_xlabel("Precio por publicación (USD)")
        axis.set_ylabel("Alcance nuevo y afín estimado")
        axis.grid(alpha=0.25)
        axis.margins(x=0.1, y=0.12)
    figure.tight_layout()
    figure.savefig(generated / "candidatos_tradeoffs.png", dpi=180)
    plt.close(figure)


def run_analysis(data: Path, config: Path, output: Path) -> list[dict]:
    design = load_design(config)
    raw_rows = load_raw_rows(data)
    dataset: CreatorDataset = load_creator_dataset(data)
    stability_design = load_stability_design(design.kohonen_stability_config)
    stability_rows = summarize_creator_stability(
        dataset,
        evaluate_seeds(dataset, stability_design),
        stability_design,
    )
    eligible_rows = build_all_campaign_rows(raw_rows, stability_rows, design)
    shortlist_rows = select_shortlists(raw_rows, eligible_rows, design)
    summaries = summarize_strategies(shortlist_rows, design)
    sensitivity_rows = build_threshold_sensitivity(
        raw_rows, stability_rows, design
    )
    write_outputs(
        output,
        eligible_rows,
        shortlist_rows,
        summaries,
        sensitivity_rows,
        design,
    )
    return shortlist_rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows = run_analysis(args.data, args.config, args.output)
    print(
        f"Análisis revisado completo: {len(rows)} filas comparadas; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
