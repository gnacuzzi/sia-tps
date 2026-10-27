#!/usr/bin/env python3
"""Seleccionar un creador por campaña usando primero PCA/Oja y Kohonen."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


CAMPAIGNS = (
    {
        "name": "futbol",
        "label": "Fútbol",
        "topic": "topic_football_pct",
        "pc1_side": "high",
        "objective": "ventas y descubrimiento",
        "decision_metric": "estimated_new_relevant_reach",
    },
    {
        "name": "belleza",
        "label": "Belleza",
        "topic": "topic_makeup_pct",
        "pc1_side": "high",
        "objective": "ventas y descubrimiento",
        "decision_metric": "estimated_new_relevant_reach",
    },
    {
        "name": "perros",
        "label": "Perros",
        "topic": "topic_dogs_pct",
        "pc1_side": "low",
        "objective": "consultas y adopciones",
        "decision_metric": "estimated_new_relevant_profile_visits",
    },
    {
        "name": "arte",
        "label": "Arte",
        "topic": "topic_art_pct",
        "pc1_side": "high",
        "objective": "ventas y descubrimiento",
        "decision_metric": "estimated_new_relevant_reach",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/campanas-creadores/generated/creadores_por_campana.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/campanas-seleccion-guiada-modelos"),
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("la tabla integrada está vacía")
    return rows


def _number(row: dict[str, str], field: str) -> float:
    return float(row[field])


def add_objective_metrics(row: dict[str, str]) -> dict[str, object]:
    affinity = _number(row, row["topic_feature"])
    new_relevant_reach = (
        _number(row, "estimated_argentina_reach")
        * affinity
        / 100
        * _number(row, "non_follower_reach_pct")
        / 100
    )
    new_relevant_profile_visits = (
        new_relevant_reach * _number(row, "profile_visits_per_1000") / 1000
    )
    return {
        **row,
        "topic_affinity_pct": affinity,
        "estimated_new_relevant_reach": new_relevant_reach,
        "estimated_new_relevant_profile_visits": new_relevant_profile_visits,
    }


def select_campaign(
    all_rows: list[dict[str, str]], campaign: dict[str, str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    regional = [
        row
        for row in all_rows
        if row["campaign"] == campaign["name"]
        and row["primary_candidate"] == "True"
    ]
    if not regional:
        raise ValueError(f"no hay candidatos regionales para {campaign['name']}")

    topic = campaign["topic"]
    affinity_median = statistics.median(_number(row, topic) for row in regional)
    affinity_pool = [row for row in regional if _number(row, topic) >= affinity_median]
    pc1_median = statistics.median(_number(row, "pc1_score") for row in affinity_pool)
    if campaign["pc1_side"] == "high":
        model_pool = [
            row for row in affinity_pool if _number(row, "pc1_score") >= pc1_median
        ]
    else:
        model_pool = [
            row for row in affinity_pool if _number(row, "pc1_score") <= pc1_median
        ]

    enriched = []
    for row in model_pool:
        tagged = dict(row)
        tagged["topic_feature"] = topic
        tagged.update(
            {
                "objective": campaign["objective"],
                "pc1_side": campaign["pc1_side"],
                "regional_affinity_median_pct": affinity_median,
                "affinity_pool_pc1_median": pc1_median,
                "decision_metric": campaign["decision_metric"],
            }
        )
        enriched.append(add_objective_metrics(tagged))

    metric = campaign["decision_metric"]
    ranked = sorted(
        enriched,
        key=lambda row: (
            -float(row[metric]),
            -float(row["topic_affinity_pct"]),
            float(row["price_usd"]),
            row["creator_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["selected"] = rank == 1
    return ranked, ranked[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "campaign",
        "campaign_label",
        "rank",
        "selected",
        "creator_id",
        "name",
        "region_stability_class",
        "region_stability_pct",
        "topic_feature",
        "topic_affinity_pct",
        "regional_affinity_median_pct",
        "pc1_score",
        "oja_index",
        "pc1_side",
        "affinity_pool_pc1_median",
        "decision_metric",
        "estimated_new_relevant_reach",
        "estimated_new_relevant_profile_visits",
        "estimated_argentina_reach",
        "non_follower_reach_pct",
        "profile_visits_per_1000",
        "price_usd",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 0) -> str:
    return f"{float(value):,.{digits}f}"


def build_report(
    selections: list[dict[str, object]], pools: list[dict[str, object]]
) -> str:
    lines = [
        "# Selección de campañas guiada por los modelos",
        "",
        "## Qué corrige esta versión",
        "",
        "La selección anterior recorría todo el catálogo y ordenaba por un proxy comercial. "
        "Kohonen quedaba como una descripción posterior. Esta versión invierte el orden: "
        "los modelos delimitan primero el perfil buscado y las métricas originales sólo "
        "deciden entre los creadores que cumplen ese perfil.",
        "",
        "En este informe, **afinidad temática** significa el porcentaje del contenido del "
        "creador dedicado al tema de la campaña. Es una variable original del dataset. No "
        "mide afinidad de la audiencia ni proviene de Kohonen.",
        "",
        "## Regla de selección",
        "",
        "1. **Kohonen define el conjunto temático.** Se consideran los creadores cuya región "
        "modal coincide con la campaña y cuya asignación es estable o fronteriza.",
        "2. **La afinidad elimina los casos débiles dentro de la región.** Se conserva la mitad "
        "superior según el porcentaje del tema correspondiente. El umbral es la mediana de "
        "cada región, no un 50 % común inventado.",
        "3. **PCA define el tipo de perfil dentro de esa región.** Para ventas se usa la mitad "
        "con PC1 más alto, asociada con escala y descubrimiento. Para perros se usa la mitad "
        "con PC1 más bajo, asociada con tutoriales, tiempo de reproducción y visitas al perfil.",
        "4. **Oja valida PC1, pero no agrega otro voto.** Su correlación casi perfecta con PC1 "
        "indica que ambos describen el mismo eje.",
        "5. **La métrica del objetivo decide dentro del conjunto resultante.** Para ventas se "
        "maximiza el alcance argentino temático proveniente de no seguidores. Para perros se "
        "incorpora además la tasa de visitas al perfil como señal de consideración.",
        "",
        "Las cuentas son proxies, no predicciones de ventas ni adopciones:",
        "",
        "- `alcance nuevo relevante = alcance argentino × afinidad temática × % no seguidores`",
        "- `visitas nuevas relevantes = alcance nuevo relevante × visitas al perfil por 1000 / 1000`",
        "",
        "Estas multiplicaciones permiten comparar alternativas con una unidad interpretable, "
        "pero suponen que los porcentajes se aplican proporcionalmente. El dataset no permite "
        "comprobar esa suposición ni observar ventas, consultas o adopciones.",
        "",
        "## Resultado",
        "",
        "| Campaña | Candidato | Aporte de los modelos | Métrica de decisión |",
        "| --- | --- | --- | ---: |",
    ]
    for row in selections:
        if row["campaign"] == "perros":
            metric = f"{fmt(row['estimated_new_relevant_profile_visits'], 1)} visitas estimadas"
            side = "región perros, afinidad alta y lado profundo de PC1"
        else:
            metric = f"{fmt(row['estimated_new_relevant_reach'])} cuentas estimadas"
            side = f"región {row['campaign_label'].lower()}, afinidad alta y lado de escala de PC1"
        lines.append(
            f"| {row['campaign_label']} | {row['name']} | {side} | {metric} |"
        )

    lines.extend(
        [
            "",
            "## Finalistas después de aplicar los modelos",
            "",
            "| Campaña | Orden por objetivo | Creador | Afinidad | PC1 |",
            "| --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in pools:
        lines.append(
            f"| {row['campaign_label']} | {row['rank']} | {row['name']} | "
            f"{fmt(row['topic_affinity_pct'], 1)} % | {float(row['pc1_score']):+.2f} |"
        )

    lines.extend(
        [
            "",
            "## Justificación por campaña",
            "",
        ]
    )
    explanations = {
        "futbol": "Conserva la elección anterior, pero ahora llega a ella después de exigir coherencia con la región de fútbol, afinidad superior a la mediana y posición en el lado de escala de PC1.",
        "belleza": "Reemplaza a @maquillajereal. @glamencasa pertenece de forma estable a belleza y gana en alcance nuevo relevante dentro del perfil de escala definido por PCA.",
        "perros": "Reemplaza a @crearenreels. @guiadehuellas pertenece de forma estable a perros, queda en el lado de profundidad de PC1 y maximiza las visitas nuevas relevantes dentro de ese perfil. La cuenta sigue siendo una aproximación y no mide adopciones.",
        "arte": "Reemplaza a @galeriacotidiana. @pincelcotidiano es fronterizo, pero su región modal es arte y gana en alcance nuevo relevante dentro del perfil de escala. La condición fronteriza debe mostrarse como cautela.",
    }
    for row in selections:
        lines.extend(
            [
                f"### {row['campaign_label']}: {row['name']}",
                "",
                explanations[str(row["campaign"])],
                "",
                f"- Afinidad temática: **{fmt(row['topic_affinity_pct'], 1)} %**.",
                f"- Estabilidad de región: **{row['region_stability_class']} ({fmt(row['region_stability_pct'])} %)**.",
                f"- PC1 / Oja: **{fmt(row['pc1_score'], 2)} / {fmt(row['oja_index'], 2)}**.",
                f"- Alcance argentino estimado: **{fmt(row['estimated_argentina_reach'])}**.",
                f"- Alcance de no seguidores: **{fmt(row['non_follower_reach_pct'], 1)} %**.",
                f"- Precio: **USD {fmt(row['price_usd'])}**.",
                "",
            ]
        )

    lines.extend(
        [
            "## Qué queda para la presentación",
            "",
            "La futura comparación con followers debe usar estos cuatro seleccionados. "
            "La comparación debe separar alcance, afinidad, señal de respuesta y costo; no "
            "debe reducirlos a una única barra sin unidad.",
            "",
            "El archivo `generated/conjunto_finalista.csv` conserva todos los candidatos que "
            "superaron las tres etapas de los modelos, con sus umbrales y cuentas. Así se puede "
            "explicar por qué ganó cada creador y quién quedó segundo.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "generated").mkdir(exist_ok=True)

    pools: list[dict[str, object]] = []
    selections: list[dict[str, object]] = []
    for campaign in CAMPAIGNS:
        ranked, selected = select_campaign(rows, campaign)
        pools.extend(ranked)
        selections.append(selected)

    write_csv(args.output / "generated" / "conjunto_finalista.csv", pools)
    write_csv(args.output / "generated" / "seleccion_final.csv", selections)
    (args.output / "README.md").write_text(
        build_report(selections, pools), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
