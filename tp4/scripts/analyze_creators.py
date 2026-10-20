#!/usr/bin/env python3
"""Genera el EDA reproducible del catálogo de creadores del TP4.

El script usa únicamente la biblioteca estándar para que el análisis inicial
pueda reproducirse sin instalar dependencias. No implementa PCA, Kohonen ni
Oja: su objetivo es producir la evidencia necesaria para decidir las variables
de entrada de esos métodos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


IDENTIFIER_COLUMNS = ("creator_id", "name")
EXPECTED_COLUMNS = (
    "creator_id",
    "name",
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

BASE_FEATURES = (
    ("average_reach", "Alcance único medio por publicación."),
    ("monthly_posts", "Frecuencia de publicación."),
    ("argentina_audience_pct", "Afinidad geográfica de la audiencia."),
    ("non_follower_reach_pct", "Capacidad de descubrimiento fuera de la base actual."),
    ("total_engagement_per_1000", "Interacción total comparable por alcance."),
    ("profile_visits_per_1000", "Conversión de alcance en interés por el perfil."),
    ("price_usd", "Costo de una publicación patrocinada."),
    ("sponsored_posts_pct", "Nivel de saturación comercial."),
    ("topic_football_pct", "Especialización temática en fútbol."),
    ("topic_makeup_pct", "Especialización temática en maquillaje."),
    ("topic_dogs_pct", "Especialización temática en perros."),
    ("content_tutorial_pct", "Profundidad educativa del contenido."),
    ("content_review_pct", "Contenido orientado a evaluación de productos."),
    ("content_entertainment_pct", "Contenido orientado a entretenimiento."),
    ("average_watch_time_seconds", "Profundidad de consumo audiovisual."),
)

SELECTED_CORRELATION_JUSTIFICATIONS = {
    frozenset(("average_reach", "price_usd")): (
        "El precio crece con el alcance, pero costo y exposición cumplen roles distintos."
    ),
    frozenset(("average_reach", "average_watch_time_seconds")): (
        "Distinguen escala de audiencia y profundidad de consumo."
    ),
    frozenset(("content_tutorial_pct", "content_entertainment_pct")): (
        "Son estilos opuestos dentro de una composición, ambos relevantes para campañas."
    ),
    frozenset(("content_review_pct", "content_entertainment_pct")): (
        "Son estilos opuestos dentro de una composición, ambos relevantes para campañas."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datos/creadores.csv"),
        help="CSV fuente (por defecto, datos/creadores.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/eda-creadores"),
        help="Directorio para el informe y los artefactos.",
    )
    return parser.parse_args()


def read_dataset(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("El CSV no tiene encabezado.")
        rows = list(reader)
        return reader.fieldnames, rows


def validate_dataset(columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    if tuple(columns) != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(columns))
        extra = sorted(set(columns) - set(EXPECTED_COLUMNS))
        raise ValueError(
            "El esquema no coincide con el esperado. "
            f"Faltan={missing or 'ninguna'}; sobran={extra or 'ninguna'}."
        )
    if not rows:
        raise ValueError("El CSV no contiene creadores.")


def numeric_data(
    columns: Sequence[str], rows: Sequence[dict[str, str]]
) -> tuple[list[str], dict[str, list[float]]]:
    numeric_columns = [name for name in columns if name not in IDENTIFIER_COLUMNS]
    values: dict[str, list[float]] = {name: [] for name in numeric_columns}
    for row_index, row in enumerate(rows, start=2):
        for name in numeric_columns:
            raw = row[name].strip()
            if raw == "":
                continue
            try:
                values[name].append(float(raw))
            except ValueError as error:
                raise ValueError(
                    f"Valor no numérico en fila {row_index}, columna {name}: {raw!r}."
                ) from error
    return numeric_columns, values


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def skewness(values: Sequence[float]) -> float:
    mean = statistics.fmean(values)
    second = statistics.fmean((value - mean) ** 2 for value in values)
    if second == 0:
        return 0.0
    third = statistics.fmean((value - mean) ** 3 for value in values)
    return third / (second**1.5)


def describe(name: str, values: Sequence[float], row_count: int) -> dict[str, object]:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = sum(value < lower or value > upper for value in values)
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "variable": name,
        "count": len(values),
        "missing": row_count - len(values),
        "min": min(values),
        "q1": q1,
        "median": statistics.median(values),
        "mean": mean,
        "q3": q3,
        "max": max(values),
        "std": standard_deviation,
        "cv_abs": abs(standard_deviation / mean) if mean != 0 else math.nan,
        "skewness": skewness(values),
        "iqr_outliers": outliers,
    }


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("No se puede correlacionar columnas con distinta longitud.")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_sum = sum((value - left_mean) ** 2 for value in left)
    right_sum = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_sum * right_sum)
    if not denominator:
        return math.nan
    return max(-1.0, min(1.0, numerator / denominator))


def correlation_matrix(
    columns: Sequence[str], data: dict[str, list[float]]
) -> dict[tuple[str, str], float]:
    return {
        (left, right): pearson(data[left], data[right])
        for left in columns
        for right in columns
    }


def relation_checks(rows: Sequence[dict[str, str]]) -> list[dict[str, object]]:
    def maximum_error(function) -> float:
        return max(abs(function(row)) for row in rows)

    return [
        {
            "relacion": "estimated_argentina_reach = average_reach * argentina_audience_pct / 100",
            "max_abs_error": maximum_error(
                lambda row: float(row["estimated_argentina_reach"])
                - float(row["average_reach"])
                * float(row["argentina_audience_pct"])
                / 100
            ),
            "tolerancia": 0.5,
            "interpretacion": "Identidad con redondeo a una cuenta.",
        },
        {
            "relacion": "total_engagement_per_1000 = likes + comments + saves + shares",
            "max_abs_error": maximum_error(
                lambda row: float(row["total_engagement_per_1000"])
                - sum(
                    float(row[name])
                    for name in (
                        "likes_per_1000",
                        "comments_per_1000",
                        "saves_per_1000",
                        "shares_per_1000",
                    )
                )
            ),
            "tolerancia": 1e-9,
            "interpretacion": "Identidad exacta salvo precisión de punto flotante.",
        },
        {
            "relacion": "price_ars = 1500 * price_usd",
            "max_abs_error": maximum_error(
                lambda row: float(row["price_ars"]) - 1500 * float(row["price_usd"])
            ),
            "tolerancia": 1e-6,
            "interpretacion": "Identidad exacta salvo precisión de punto flotante.",
        },
        {
            "relacion": "sponsored_posts_pct + organic_posts_pct = 100",
            "max_abs_error": maximum_error(
                lambda row: float(row["sponsored_posts_pct"])
                + float(row["organic_posts_pct"])
                - 100
            ),
            "tolerancia": 1e-9,
            "interpretacion": "Composición cerrada de dos porcentajes.",
        },
        {
            "relacion": "topic_football_pct + topic_makeup_pct + topic_dogs_pct + topic_art_pct = 100",
            "max_abs_error": maximum_error(
                lambda row: sum(
                    float(row[name])
                    for name in (
                        "topic_football_pct",
                        "topic_makeup_pct",
                        "topic_dogs_pct",
                        "topic_art_pct",
                    )
                )
                - 100
            ),
            "tolerancia": 1e-9,
            "interpretacion": "Composición temática cerrada.",
        },
        {
            "relacion": "content_tutorial_pct + content_review_pct + content_entertainment_pct + content_showcase_pct = 100",
            "max_abs_error": maximum_error(
                lambda row: sum(
                    float(row[name])
                    for name in (
                        "content_tutorial_pct",
                        "content_review_pct",
                        "content_entertainment_pct",
                        "content_showcase_pct",
                    )
                )
                - 100
            ),
            "tolerancia": 1e-9,
            "interpretacion": "Composición de contenido cerrada.",
        },
    ]


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No hay filas para escribir en {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def svg_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def correlation_color(value: float) -> str:
    if math.isnan(value):
        return "#d1d5db"
    value = max(-1.0, min(1.0, value))
    negative = (37, 99, 235)
    neutral = (255, 255, 255)
    positive = (220, 38, 38)
    if value < 0:
        weight = -value
        start, end = neutral, negative
    else:
        weight = value
        start, end = neutral, positive
    rgb = tuple(round(start[i] * (1 - weight) + end[i] * weight) for i in range(3))
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def write_correlation_heatmap(
    path: Path,
    columns: Sequence[str],
    correlations: dict[tuple[str, str], float],
    title: str = "Correlación de Pearson entre variables numéricas",
) -> None:
    cell = 22
    left = 250
    top = 255
    size = len(columns) * cell
    width = left + size + 110
    height = top + size + 55
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.label{font-size:10px}.title{font-size:18px;font-weight:700}.note{font-size:11px;fill:#4b5563}</style>',
        f'<text x="20" y="28" class="title">{svg_text(title)}</text>',
        '<text x="20" y="48" class="note">Azul: correlación negativa. Rojo: correlación positiva. La diagonal vale 1.</text>',
    ]
    for index, name in enumerate(columns):
        y = top + index * cell + cell * 0.72
        parts.append(
            f'<text x="{left - 8}" y="{y:.1f}" text-anchor="end" class="label">{svg_text(name)}</text>'
        )
        x = left + index * cell + cell * 0.65
        parts.append(
            f'<text transform="translate({x:.1f},{top - 8}) rotate(-60)" text-anchor="start" class="label">{svg_text(name)}</text>'
        )
    for row_index, left_name in enumerate(columns):
        for column_index, right_name in enumerate(columns):
            value = correlations[(left_name, right_name)]
            x = left + column_index * cell
            y = top + row_index * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{correlation_color(value)}" stroke="#e5e7eb" stroke-width="0.5"><title>{svg_text(left_name)} / {svg_text(right_name)}: {value:.4f}</title></rect>'
            )
    legend_x = left + size + 35
    for step in range(101):
        value = 1 - 2 * step / 100
        parts.append(
            f'<rect x="{legend_x}" y="{top + step * size / 101:.2f}" width="18" height="{size / 101 + 0.5:.2f}" fill="{correlation_color(value)}"/>'
        )
    for value in (1, 0.5, 0, -0.5, -1):
        y = top + (1 - value) * size / 2
        parts.append(f'<text x="{legend_x + 26}" y="{y + 4:.1f}" class="label">{value:g}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def histogram(values: Sequence[float], bin_count: int = 10) -> list[int]:
    low = min(values)
    high = max(values)
    if low == high:
        result = [0] * bin_count
        result[0] = len(values)
        return result
    counts = [0] * bin_count
    for value in values:
        index = min(bin_count - 1, int((value - low) / (high - low) * bin_count))
        counts[index] += 1
    return counts


def write_distributions(
    path: Path,
    columns: Sequence[str],
    data: dict[str, list[float]],
    title: str = "Distribuciones de las variables numéricas",
) -> None:
    chart_columns = 3
    panel_width = 390
    panel_height = 150
    margin = 24
    title_height = 65
    rows = math.ceil(len(columns) / chart_columns)
    width = chart_columns * panel_width + margin * 2
    height = title_height + rows * panel_height + margin
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:18px;font-weight:700}.panel-title{font-size:11px;font-weight:700}.axis{font-size:9px;fill:#4b5563}</style>',
        f'<text x="24" y="28" class="title">{svg_text(title)}</text>',
        '<text x="24" y="48" class="axis">Histogramas de 10 intervalos. Cada panel usa su propia escala horizontal y vertical.</text>',
    ]
    for index, name in enumerate(columns):
        panel_row, panel_column = divmod(index, chart_columns)
        x0 = margin + panel_column * panel_width
        y0 = title_height + panel_row * panel_height
        plot_x = x0 + 42
        plot_y = y0 + 28
        plot_width = panel_width - 65
        plot_height = 82
        values = data[name]
        counts = histogram(values)
        maximum = max(counts) or 1
        parts.append(f'<text x="{x0 + 8}" y="{y0 + 16}" class="panel-title">{svg_text(name)}</text>')
        parts.append(
            f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="#64748b" stroke-width="1"/>'
        )
        bar_width = plot_width / len(counts)
        for bin_index, count in enumerate(counts):
            bar_height = plot_height * count / maximum
            x = plot_x + bin_index * bar_width + 1
            y = plot_y + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" height="{bar_height:.2f}" fill="#2563eb"><title>n={count}</title></rect>'
            )
        parts.extend(
            [
                f'<text x="{plot_x}" y="{plot_y + plot_height + 15}" text-anchor="start" class="axis">{min(values):.4g}</text>',
                f'<text x="{plot_x + plot_width}" y="{plot_y + plot_height + 15}" text-anchor="end" class="axis">{max(values):.4g}</text>',
                f'<text x="{plot_x - 5}" y="{plot_y + 8}" text-anchor="end" class="axis">{maximum}</text>',
                f'<text x="{plot_x - 5}" y="{plot_y + plot_height}" text-anchor="end" class="axis">0</text>',
            ]
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def format_number(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "n.a."
        return f"{value:.4g}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    result = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        result.append("| " + " | ".join(format_number(value) for value in row) + " |")
    return "\n".join(result)


def report_markdown(
    *,
    data_path: Path,
    digest: str,
    columns: Sequence[str],
    rows: Sequence[dict[str, str]],
    numeric_columns: Sequence[str],
    summaries: Sequence[dict[str, object]],
    checks: Sequence[dict[str, object]],
    high_correlations: Sequence[dict[str, object]],
    selected_high_correlations: Sequence[dict[str, object]],
) -> str:
    missing_cells = sum(
        value.strip() == "" for row in rows for value in row.values() if value is not None
    )
    duplicate_rows = len(rows) - len({tuple(row[name] for name in columns) for row in rows})
    duplicate_ids = sum(count - 1 for count in Counter(row["creator_id"] for row in rows).values() if count > 1)
    duplicate_names = sum(count - 1 for count in Counter(row["name"] for row in rows).values() if count > 1)
    most_skewed = sorted(summaries, key=lambda item: abs(float(item["skewness"])), reverse=True)[:8]
    most_outliers = sorted(
        summaries, key=lambda item: int(item["iqr_outliers"]), reverse=True
    )[:8]
    correlation_preview = high_correlations[:20]
    summary_preview = [
        item
        for item in summaries
        if item["variable"]
        in {
            "followers",
            "average_reach",
            "estimated_argentina_reach",
            "total_engagement_per_1000",
            "price_usd",
            "sponsored_posts_pct",
            "average_caption_words",
            "average_watch_time_seconds",
        }
    ]
    check_table = markdown_table(
        ("Relación", "Error máximo", "Tolerancia", "Resultado"),
        (
            (
                check["relacion"],
                check["max_abs_error"],
                check["tolerancia"],
                "Verificada"
                if float(check["max_abs_error"]) <= float(check["tolerancia"])
                else "No verificada",
            )
            for check in checks
        ),
    )
    descriptive_table = markdown_table(
        ("Variable", "Mínimo", "Mediana", "Media", "Máximo", "Desvío", "Outliers IQR"),
        (
            (
                item["variable"],
                item["min"],
                item["median"],
                item["mean"],
                item["max"],
                item["std"],
                item["iqr_outliers"],
            )
            for item in summary_preview
        ),
    )
    correlation_table = markdown_table(
        ("Variable A", "Variable B", "r"),
        (
            (item["variable_a"], item["variable_b"], item["correlation"])
            for item in correlation_preview
        ),
    )
    proposal_table = markdown_table(
        ("Variable base", "Justificación"), BASE_FEATURES
    )
    selected_correlation_table = markdown_table(
        ("Variable A", "Variable B", "r", "Motivo para conservar ambas"),
        (
            (
                item["variable_a"],
                item["variable_b"],
                item["correlation"],
                item["justification"],
            )
            for item in selected_high_correlations
        ),
    )
    skewness_list = "\n".join(
        f"- `{item['variable']}`: asimetría {float(item['skewness']):.2f}."
        for item in most_skewed
    )
    outlier_list = "\n".join(
        f"- `{item['variable']}`: {item['iqr_outliers']} "
        f"{'observación' if int(item['iqr_outliers']) == 1 else 'observaciones'} "
        "fuera de 1,5 IQR."
        for item in most_outliers
        if int(item["iqr_outliers"]) > 0
    ) or "- Ninguna variable presenta observaciones fuera de 1,5 IQR."
    return f"""# EDA del catálogo de creadores

Este informe describe el conjunto de datos antes de implementar PCA, Kohonen u
Oja. Su objetivo es detectar problemas de calidad, escalas incompatibles y
redundancias, y proponer una primera selección de variables para discutir.

El informe se regenera ejecutando, desde `tp4/`:

```bash
python3 scripts/analyze_creators.py
```

Fuente analizada: `{data_path.as_posix()}`. SHA-256: `{digest}`.

## Alcance y protocolo

- Se analizaron todas las observaciones disponibles, porque todavía no se está
  ajustando ni evaluando un modelo supervisado.
- `creator_id` y `name` se trataron como identificadores.
- Las otras columnas se validaron como numéricas.
- Los posibles outliers se marcaron con la regla exploratoria de 1,5 IQR. No se
  eliminaron ni modificaron datos.
- Las asociaciones lineales se midieron con correlación de Pearson.
- No se aplicaron transformaciones, estandarización ni reducción de dimensión.

## Estructura y calidad

- Filas: **{len(rows)}** creadores.
- Columnas: **{len(columns)}** en total, **{len(numeric_columns)}** numéricas y
  **{len(IDENTIFIER_COLUMNS)}** identificadores.
- Celdas vacías: **{missing_cells}**.
- Filas duplicadas: **{duplicate_rows}**.
- Identificadores `creator_id` repetidos: **{duplicate_ids}**.
- Nombres repetidos: **{duplicate_names}**.

No se detectaron faltantes ni duplicados. Esto permite avanzar sin imputación,
pero no elimina la necesidad de revisar redundancias y distribuciones.

## Escalas y distribuciones

Las variables operan en escalas muy diferentes. Seguidores, alcance y precios
llegan a cientos de miles o millones de unidades, mientras que otras columnas
son porcentajes o tasas por 1000. Por eso los métodos basados en distancia o
varianza requerirán estandarización.

Resumen de variables representativas:

{descriptive_table}

![Distribuciones de las variables](generated/distribuciones.svg)

Las mayores asimetrías en valor absoluto son:

{skewness_list}

La estandarización iguala media y desvío, pero no elimina asimetrías ni el
efecto de observaciones extremas. Antes de aplicar una transformación, hay que
discutir si está justificada y si forma parte del preprocesamiento permitido.

Variables con más observaciones marcadas por la regla de 1,5 IQR:

{outlier_list}

Estos puntos no deben eliminarse automáticamente: un creador grande o muy
especializado puede ser un caso comercial válido y relevante.

## Identidades y redundancias exactas

{check_table}

Estas relaciones implican que no conviene introducir simultáneamente todas las
columnas involucradas sin una justificación. Repetir una magnitud aumenta su
peso relativo en PCA y en las distancias de Kohonen.

En los grupos de porcentajes que suman 100, una columna puede reconstruirse a
partir de las otras. La propuesta inicial usa `topic_art_pct` y
`content_showcase_pct` como categorías de referencia omitidas. Eso evita una
dependencia lineal exacta, pero la elección es convencional y debe discutirse.

## Correlaciones

Se encontraron **{len(high_correlations)}** pares con `|r| >= 0,85`. Los veinte
más intensos son:

{correlation_table}

![Mapa de correlaciones](generated/correlaciones.svg)

Los principales bloques son:

- seguidores, alcance, impresiones y vistas de video;
- alcance, vistas y precio;
- `price_usd` y `price_ars`, que además son la misma magnitud convertida;
- publicaciones patrocinadas y orgánicas, que son complementarias;
- publicaciones con video, formato breve y participación en tendencias.

La correlación ayuda a detectar información repetida, pero no decide por sí
sola qué columna conservar. La selección también debe mantener la semántica
necesaria para interpretar campañas.

## Variables acordadas para la primera variante

La primera variante de PCA usará estas 15 variables. La selección
mantiene dimensiones de alcance, frecuencia, audiencia, interacción, costo,
saturación comercial, tema, tipo de contenido y profundidad de consumo.

{proposal_table}

Decisiones confirmadas:

- `followers` se reserva como baseline de comparación y no entra inicialmente
  al modelo. `average_reach` representa mejor la exposición efectiva. Después
  se colorearán o etiquetarán resultados con `followers` para compararlos con
  la estrategia actual de la agencia.
- La primera variante usa engagement total y conserva sus cuatro componentes
  como auxiliares. Más adelante se ejecutará una segunda variante que reemplace
  el total por likes, comentarios, guardados y compartidos. No se usarán el
  total y sus cuatro componentes juntos. La representación final se decidirá
  según cuánto cambien PCA y los perfiles encontrados.
- Se conserva `price_usd` y se descarta su conversión exacta a ARS.
- Se conserva `sponsored_posts_pct` y se descarta su complemento orgánico.
- Se omiten `topic_art_pct` y `content_showcase_pct` sólo para romper las sumas
  exactas; se reconstruyen como el complemento de las otras tres categorías.
- `video_posts_pct`, `short_form_posts_pct` y `trend_posts_pct` quedan como
  auxiliares. Entre sí tienen correlaciones de 0,95 a 0,97: incluir las tres
  haría que el mismo estilo breve, audiovisual y de tendencia pesara tres veces.
  Además, la selección ya representa descubrimiento con
  `non_follower_reach_pct`, entretenimiento con `content_entertainment_pct` y
  profundidad con `average_watch_time_seconds`. Si luego falta una variable de
  formato, la primera candidata será `short_form_posts_pct`.
- `estimated_argentina_reach` queda como métrica derivada de interpretación,
  no como entrada simultánea a `average_reach` y `argentina_audience_pct`.

## Revisión específica de las variables elegidas

Dentro de las 15 variables permanecen estas asociaciones con `|r| >= 0,70`:

{selected_correlation_table}

![Correlaciones de las variables elegidas](generated/correlaciones_seleccionadas.svg)

![Distribuciones de las variables elegidas](generated/distribuciones_seleccionadas.svg)

No hay redundancias exactas dentro de la selección. Las correlaciones restantes
responden a relaciones comerciales o a las composiciones del dataset. Se
conservan porque las variables tienen significados distintos que se necesitan
para interpretar campañas. El análisis PCA posterior muestra cuánto de esa
información se resume en las mismas componentes y queda documentado en
`docs/pca-creadores/README.md`.

## Variables auxiliares para interpretar

Aunque no entren inicialmente al modelo, deben conservarse junto a los
resultados:

- `creator_id` y `name`, para identificar observaciones;
- `followers`, para el baseline exigido por el enunciado;
- `estimated_argentina_reach`, para expresar alcance local en cuentas;
- `price_ars`, para comunicar costo local sin duplicarlo en el entrenamiento;
- `topic_art_pct` y `content_showcase_pct`, reconstruibles como complementos;
- las cuatro interacciones individuales, para explicar la composición del
  engagement y construir la variante comparativa acordada;
- `video_posts_pct`, `short_form_posts_pct` y `trend_posts_pct`, para describir
  el formato de las regiones sin darle tres veces ese peso al entrenamiento.

## Conclusiones del EDA

El dataset está completo y es pequeño, pero contiene escalas incompatibles,
variables derivadas, composiciones cerradas y bloques con correlación muy alta.
No conviene pasar las 31 variables numéricas directamente a los modelos.

El EDA descriptivo permitió cerrar la selección de 15 variables. Los análisis
posteriores usan únicamente z-score; no fue necesario agregar una
transformación logarítmica. PCA incluye varianza explicada, cargas, scores y
biplot, y la comparación de representaciones de engagement confirmó el total
como entrada definitiva para PCA, Kohonen y Oja.

## Archivos generados

- `generated/resumen_variables.csv`: estadísticos por variable.
- `generated/resumen_variables_seleccionadas.csv`: estadísticos de las 15
  variables base.
- `generated/relaciones_exactas.csv`: controles de identidades algebraicas.
- `generated/correlaciones.csv`: matriz completa en formato largo.
- `generated/correlaciones_altas.csv`: pares con `|r| >= 0,85`.
- `generated/correlaciones_seleccionadas.csv`: matriz de correlación de las 15
  variables base.
- `generated/distribuciones.svg`: histogramas por variable.
- `generated/distribuciones_seleccionadas.svg`: histogramas del conjunto base.
- `generated/correlaciones.svg`: mapa de correlaciones.
- `generated/correlaciones_seleccionadas.svg`: mapa de correlaciones del
  conjunto base.
"""


def main() -> None:
    args = parse_args()
    data_path = args.data.resolve()
    output_path = args.output.resolve()
    generated_path = output_path / "generated"
    output_path.mkdir(parents=True, exist_ok=True)
    generated_path.mkdir(parents=True, exist_ok=True)

    columns, rows = read_dataset(data_path)
    validate_dataset(columns, rows)
    numeric_columns, data = numeric_data(columns, rows)
    if any(len(data[name]) != len(rows) for name in numeric_columns):
        raise ValueError("Hay valores numéricos faltantes; revisar el CSV antes de continuar.")

    summaries = [describe(name, data[name], len(rows)) for name in numeric_columns]
    checks = relation_checks(rows)
    correlations = correlation_matrix(numeric_columns, data)
    correlation_rows = [
        {
            "variable_a": left,
            "variable_b": right,
            "correlation": correlations[(left, right)],
        }
        for left in numeric_columns
        for right in numeric_columns
    ]
    high_correlations = sorted(
        (
            {
                "variable_a": left,
                "variable_b": right,
                "correlation": correlations[(left, right)],
                "abs_correlation": abs(correlations[(left, right)]),
            }
            for left_index, left in enumerate(numeric_columns)
            for right in numeric_columns[left_index + 1 :]
            if abs(correlations[(left, right)]) >= 0.85
        ),
        key=lambda item: float(item["abs_correlation"]),
        reverse=True,
    )
    selected_columns = [name for name, _ in BASE_FEATURES]
    selected_summaries = [
        summary for summary in summaries if summary["variable"] in selected_columns
    ]
    selected_correlation_rows = [
        {
            "variable_a": left,
            "variable_b": right,
            "correlation": correlations[(left, right)],
        }
        for left in selected_columns
        for right in selected_columns
    ]
    selected_high_correlations = sorted(
        (
            {
                "variable_a": left,
                "variable_b": right,
                "correlation": correlations[(left, right)],
                "abs_correlation": abs(correlations[(left, right)]),
                "justification": SELECTED_CORRELATION_JUSTIFICATIONS[
                    frozenset((left, right))
                ],
            }
            for left_index, left in enumerate(selected_columns)
            for right in selected_columns[left_index + 1 :]
            if abs(correlations[(left, right)]) >= 0.70
        ),
        key=lambda item: float(item["abs_correlation"]),
        reverse=True,
    )

    write_csv(generated_path / "resumen_variables.csv", summaries)
    write_csv(
        generated_path / "resumen_variables_seleccionadas.csv", selected_summaries
    )
    write_csv(generated_path / "relaciones_exactas.csv", checks)
    write_csv(generated_path / "correlaciones.csv", correlation_rows)
    write_csv(generated_path / "correlaciones_altas.csv", high_correlations)
    write_csv(
        generated_path / "correlaciones_seleccionadas.csv",
        selected_correlation_rows,
    )
    write_correlation_heatmap(
        generated_path / "correlaciones.svg", numeric_columns, correlations
    )
    write_distributions(generated_path / "distribuciones.svg", numeric_columns, data)
    write_correlation_heatmap(
        generated_path / "correlaciones_seleccionadas.svg",
        selected_columns,
        correlations,
        "Correlación de Pearson de las 15 variables base",
    )
    write_distributions(
        generated_path / "distribuciones_seleccionadas.svg",
        selected_columns,
        data,
        "Distribuciones de las 15 variables base",
    )

    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    try:
        displayed_data_path = data_path.relative_to(Path.cwd().resolve())
    except ValueError:
        displayed_data_path = data_path
    report = report_markdown(
        data_path=displayed_data_path,
        digest=digest,
        columns=columns,
        rows=rows,
        numeric_columns=numeric_columns,
        summaries=summaries,
        checks=checks,
        high_correlations=high_correlations,
        selected_high_correlations=selected_high_correlations,
    )
    (output_path / "README.md").write_text(report, encoding="utf-8")
    print(f"EDA generado en {output_path}")


if __name__ == "__main__":
    main()
