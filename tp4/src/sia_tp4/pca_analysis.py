"""Ejecutar y documentar PCA sobre el catálogo de creadores."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .pca import PCAResult, fit_pca


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
        default=Path("docs/pca-creadores"),
        help="Directorio del informe (por defecto, docs/pca-creadores).",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.90,
        help="Umbral de varianza acumulada acordado (por defecto, 0.90).",
    )
    return parser.parse_args(argv)


def _component_names(count: int) -> list[str]:
    return [f"PC{index}" for index in range(1, count + 1)]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(output: Path, dataset: CreatorDataset, result: PCAResult) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    component_names = _component_names(len(result.eigenvalues))

    _write_csv(
        generated / "varianza_explicada.csv",
        ["component", "eigenvalue", "explained_variance_ratio", "cumulative_variance"],
        (
            {
                "component": name,
                "eigenvalue": eigenvalue,
                "explained_variance_ratio": ratio,
                "cumulative_variance": cumulative,
            }
            for name, eigenvalue, ratio, cumulative in zip(
                component_names,
                result.eigenvalues,
                result.explained_variance_ratio,
                result.cumulative_explained_variance,
            )
        ),
    )
    _write_csv(
        generated / "cargas.csv",
        ["variable", *component_names],
        (
            {"variable": variable, **dict(zip(component_names, row))}
            for variable, row in zip(result.feature_names, result.loadings)
        ),
    )
    _write_csv(
        generated / "autovectores.csv",
        ["variable", *component_names],
        (
            {"variable": variable, **dict(zip(component_names, row))}
            for variable, row in zip(result.feature_names, result.components)
        ),
    )
    _write_csv(
        generated / "scores.csv",
        ["creator_id", "name", "followers", *component_names],
        (
            {
                "creator_id": creator_id,
                "name": name,
                "followers": followers,
                **dict(zip(component_names, row)),
            }
            for creator_id, name, followers, row in zip(
                dataset.creator_ids,
                dataset.names,
                dataset.followers,
                result.scores,
            )
        ),
    )
    _write_csv(
        generated / "estandarizacion.csv",
        ["variable", "mean", "sample_standard_deviation"],
        (
            {
                "variable": variable,
                "mean": mean,
                "sample_standard_deviation": deviation,
            }
            for variable, mean, deviation in zip(
                result.feature_names,
                result.standardizer.means,
                result.standardizer.standard_deviations,
            )
        ),
    )


def _load_pyplot():
    cache = Path(tempfile.gettempdir()) / "sia-tp4-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "sia-tp4"
    import matplotlib.pyplot as plt

    return plt


def _save_figure(figure, path: Path) -> None:
    figure.savefig(
        path, format="svg", bbox_inches="tight", metadata={"Date": None}
    )
    svg = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


def write_plots(
    output: Path,
    dataset: CreatorDataset,
    result: PCAResult,
    selected_count: int,
) -> None:
    plt = _load_pyplot()
    generated = output / "generated"
    components = np.arange(1, len(result.eigenvalues) + 1)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.bar(
        components,
        result.explained_variance_ratio * 100,
        color="#4C78A8",
        label="Por componente",
    )
    axis.plot(
        components,
        result.cumulative_explained_variance * 100,
        marker="o",
        color="#E45756",
        label="Acumulada",
    )
    axis.axvline(selected_count, color="#555555", linestyle="--", linewidth=1)
    axis.set(
        title="Varianza explicada por las componentes principales",
        xlabel="Componente",
        ylabel="Varianza explicada (%)",
        xticks=components,
        ylim=(0, 105),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, generated / "varianza_explicada.svg")
    plt.close(figure)

    shown = min(max(selected_count, 2), 8)
    figure, axis = plt.subplots(figsize=(max(8, shown * 1.25), 7.5))
    image = axis.imshow(
        result.loadings[:, :shown], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"
    )
    axis.set_xticks(np.arange(shown), _component_names(shown))
    axis.set_yticks(np.arange(len(result.feature_names)), result.feature_names)
    axis.set_title("Cargas de las variables en las primeras componentes")
    for row in range(len(result.feature_names)):
        for column in range(shown):
            value = result.loadings[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.55 else "black",
            )
    figure.colorbar(image, ax=axis, label="Carga")
    _save_figure(figure, generated / "cargas.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 8))
    colors = np.log10(dataset.followers + 1)
    scatter = axis.scatter(
        result.scores[:, 0],
        result.scores[:, 1],
        c=colors,
        cmap="viridis",
        s=45,
        alpha=0.8,
    )
    for creator_id, x_value, y_value in zip(
        dataset.creator_ids, result.scores[:, 0], result.scores[:, 1]
    ):
        axis.annotate(
            creator_id,
            (x_value, y_value),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            alpha=0.75,
        )

    score_extent = max(np.max(np.abs(result.scores[:, :2])), 1.0)
    loading_extent = max(np.max(np.abs(result.loadings[:, :2])), 1e-12)
    arrow_scale = score_extent * 0.80 / loading_extent
    arrow_ends = []
    for variable, (x_loading, y_loading) in zip(
        result.feature_names, result.loadings[:, :2]
    ):
        x_end, y_end = x_loading * arrow_scale, y_loading * arrow_scale
        arrow_ends.append((x_end, y_end))
        axis.arrow(
            0,
            0,
            x_end,
            y_end,
            color="#B22222",
            alpha=0.7,
            width=0.012,
            head_width=0.10,
            length_includes_head=True,
        )
        short_label = (
            variable.replace("_per_1000", "")
            .replace("_pct", "")
            .replace("average_", "")
        )
        axis.annotate(
            short_label,
            (x_end, y_end),
            xytext=(4 if x_end >= 0 else -4, 4 if y_end >= 0 else -4),
            textcoords="offset points",
            ha="left" if x_end >= 0 else "right",
            va="bottom" if y_end >= 0 else "top",
            color="#8B0000",
            fontsize=7,
        )
    x_values = np.concatenate(
        (result.scores[:, 0], np.asarray(arrow_ends)[:, 0])
    )
    y_values = np.concatenate(
        (result.scores[:, 1], np.asarray(arrow_ends)[:, 1])
    )
    x_padding = max(np.ptp(x_values) * 0.16, 0.5)
    y_padding = max(np.ptp(y_values) * 0.12, 0.5)
    axis.set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)
    axis.set_ylim(y_values.min() - y_padding, y_values.max() + y_padding)
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.set(
        title="Biplot de creadores y variables",
        xlabel=f"PC1 ({result.explained_variance_ratio[0] * 100:.1f} %)",
        ylabel=f"PC2 ({result.explained_variance_ratio[1] * 100:.1f} %)",
    )
    axis.grid(alpha=0.2)
    figure.colorbar(scatter, ax=axis, label="log10(followers + 1), sólo como baseline")
    _save_figure(figure, generated / "biplot.svg")
    plt.close(figure)


def _format_loading(value: float) -> str:
    return f"{value:+.3f}"


def _top_loadings(result: PCAResult, component: int, count: int = 3) -> str:
    values = result.loadings[:, component]
    order = np.argsort(np.abs(values))[::-1][:count]
    return ", ".join(
        f"`{result.feature_names[index]}` ({_format_loading(values[index])})"
        for index in order
    )


def _loading_direction(result: PCAResult, component: int, sign: int) -> str:
    values = result.loadings[:, component]
    candidates = [
        index
        for index, value in enumerate(values)
        if np.sign(value) == sign and abs(value) >= 0.45
    ]
    if not candidates:
        signed = np.where(np.sign(values) == sign, np.abs(values), -np.inf)
        candidates = [int(np.argmax(signed))]
    candidates.sort(key=lambda index: abs(values[index]), reverse=True)
    return ", ".join(f"`{result.feature_names[index]}`" for index in candidates)


def write_report(
    output: Path,
    dataset: CreatorDataset,
    result: PCAResult,
    threshold: float,
    selected_count: int,
) -> None:
    rows = []
    directions = []
    for index in range(selected_count):
        rows.append(
            f"| PC{index + 1} | {result.explained_variance_ratio[index] * 100:.2f} % "
            f"| {result.cumulative_explained_variance[index] * 100:.2f} % "
            f"| {_top_loadings(result, index)} |"
        )
        directions.append(
            f"- **PC{index + 1}:** dirección positiva asociada con "
            f"{_loading_direction(result, index, 1)}; dirección negativa "
            f"asociada con {_loading_direction(result, index, -1)}."
        )

    report = f"""# PCA del catálogo de creadores

Este informe aplica PCA a {len(dataset.creator_ids)} creadores y a las 15
variables base acordadas en `docs/decisiones.md`. Los identificadores, nombres y
`followers` no intervienen en el ajuste. `followers` se usa solamente como
baseline visual en el biplot.

## Procedimiento

1. Se leen las 15 variables en el orden documentado.
2. Se ajusta un z-score por columna con media y desvío muestral (`ddof=1`).
3. Se calcula la covarianza de la matriz estandarizada, equivalente a la matriz
   de correlaciones de las variables originales.
4. Se obtienen y ordenan sus autovalores y autovectores.
5. Se proyecta cada creador sobre todas las componentes.

La orientación de cada autovector se fija haciendo positiva su carga de mayor
magnitud. Esto sólo estabiliza la presentación: invertir simultáneamente cargas
y scores no cambia la solución de PCA.

## Varianza explicada

Según el criterio acordado en clase, el menor número que alcanza al menos
{threshold * 100:.0f} % de varianza acumulada es **{selected_count} componentes**,
con {result.cumulative_explained_variance[selected_count - 1] * 100:.2f} %.
Se adopta este corte porque supera el 80 % y se acerca al 90 % solicitado, sin
conservar automáticamente las 15 componentes.

| Componente | Varianza | Acumulada | Tres cargas de mayor magnitud |
| --- | ---: | ---: | --- |
{chr(10).join(rows)}

![Varianza explicada](generated/varianza_explicada.svg)

## Cargas e interpretación

Las cargas son correlaciones entre cada variable estandarizada y cada
componente. Valores con magnitud alta son los que más ayudan a interpretar el
eje. El signo indica direcciones opuestas dentro del eje, no que una dirección
sea mejor que la otra.

![Cargas](generated/cargas.svg)

La tabla anterior permite formular nombres de negocio para las componentes,
pero esos nombres deben revisarse junto con el biplot y las campañas. No se
adopta automáticamente una componente sólo por superar un umbral numérico.

Lectura objetiva de las direcciones con cargas de magnitud al menos 0,45. Si
ninguna carga de un signo alcanza ese valor, se muestra la mayor de ese signo:

{chr(10).join(directions)}

PC1 ofrece la oposición más clara: perfiles de mayor escala, costo,
descubrimiento y entretenimiento frente a perfiles de mayor profundidad e
interés posterior. Las componentes siguientes separan ejes temáticos,
comerciales y de actividad más específicos. Las últimas componentes retenidas
son más específicas y tienen cargas más repartidas, pero permiten conservar el
porcentaje de información acordado.

## Proyección de los creadores

![Biplot](generated/biplot.svg)

El biplot muestra PC1 y PC2, que juntas explican
{result.cumulative_explained_variance[1] * 100:.2f} % de la varianza. Los puntos
cercanos tienen perfiles parecidos sólo respecto de este plano. Las flechas
representan las cargas y fueron reescaladas únicamente para poder dibujarlas
junto con los scores. El color usa `followers` como referencia externa y no
afecta PCA.

## Decisión adoptada

Se conservan **{selected_count} componentes**, que reúnen
**{result.cumulative_explained_variance[selected_count - 1] * 100:.2f} %** de la
varianza. La comparación de engagement confirmó además las 15 variables base.
Esta reducción describe la representación PCA: Kohonen y Oja reciben las 15
variables estandarizadas, y Oja se compara específicamente con PC1.

## Archivos generados

- `generated/varianza_explicada.csv`: autovalores y proporciones de varianza;
- `generated/cargas.csv`: correlación de cada variable con cada componente;
- `generated/autovectores.csv`: coeficientes de las combinaciones lineales;
- `generated/scores.csv`: proyección de cada creador;
- `generated/estandarizacion.csv`: medias y desvíos reutilizables;
- `generated/varianza_explicada.svg`, `generated/cargas.svg` y
  `generated/biplot.svg`: visualizaciones del análisis.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def run_analysis(data: Path, output: Path, threshold: float) -> tuple[PCAResult, int]:
    dataset = load_creator_dataset(data)
    result = fit_pca(dataset.values, dataset.feature_names)
    selected_count = result.component_count(threshold)
    write_tables(output, dataset, result)
    write_plots(output, dataset, result, selected_count)
    write_report(output, dataset, result, threshold, selected_count)
    return result, selected_count


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result, selected_count = run_analysis(
        args.data, args.output, args.variance_threshold
    )
    print(
        f"PCA completo: {selected_count} componentes alcanzan "
        f"{result.cumulative_explained_variance[selected_count - 1] * 100:.2f} %; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
