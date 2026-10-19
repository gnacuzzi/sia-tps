"""Comparar tamaños de grilla de Kohonen mediante corridas independientes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen_analysis import (
    KohonenRun,
    KohonenRunResult,
    load_run,
    run_kohonen,
)
from .pca_analysis import _load_pyplot, _save_figure


@dataclass(frozen=True)
class GridRunMetrics:
    """Indicadores de una corrida individual del barrido de grilla."""

    run: str
    grid_size: int
    seed: int
    quantization_error: float
    occupied_units: int
    dead_units: int
    dead_units_pct: float
    maximum_hits: int
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
        default=Path("configs/kohonen/01-grid"),
        help="Directorio con un JSON por corrida.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/01-grid"),
        help="Directorio del informe comparativo.",
    )
    return parser.parse_args(argv)


def load_grid_runs(directory: Path | str) -> list[KohonenRun]:
    """Cargar las corridas y verificar que formen un experimento controlado."""

    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no hay configuraciones JSON en {directory}")

    runs = [load_run(path) for path in paths]
    names = [run.name for run in runs]
    if len(names) != len(set(names)):
        raise ValueError("los nombres de las corridas deben ser únicos")

    reference = runs[0].config
    controlled_fields = (
        "epochs",
        "learning_rate_start",
        "learning_rate_end",
        "radius_end",
        "initialization",
        "shuffle",
    )
    for run in runs:
        changed = [
            field
            for field in controlled_fields
            if getattr(run.config, field) != getattr(reference, field)
        ]
        if changed:
            raise ValueError(
                f"'{run.name}' cambia parámetros controlados: {changed}"
            )
        expected_radius = np.sqrt(2) * (run.config.grid_size - 1)
        if not np.isclose(run.config.radius_start, expected_radius):
            raise ValueError(
                f"'{run.name}' debe usar el diámetro de su grilla como radio inicial"
            )
    return runs


def evaluate_runs(
    dataset: CreatorDataset, runs: Sequence[KohonenRun]
) -> list[GridRunMetrics]:
    """Entrenar cada JSON y extraer indicadores comparables."""

    metrics = []
    sample_count = dataset.values.shape[0]
    for run in runs:
        result = run_kohonen(dataset, run)
        total_units = result.hit_map.size
        metrics.append(
            GridRunMetrics(
                run=run.name,
                grid_size=run.config.grid_size,
                seed=run.config.seed,
                quantization_error=result.quantization_error,
                occupied_units=result.occupied_units,
                dead_units=result.dead_units,
                dead_units_pct=100 * result.dead_units / total_units,
                maximum_hits=int(result.hit_map.max()),
                maximum_hits_pct=100 * int(result.hit_map.max()) / sample_count,
            )
        )
    return metrics


def aggregate_metrics(metrics: Sequence[GridRunMetrics]) -> list[dict]:
    """Calcular media y desvío muestral de cada indicador por tamaño."""

    rows = []
    for grid_size in sorted({metric.grid_size for metric in metrics}):
        group = [metric for metric in metrics if metric.grid_size == grid_size]
        if len(group) < 2:
            raise ValueError("cada tamaño de grilla necesita al menos dos semillas")
        row = {"grid_size": grid_size, "runs": len(group)}
        for field in (
            "quantization_error",
            "occupied_units",
            "dead_units",
            "dead_units_pct",
            "maximum_hits",
            "maximum_hits_pct",
        ):
            values = np.asarray([getattr(metric, field) for metric in group])
            row[f"{field}_mean"] = float(np.mean(values))
            row[f"{field}_std"] = float(np.std(values, ddof=1))
        rows.append(row)
    return rows


def select_representative_runs(
    runs: Sequence[KohonenRun], metrics: Sequence[GridRunMetrics]
) -> list[KohonenRun]:
    """Elegir por cada k la semilla cuyo error quede más cerca de la mediana."""

    runs_by_name = {run.name: run for run in runs}
    selected = []
    for grid_size in sorted({metric.grid_size for metric in metrics}):
        group = [metric for metric in metrics if metric.grid_size == grid_size]
        median = float(np.median([metric.quantization_error for metric in group]))
        representative = min(
            group,
            key=lambda metric: (
                abs(metric.quantization_error - median),
                metric.run,
            ),
        )
        selected.append(runs_by_name[representative.run])
    return selected


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    metrics: Sequence[GridRunMetrics],
    aggregate: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    metric_fields = list(GridRunMetrics.__dataclass_fields__)
    _write_csv(
        generated / "corridas.csv",
        metric_fields,
        (vars(metric) for metric in metrics),
    )
    _write_csv(
        generated / "resumen_por_k.csv",
        list(aggregate[0]),
        aggregate,
    )


def write_representative_tables(
    output: Path, results: Sequence[KohonenRunResult]
) -> None:
    rows = []
    for result in results:
        for row in range(result.hit_map.shape[0]):
            for column in range(result.hit_map.shape[1]):
                rows.append(
                    {
                        "run": result.run.name,
                        "grid_size": result.run.config.grid_size,
                        "seed": result.run.config.seed,
                        "row": row,
                        "column": column,
                        "hits": int(result.hit_map[row, column]),
                        "u_distance": float(result.u_matrix[row, column]),
                    }
                )
    _write_csv(
        output / "generated" / "matrices_representativas.csv",
        [
            "run",
            "grid_size",
            "seed",
            "row",
            "column",
            "hits",
            "u_distance",
        ],
        rows,
    )


def write_plot(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    grid_sizes = np.asarray([row["grid_size"] for row in aggregate])
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    definitions = (
        (
            "quantization_error",
            "Error de cuantización",
            "Distancia media a la BMU (menor es mejor)",
        ),
        (
            "dead_units_pct",
            "Unidades muertas",
            "Porcentaje de la grilla (menor es mejor)",
        ),
        (
            "maximum_hits_pct",
            "Concentración máxima",
            "Muestras en la neurona más ocupada (%)",
        ),
        (
            "occupied_units",
            "Unidades ocupadas",
            "Cantidad media de neuronas con hits",
        ),
    )
    for axis, (field, title, ylabel) in zip(axes.flat, definitions):
        means = np.asarray([row[f"{field}_mean"] for row in aggregate])
        deviations = np.asarray([row[f"{field}_std"] for row in aggregate])
        axis.errorbar(
            grid_sizes,
            means,
            yerr=deviations,
            marker="o",
            capsize=4,
            linewidth=1.5,
        )
        axis.set(title=title, xlabel="Tamaño k de la grilla k × k", ylabel=ylabel)
        axis.set_xticks(grid_sizes)
        axis.grid(alpha=0.25)
    figure.suptitle("Barrido de tamaño de grilla — media ± desvío (5 semillas)")
    figure.tight_layout()
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _save_figure(figure, generated / "comparacion_k.svg")
    plt.close(figure)


def _representative_figure(plt, results, matrix_name, title, colorbar_label):
    matrices = [getattr(result, matrix_name) for result in results]
    values = np.concatenate([matrix.ravel() for matrix in matrices])
    minimum = 0.0 if matrix_name == "hit_map" else float(values.min())
    maximum = float(values.max())
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    colored = None
    for axis, result, matrix in zip(axes.flat, results, matrices):
        size = matrix.shape[0]
        rows, columns = np.indices(matrix.shape)
        colored = axis.scatter(
            columns.ravel(),
            rows.ravel(),
            c=matrix.ravel(),
            s=(250 / size) ** 2,
            cmap="turbo",
            vmin=minimum,
            vmax=maximum,
            edgecolors="black",
            linewidths=0.6,
        )
        for row in range(size):
            for column in range(size):
                label = (
                    f"{int(matrix[row, column])}"
                    if matrix_name == "hit_map"
                    else f"{matrix[row, column]:.1f}"
                )
                axis.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=max(6, 10 - size // 2),
                )
        axis.set_title(
            f"k={size}, semilla={result.run.config.seed}\n"
            f"QE={result.quantization_error:.3f}"
        )
        axis.set_xticks(range(size))
        axis.set_yticks(range(size))
        axis.set_xlim(-0.6, size - 0.4)
        axis.set_ylim(size - 0.4, -0.6)
        axis.set_aspect("equal")
    axes.flat[-1].set_visible(False)
    figure.suptitle(title)
    figure.colorbar(
        colored,
        ax=[axis for axis in axes.flat if axis.get_visible()],
        label=colorbar_label,
        shrink=0.85,
    )
    return figure


def write_representative_plots(
    output: Path, results: Sequence[KohonenRunResult]
) -> None:
    plt = _load_pyplot()
    definitions = (
        (
            "hit_map",
            "Mapa de hits representativo de cada tamaño",
            "Creadores asignados a la neurona",
            "mapas_hits_representativos.svg",
        ),
        (
            "u_matrix",
            "Matriz U representativa de cada tamaño",
            "Distancia media a las neuronas vecinas",
            "matrices_u_representativas.svg",
        ),
    )
    for matrix_name, title, colorbar_label, filename in definitions:
        figure = _representative_figure(
            plt, results, matrix_name, title, colorbar_label
        )
        _save_figure(figure, output / "generated" / filename)
        figure.savefig(
            output / "generated" / filename.replace(".svg", ".png"),
            format="png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)


def write_report(
    output: Path,
    aggregate: Sequence[dict],
    representative_results: Sequence[KohonenRunResult],
) -> None:
    table_rows = "\n".join(
        "| {grid_size} | {quantization_error_mean:.4f} ± "
        "{quantization_error_std:.4f} | {dead_units_pct_mean:.1f} ± "
        "{dead_units_pct_std:.1f} | {maximum_hits_pct_mean:.1f} ± "
        "{maximum_hits_pct_std:.1f} |".format(**row)
        for row in aggregate
    )
    representative_rows = "\n".join(
        f"| {result.run.config.grid_size} | {result.run.config.seed} | "
        f"{result.quantization_error:.4f} | `{result.run.name}` |"
        for result in representative_results
    )
    report = f"""# Experimento 1: tamaño de la grilla de Kohonen

Se compararon grillas desde `2 × 2` hasta `8 × 8`, con cinco semillas por
tamaño. Cada una de las 35 corridas está definida en un JSON independiente.
Se mantuvieron fijos las épocas, las tasas, el radio final, la inicialización y
la mezcla. El radio inicial se normalizó al diámetro de cada grilla para que al
comenzar todas puedan actualizar el mapa completo.

| k | Error de cuantización | Unidades muertas (%) | Máximo de hits (%) |
| ---: | ---: | ---: | ---: |
{table_rows}

![Comparación de tamaños de grilla](generated/comparacion_k.svg)

## Una corrida representativa por tamaño

Para visualizar cada `k` sin elegir una semilla arbitraria, se tomó la corrida
cuyo error de cuantización queda más cerca de la mediana de sus cinco semillas.
La selección sólo decide qué mapa mostrar: las curvas anteriores siguen usando
las 35 corridas.

| k | Semilla | Error de cuantización | Corrida |
| ---: | ---: | ---: | --- |
{representative_rows}

En el mapa de hits, el número y el color de cada círculo indican cuántos
creadores ganaron esa neurona. Un `0` identifica una unidad muerta.

![Mapas de hits representativos](generated/mapas_hits_representativos.png)

La matriz U usa la misma disposición, pero el color representa la distancia
media entre los pesos de una neurona y los de sus vecinas ortogonales. Valores
altos pueden señalar separaciones entre regiones del mapa.

![Matrices U representativas](generated/matrices_u_representativas.png)

## Cómo se toma la decisión

El error de cuantización mide, en el espacio estandarizado, la distancia media
entre cada creador y su BMU. Es una métrica auxiliar derivada de la misma
distancia euclídea usada por el algoritmo; no fue presentada como criterio
único en las clases. Agregar neuronas suele reducirla, por lo que no esperamos
necesariamente un pico. Buscamos el punto a partir del cual su mejora se vuelve
pequeña mientras aumentan las unidades muertas o la solución se fragmenta.

Las barras muestran un desvío muestral alrededor de la media de cinco semillas.
Así se evita elegir un `k` por una inicialización afortunada. Este barrido sirve
para descartar extremos y conservar dos candidatos. La decisión final exige
entrenar esos candidatos con los demás hiperparámetros y revisar sus mapas de
hits, matrices U y mapas por variable; este informe no fija todavía un `k`.

## Lectura preliminar

Las grillas `2 × 2` y `3 × 3` quedan demasiado concentradas: su neurona más
ocupada reúne en promedio 45,2 % y 17,2 % de los creadores. En `4 × 4` y
`5 × 5` esa concentración baja a 11,6 % y 8,8 %, manteniendo las unidades
muertas en 6,2 % y 9,6 %. Al pasar a `6 × 6`, la concentración apenas mejora
0,4 puntos porcentuales, pero las unidades muertas saltan a 21,1 %. Desde allí
siguen aumentando hasta 46,9 % en `8 × 8`.

Por este compromiso, `k = 4` y `k = 5` quedan como candidatos para la siguiente
etapa. No se declara todavía un ganador: hay que comprobar si la conclusión se
mantiene al variar la tasa, el radio y las épocas, y después inspeccionar los
mapas resultantes.

Los valores de todas las corridas están en `generated/corridas.csv` y los
promedios por tamaño en `generated/resumen_por_k.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_creator_dataset(args.data)
    runs = load_grid_runs(args.configs)
    metrics = evaluate_runs(dataset, runs)
    aggregate = aggregate_metrics(metrics)
    representative_runs = select_representative_runs(runs, metrics)
    representative_results = [
        run_kohonen(dataset, run) for run in representative_runs
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(args.output, metrics, aggregate)
    write_representative_tables(args.output, representative_results)
    write_plot(args.output, aggregate)
    write_representative_plots(args.output, representative_results)
    write_report(args.output, aggregate, representative_results)
    print(f"Informe del barrido de grilla generado en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
