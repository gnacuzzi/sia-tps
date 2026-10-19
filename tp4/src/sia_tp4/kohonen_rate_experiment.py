"""Comparar conjuntamente tamaños de grilla y tasas constantes de Kohonen."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen_analysis import KohonenRun, load_run, run_kohonen
from .pca_analysis import _load_pyplot, _save_figure


@dataclass(frozen=True)
class RateRunMetrics:
    """Indicadores de una combinación individual de k, tasa y semilla."""

    run: str
    grid_size: int
    learning_rate: float
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
        default=Path("configs/kohonen/02-rate"),
        help="Directorio con un JSON por corrida.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/02-rate"),
        help="Directorio del informe comparativo.",
    )
    return parser.parse_args(argv)


def load_rate_runs(directory: Path | str) -> list[KohonenRun]:
    """Cargar y validar un barrido controlado de k por tasa constante."""

    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no hay configuraciones JSON en {directory}")
    runs = [load_run(path) for path in paths]
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("los nombres de las corridas deben ser únicos")

    reference = runs[0].config
    controlled_fields = (
        "epochs",
        "radius_end",
        "initialization",
        "shuffle",
    )
    combinations = set()
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
        if run.config.learning_rate_start != run.config.learning_rate_end:
            raise ValueError(f"'{run.name}' no usa una tasa constante")
        expected_radius = np.sqrt(2) * (run.config.grid_size - 1)
        if not np.isclose(run.config.radius_start, expected_radius):
            raise ValueError(
                f"'{run.name}' debe usar el diámetro de su grilla como radio inicial"
            )
        combination = (
            run.config.grid_size,
            run.config.learning_rate_start,
            run.config.seed,
        )
        if combination in combinations:
            raise ValueError(f"combinación repetida en '{run.name}'")
        combinations.add(combination)
    return runs


def evaluate_runs(
    dataset: CreatorDataset, runs: Sequence[KohonenRun]
) -> list[RateRunMetrics]:
    """Entrenar cada configuración y extraer indicadores comparables."""

    metrics = []
    sample_count = dataset.values.shape[0]
    for run in runs:
        result = run_kohonen(dataset, run)
        total_units = result.hit_map.size
        metrics.append(
            RateRunMetrics(
                run=run.name,
                grid_size=run.config.grid_size,
                learning_rate=run.config.learning_rate_start,
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


def aggregate_metrics(metrics: Sequence[RateRunMetrics]) -> list[dict]:
    """Calcular media y desvío por cada par k y tasa."""

    rows = []
    combinations = sorted(
        {(metric.grid_size, metric.learning_rate) for metric in metrics}
    )
    for grid_size, learning_rate in combinations:
        group = [
            metric
            for metric in metrics
            if metric.grid_size == grid_size
            and metric.learning_rate == learning_rate
        ]
        if len(group) < 2:
            raise ValueError("cada combinación de k y tasa necesita dos semillas")
        row = {
            "grid_size": grid_size,
            "learning_rate": learning_rate,
            "runs": len(group),
        }
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


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    metrics: Sequence[RateRunMetrics],
    aggregate: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "corridas.csv",
        list(RateRunMetrics.__dataclass_fields__),
        (vars(metric) for metric in metrics),
    )
    _write_csv(
        generated / "resumen_k_tasa.csv",
        list(aggregate[0]),
        aggregate,
    )


_PLOT_DEFINITIONS = (
    (
        "quantization_error",
        "Error de cuantización",
        "Distancia media a la BMU",
    ),
    (
        "dead_units_pct",
        "Unidades muertas",
        "Porcentaje de la grilla",
    ),
    (
        "maximum_hits_pct",
        "Concentración máxima",
        "Muestras en la neurona más ocupada (%)",
    ),
)


def write_curve_plot(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    grid_sizes = sorted({row["grid_size"] for row in aggregate})
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, (field, title, ylabel) in zip(axes, _PLOT_DEFINITIONS):
        for grid_size in grid_sizes:
            rows = [row for row in aggregate if row["grid_size"] == grid_size]
            rates = np.asarray([row["learning_rate"] for row in rows])
            means = np.asarray([row[f"{field}_mean"] for row in rows])
            deviations = np.asarray([row[f"{field}_std"] for row in rows])
            axis.errorbar(
                rates,
                means,
                yerr=deviations,
                marker="o",
                capsize=3,
                linewidth=1.2,
                label=f"k={grid_size}",
            )
        axis.set(title=title, xlabel="Tasa constante η", ylabel=ylabel)
        axis.set_xticks(sorted({row["learning_rate"] for row in aggregate}))
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(alpha=0.25)
    axes[-1].legend(title="Grilla", bbox_to_anchor=(1.04, 1), loc="upper left")
    figure.suptitle("Interacción entre tamaño de grilla y tasa constante")
    figure.tight_layout()
    _save_figure(figure, output / "generated" / "curvas_k_tasa.svg")
    figure.savefig(
        output / "generated" / "curvas_k_tasa.png",
        format="png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_heatmap_plot(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    grid_sizes = sorted({row["grid_size"] for row in aggregate})
    rates = sorted({row["learning_rate"] for row in aggregate})
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, (field, title, colorbar_label) in zip(axes, _PLOT_DEFINITIONS):
        matrix = np.asarray(
            [
                [
                    next(
                        row[f"{field}_mean"]
                        for row in aggregate
                        if row["grid_size"] == grid_size
                        and row["learning_rate"] == rate
                    )
                    for rate in rates
                ]
                for grid_size in grid_sizes
            ]
        )
        image = axis.imshow(matrix, cmap="viridis", aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if matrix[row_index, column_index] > np.mean(matrix)
                        else "black"
                    ),
                )
        axis.set(
            title=title,
            xlabel="Tasa constante η",
            ylabel="Tamaño k",
            xticks=np.arange(len(rates)),
            xticklabels=[f"{rate:g}" for rate in rates],
            yticks=np.arange(len(grid_sizes)),
            yticklabels=grid_sizes,
        )
        figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.suptitle("Promedios de cinco semillas para cada combinación")
    figure.tight_layout()
    _save_figure(figure, output / "generated" / "mapas_calor_k_tasa.svg")
    figure.savefig(
        output / "generated" / "mapas_calor_k_tasa.png",
        format="png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def _best_rows_by_grid(aggregate: Sequence[dict]) -> list[dict]:
    rows = []
    for grid_size in sorted({row["grid_size"] for row in aggregate}):
        group = [row for row in aggregate if row["grid_size"] == grid_size]
        rows.append(min(group, key=lambda row: row["quantization_error_mean"]))
    return rows


def write_report(output: Path, aggregate: Sequence[dict]) -> None:
    best_rows = _best_rows_by_grid(aggregate)
    table_rows = "\n".join(
        "| {grid_size} | {learning_rate:g} | {quantization_error_mean:.4f} ± "
        "{quantization_error_std:.4f} | {dead_units_pct_mean:.1f} ± "
        "{dead_units_pct_std:.1f} | {maximum_hits_pct_mean:.1f} ± "
        "{maximum_hits_pct_std:.1f} |".format(**row)
        for row in best_rows
    )
    report = f"""# Experimento 2: tamaño de grilla por tasa constante

Se evaluaron todos los pares entre `k = 2, ..., 8` y tasas constantes
`η = 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9`, con cinco semillas por combinación:
245 corridas independientes. Se mantuvieron fijos las 150 épocas, el radio
decreciente desde el diámetro de la grilla hasta 1, la inicialización por
muestras y la mezcla del orden de presentación.

Este diseño permite observar la interacción entre `k` y `η`: el primer barrido
de grilla, realizado con tasa decreciente `0.5 → 0.05`, no alcanzaba para
descartar que otro tamaño mejorara al cambiar la tasa.

![Curvas por tamaño y tasa](generated/curvas_k_tasa.png)

![Mapas de calor de k por tasa](generated/mapas_calor_k_tasa.png)

## Mejor error de cuantización dentro de cada tamaño

La tabla muestra el mínimo error medio encontrado para cada `k`, pero no elige
la configuración únicamente por esa cifra. Las unidades muertas y la
concentración de hits permiten detectar soluciones demasiado fragmentadas o
demasiado agrupadas.

| k | η | Error de cuantización | Unidades muertas (%) | Máximo de hits (%) |
| ---: | ---: | ---: | ---: | ---: |
{table_rows}

## Lectura de los resultados

Las tasas altas (`0.5`, `0.7` y `0.9`) empeoran el error y, en general, también
la estabilidad: con una tasa constante los pesos continúan siguiendo con fuerza
cada muestra hasta el final. La zona útil se concentra entre `0.05` y `0.2`
para las grillas que todavía mantienen una ocupación razonable.

- Para `k = 4`, `η = 0.05` y `η = 0.1` tienen prácticamente el mismo error;
  `0.05` deja menos unidades muertas y menor concentración máxima.
- Para `k = 5`, `η = 0.1` minimiza el error, pero `η = 0.2` reduce las unidades
  muertas de 12,0 % a 4,8 % y la concentración de 10,4 % a 8,8 %, a cambio de
  un aumento pequeño del error.
- Para `k = 6`, `η = 0.1` conserva mejor ocupación que `0.2` con el mismo máximo
  medio de hits y queda como control para no descartar ese tamaño antes de
  estudiar la disminución de la tasa.
- `k = 2` y `k = 3` siguen concentrando demasiado; `k = 7` y `k = 8` mantienen
  entre aproximadamente 33 % y 49 % de unidades muertas aun en sus mejores
  zonas. La tasa no corrige esas limitaciones estructurales.

Por lo tanto, continúan `k = 4` con `η = 0.05`, `k = 5` con `η = 0.2` y
`k = 6` con `η = 0.1`. Son tres puntos de partida para comparar una tasa
constante contra una decreciente; todavía no constituyen una selección final.

## Criterio para continuar

Una tasa pequeña puede requerir más iteraciones para organizar el mapa; una
tasa cercana a 1 hace que los pesos sigan demasiado cada muestra y puede
aumentar la variabilidad. Se conservarán las regiones de tasa que tengan buen
error, ocupación razonable y poca dispersión entre semillas. Después se
comparará tasa constante contra tasa decreciente sólo sobre varios candidatos,
sin dar por definitivo un único `k`.

Los resultados de las 245 corridas están en `generated/corridas.csv` y los
promedios por combinación en `generated/resumen_k_tasa.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_creator_dataset(args.data)
    runs = load_rate_runs(args.configs)
    metrics = evaluate_runs(dataset, runs)
    aggregate = aggregate_metrics(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(args.output, metrics, aggregate)
    write_curve_plot(args.output, aggregate)
    write_heatmap_plot(args.output, aggregate)
    write_report(args.output, aggregate)
    print(f"Informe de k por tasa generado en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
