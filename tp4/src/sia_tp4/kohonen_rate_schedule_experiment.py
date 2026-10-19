"""Comparar tasas constantes y decrecientes para todos los pares de k y eta."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen_analysis import KohonenRun, load_run
from .kohonen_rate_experiment import evaluate_runs, load_rate_runs
from .pca_analysis import _load_pyplot, _save_figure


@dataclass(frozen=True)
class ScheduleRunMetrics:
    """Indicadores de una corrida identificada por su schedule de tasa."""

    run: str
    schedule: str
    grid_size: int
    learning_rate_start: float
    learning_rate_end: float
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
        "--constant-configs",
        type=Path,
        default=Path("configs/kohonen/02-rate"),
        help="Directorio del barrido con tasa constante.",
    )
    parser.add_argument(
        "--decay-configs",
        type=Path,
        default=Path("configs/kohonen/03-rate-decay"),
        help="Directorio del barrido con tasa decreciente.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/03-rate-schedule"),
        help="Directorio del informe comparativo.",
    )
    return parser.parse_args(argv)


def load_decay_runs(directory: Path | str) -> list[KohonenRun]:
    """Cargar corridas cuya tasa final sea un décimo de la inicial."""

    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no hay configuraciones JSON en {directory}")
    runs = [load_run(path) for path in paths]
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("los nombres de las corridas deben ser únicos")

    reference = runs[0].config
    controlled_fields = ("epochs", "radius_end", "initialization", "shuffle")
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
        if not np.isclose(
            run.config.learning_rate_end,
            run.config.learning_rate_start * 0.1,
        ):
            raise ValueError(f"'{run.name}' no disminuye la tasa al 10 %")
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


def validate_matching_designs(
    constant_runs: Sequence[KohonenRun], decay_runs: Sequence[KohonenRun]
) -> None:
    """Exigir los mismos k, tasas iniciales y semillas en ambos schedules."""

    def design(runs):
        return {
            (
                run.config.grid_size,
                run.config.learning_rate_start,
                run.config.seed,
            )
            for run in runs
        }

    if design(constant_runs) != design(decay_runs):
        raise ValueError("los diseños constante y decreciente no coinciden")


def evaluate_schedule(
    dataset: CreatorDataset,
    runs: Sequence[KohonenRun],
    schedule: str,
) -> list[ScheduleRunMetrics]:
    base_metrics = evaluate_runs(dataset, runs)
    runs_by_name = {run.name: run for run in runs}
    return [
        ScheduleRunMetrics(
            run=metric.run,
            schedule=schedule,
            grid_size=metric.grid_size,
            learning_rate_start=metric.learning_rate,
            learning_rate_end=runs_by_name[metric.run].config.learning_rate_end,
            seed=metric.seed,
            quantization_error=metric.quantization_error,
            occupied_units=metric.occupied_units,
            dead_units=metric.dead_units,
            dead_units_pct=metric.dead_units_pct,
            maximum_hits=metric.maximum_hits,
            maximum_hits_pct=metric.maximum_hits_pct,
        )
        for metric in base_metrics
    ]


def aggregate_metrics(metrics: Sequence[ScheduleRunMetrics]) -> list[dict]:
    """Calcular media y desvío por schedule, k y tasa inicial."""

    rows = []
    combinations = sorted(
        {
            (metric.schedule, metric.grid_size, metric.learning_rate_start)
            for metric in metrics
        }
    )
    for schedule, grid_size, learning_rate_start in combinations:
        group = [
            metric
            for metric in metrics
            if metric.schedule == schedule
            and metric.grid_size == grid_size
            and metric.learning_rate_start == learning_rate_start
        ]
        if len(group) < 2:
            raise ValueError("cada combinación necesita al menos dos semillas")
        row = {
            "schedule": schedule,
            "grid_size": grid_size,
            "learning_rate_start": learning_rate_start,
            "learning_rate_end": group[0].learning_rate_end,
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


def schedule_differences(aggregate: Sequence[dict]) -> list[dict]:
    """Restar constante a decreciente para cada k y tasa inicial."""

    rows = []
    grid_sizes = sorted({row["grid_size"] for row in aggregate})
    rates = sorted({row["learning_rate_start"] for row in aggregate})
    for grid_size in grid_sizes:
        for rate in rates:
            constant = next(
                row
                for row in aggregate
                if row["schedule"] == "constant"
                and row["grid_size"] == grid_size
                and row["learning_rate_start"] == rate
            )
            decay = next(
                row
                for row in aggregate
                if row["schedule"] == "decay"
                and row["grid_size"] == grid_size
                and row["learning_rate_start"] == rate
            )
            rows.append(
                {
                    "grid_size": grid_size,
                    "learning_rate_start": rate,
                    "quantization_error_delta": (
                        decay["quantization_error_mean"]
                        - constant["quantization_error_mean"]
                    ),
                    "dead_units_pct_delta": (
                        decay["dead_units_pct_mean"]
                        - constant["dead_units_pct_mean"]
                    ),
                    "maximum_hits_pct_delta": (
                        decay["maximum_hits_pct_mean"]
                        - constant["maximum_hits_pct_mean"]
                    ),
                }
            )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    metrics: Sequence[ScheduleRunMetrics],
    aggregate: Sequence[dict],
    differences: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "corridas.csv",
        list(ScheduleRunMetrics.__dataclass_fields__),
        (vars(metric) for metric in metrics),
    )
    _write_csv(
        generated / "resumen_schedules.csv",
        list(aggregate[0]),
        aggregate,
    )
    _write_csv(
        generated / "diferencias_decay_menos_constante.csv",
        list(differences[0]),
        differences,
    )


def _save_both(figure, path: Path) -> None:
    _save_figure(figure, path.with_suffix(".svg"))
    figure.savefig(
        path.with_suffix(".png"),
        format="png",
        dpi=160,
        bbox_inches="tight",
    )


def write_error_curves(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    grid_sizes = sorted({row["grid_size"] for row in aggregate})
    rates = sorted({row["learning_rate_start"] for row in aggregate})
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)
    styles = (("constant", "Constante", "-"), ("decay", "Decreciente", "--"))
    for axis, grid_size in zip(axes.flat, grid_sizes):
        for schedule, label, linestyle in styles:
            rows = [
                row
                for row in aggregate
                if row["grid_size"] == grid_size
                and row["schedule"] == schedule
            ]
            axis.errorbar(
                rates,
                [row["quantization_error_mean"] for row in rows],
                yerr=[row["quantization_error_std"] for row in rows],
                marker="o",
                capsize=3,
                linestyle=linestyle,
                label=label,
            )
        axis.set_title(f"k={grid_size}")
        axis.set_xlabel("Tasa inicial η")
        axis.set_ylabel("Error de cuantización")
        axis.set_xticks(rates)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(alpha=0.25)
        axis.legend()
    axes.flat[-1].set_visible(False)
    figure.suptitle("Tasa constante frente a decreciente — media ± desvío")
    figure.tight_layout()
    _save_both(figure, output / "generated" / "curvas_schedules")
    plt.close(figure)


def write_difference_heatmaps(output: Path, differences: Sequence[dict]) -> None:
    plt = _load_pyplot()
    grid_sizes = sorted({row["grid_size"] for row in differences})
    rates = sorted({row["learning_rate_start"] for row in differences})
    definitions = (
        ("quantization_error_delta", "Error de cuantización"),
        ("dead_units_pct_delta", "Unidades muertas (p.p.)"),
        ("maximum_hits_pct_delta", "Concentración máxima (p.p.)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, (field, title) in zip(axes, definitions):
        matrix = np.asarray(
            [
                [
                    next(
                        row[field]
                        for row in differences
                        if row["grid_size"] == grid_size
                        and row["learning_rate_start"] == rate
                    )
                    for rate in rates
                ]
                for grid_size in grid_sizes
            ]
        )
        limit = float(np.max(np.abs(matrix))) or 1.0
        image = axis.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        axis.set(
            title=title,
            xlabel="Tasa inicial η",
            ylabel="Tamaño k",
            xticks=np.arange(len(rates)),
            xticklabels=[f"{rate:g}" for rate in rates],
            yticks=np.arange(len(grid_sizes)),
            yticklabels=grid_sizes,
        )
        figure.colorbar(image, ax=axis, label="Decreciente − constante")
    figure.suptitle(
        "Efecto de disminuir η al 10 % — valores negativos favorecen al descenso"
    )
    figure.tight_layout()
    _save_both(figure, output / "generated" / "diferencias_schedules")
    plt.close(figure)


def _minimum_error_rows(aggregate: Sequence[dict]) -> list[dict]:
    rows = []
    for schedule in ("constant", "decay"):
        for grid_size in sorted({row["grid_size"] for row in aggregate}):
            group = [
                row
                for row in aggregate
                if row["schedule"] == schedule
                and row["grid_size"] == grid_size
            ]
            rows.append(min(group, key=lambda row: row["quantization_error_mean"]))
    return rows


def write_report(output: Path, aggregate: Sequence[dict]) -> None:
    table_rows = "\n".join(
        "| {schedule} | {grid_size} | {learning_rate_start:g} → "
        "{learning_rate_end:g} | {quantization_error_mean:.4f} ± "
        "{quantization_error_std:.4f} | {dead_units_pct_mean:.1f} ± "
        "{dead_units_pct_std:.1f} | {maximum_hits_pct_mean:.1f} ± "
        "{maximum_hits_pct_std:.1f} |".format(**row)
        for row in _minimum_error_rows(aggregate)
    )
    report = f"""# Experimento 3: tasa constante frente a decreciente

Se compararon los mismos 245 pares de `k`, tasa inicial y semilla con dos
schedules: tasa constante y disminución lineal hasta el 10 % del valor inicial.
En total se entrenaron 490 redes con 150 épocas. El radio mantuvo en ambos casos
su descenso desde el diámetro de la grilla hasta 1, por lo que la única
diferencia experimental es la evolución de la tasa.

![Curvas de error por schedule](generated/curvas_schedules.png)

Los siguientes mapas muestran `decreciente − constante` para cada indicador.
Un valor negativo favorece al descenso; uno positivo favorece mantener la tasa.

![Diferencias entre schedules](generated/diferencias_schedules.png)

## Mínimo error dentro de cada tamaño y schedule

| Schedule | k | Tasa | Error | Unidades muertas (%) | Máximo de hits (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
{table_rows}

Esta tabla sólo localiza mínimos de error. En particular, los mínimos de
`k = 5`, `k = 7` y `k = 8` con descenso dejan muchas unidades muertas; no se
seleccionan automáticamente.

## Lectura de la interacción

- En `k = 4`, el descenso vuelve competitivas las tasas iniciales altas. La
  región `0.3 → 0.03` logra error `2.5533` con sólo `1.25 %` de unidades
  muertas; `0.7 → 0.07` baja el error a `2.5448`, pero aumenta las unidades
  muertas a `3.75 %`. La constante `0.05` sigue siendo un control fuerte por su
  buena ocupación y estabilidad.
- En `k = 5`, `0.7 → 0.07` logra un compromiso cercano a la constante `0.2`:
  errores `2.3082` frente a `2.3207`, unidades muertas `5.6 %` frente a `4.8 %`
  y la misma concentración máxima de `8.8 %`. El mínimo `0.9 → 0.09` reduce el
  error, pero eleva las unidades muertas a `17.6 %`.
- En `k = 6`, el descenso no domina a las tasas constantes bajas. La constante
  `0.1` conserva `16.7 %` de unidades muertas; los mejores descensos quedan
  alrededor de `20–21 %`. Se mantiene como control de una grilla más grande.
- `k = 2` y `k = 3` continúan demasiado concentrados. `k = 7` y `k = 8`
  continúan demasiado fragmentados incluso cuando el descenso mejora su error.

Por lo tanto no se elige aún una única configuración. Para estudiar el radio y
las épocas se conservan regiones representativas de `k = 4`, `k = 5` y `k = 6`,
incluyendo alternativas constantes y decrecientes. Así se mantiene visible la
interacción sin volver a arrastrar combinaciones claramente dominadas.

Los datos están en `generated/corridas.csv`, los promedios en
`generated/resumen_schedules.csv` y las diferencias emparejadas por combinación
en `generated/diferencias_decay_menos_constante.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_creator_dataset(args.data)
    constant_runs = load_rate_runs(args.constant_configs)
    decay_runs = load_decay_runs(args.decay_configs)
    validate_matching_designs(constant_runs, decay_runs)
    metrics = [
        *evaluate_schedule(dataset, constant_runs, "constant"),
        *evaluate_schedule(dataset, decay_runs, "decay"),
    ]
    aggregate = aggregate_metrics(metrics)
    differences = schedule_differences(aggregate)
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(args.output, metrics, aggregate, differences)
    write_error_curves(args.output, aggregate)
    write_difference_heatmaps(args.output, differences)
    write_report(args.output, aggregate)
    print(f"Informe de schedules generado en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
