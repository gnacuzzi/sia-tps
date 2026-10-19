"""Comparar radios iniciales para las regiones útiles de k y tasa."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .kohonen_analysis import KohonenRun, load_run, run_kohonen
from .pca_analysis import _load_pyplot, _save_figure


EXPECTED_RATE_PROFILES = {
    4: {(0.05, 0.05), (0.1, 0.1), (0.3, 0.03), (0.7, 0.07)},
    5: {(0.1, 0.1), (0.2, 0.2), (0.7, 0.07), (0.9, 0.09)},
    6: {(0.1, 0.1), (0.2, 0.2), (0.7, 0.07), (0.9, 0.09)},
}
EXPECTED_RADIUS_FRACTIONS = {0.0, 0.25, 0.5, 0.75, 1.0}


@dataclass(frozen=True)
class RadiusRunMetrics:
    """Indicadores y tiempo de una corrida del barrido de radio."""

    run: str
    grid_size: int
    schedule: str
    learning_rate_start: float
    learning_rate_end: float
    radius_fraction: float
    radius_start: float
    radius_end: float
    seed: int
    training_seconds: float
    quantization_error: float
    occupied_units: int
    dead_units: int
    dead_units_pct: float
    maximum_hits: int
    maximum_hits_pct: float
    mean_u_distance: float


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
        default=Path("configs/kohonen/04-radius"),
        help="Directorio con un JSON por corrida.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/04-radius"),
        help="Directorio del informe comparativo.",
    )
    return parser.parse_args(argv)


def _radius_fraction(run: KohonenRun) -> float:
    diameter = np.sqrt(2) * (run.config.grid_size - 1)
    return float((run.config.radius_start - 1) / (diameter - 1))


def _schedule(run: KohonenRun) -> str:
    return (
        "constant"
        if run.config.learning_rate_start == run.config.learning_rate_end
        else "decay"
    )


def load_radius_runs(directory: Path | str) -> list[KohonenRun]:
    """Cargar y validar el diseño controlado de radios iniciales."""

    directory = Path(directory)
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"no hay configuraciones JSON en {directory}")
    runs = [load_run(path) for path in paths]
    if len({run.name for run in runs}) != len(runs):
        raise ValueError("los nombres de las corridas deben ser únicos")

    combinations = set()
    for run in runs:
        config = run.config
        if config.grid_size not in EXPECTED_RATE_PROFILES:
            raise ValueError(f"'{run.name}' usa un k no seleccionado")
        rate_profile = (config.learning_rate_start, config.learning_rate_end)
        if rate_profile not in EXPECTED_RATE_PROFILES[config.grid_size]:
            raise ValueError(f"'{run.name}' usa un perfil de tasa no seleccionado")
        if config.epochs != 150 or config.initialization != "sample":
            raise ValueError(f"'{run.name}' cambia parámetros controlados")
        if config.radius_end != 1 or not config.shuffle:
            raise ValueError(f"'{run.name}' cambia parámetros controlados")
        radius_fraction = _radius_fraction(run)
        if not any(
            np.isclose(radius_fraction, expected)
            for expected in EXPECTED_RADIUS_FRACTIONS
        ):
            raise ValueError(f"'{run.name}' usa un radio no seleccionado")
        combination = (
            config.grid_size,
            *rate_profile,
            round(radius_fraction, 8),
            config.seed,
        )
        if combination in combinations:
            raise ValueError(f"combinación repetida en '{run.name}'")
        combinations.add(combination)
    return runs


def evaluate_runs(
    dataset: CreatorDataset, runs: Sequence[KohonenRun]
) -> list[RadiusRunMetrics]:
    """Entrenar cada corrida y medir calidad y tiempo de entrenamiento."""

    metrics = []
    sample_count = dataset.values.shape[0]
    for run in runs:
        started = perf_counter()
        result = run_kohonen(dataset, run)
        training_seconds = perf_counter() - started
        total_units = result.hit_map.size
        metrics.append(
            RadiusRunMetrics(
                run=run.name,
                grid_size=run.config.grid_size,
                schedule=_schedule(run),
                learning_rate_start=run.config.learning_rate_start,
                learning_rate_end=run.config.learning_rate_end,
                radius_fraction=_radius_fraction(run),
                radius_start=run.config.radius_start,
                radius_end=run.config.radius_end,
                seed=run.config.seed,
                training_seconds=training_seconds,
                quantization_error=result.quantization_error,
                occupied_units=result.occupied_units,
                dead_units=result.dead_units,
                dead_units_pct=100 * result.dead_units / total_units,
                maximum_hits=int(result.hit_map.max()),
                maximum_hits_pct=100 * int(result.hit_map.max()) / sample_count,
                mean_u_distance=float(result.u_matrix.mean()),
            )
        )
    return metrics


def aggregate_metrics(metrics: Sequence[RadiusRunMetrics]) -> list[dict]:
    """Calcular media y desvío por k, perfil de tasa y radio."""

    keys = sorted(
        {
            (
                metric.grid_size,
                metric.learning_rate_start,
                metric.learning_rate_end,
                metric.radius_fraction,
            )
            for metric in metrics
        }
    )
    rows = []
    for grid_size, rate_start, rate_end, radius_fraction in keys:
        group = [
            metric
            for metric in metrics
            if metric.grid_size == grid_size
            and metric.learning_rate_start == rate_start
            and metric.learning_rate_end == rate_end
            and np.isclose(metric.radius_fraction, radius_fraction)
        ]
        if len(group) < 2:
            raise ValueError("cada combinación necesita al menos dos semillas")
        row = {
            "grid_size": grid_size,
            "schedule": group[0].schedule,
            "learning_rate_start": rate_start,
            "learning_rate_end": rate_end,
            "radius_fraction": radius_fraction,
            "radius_start": group[0].radius_start,
            "radius_end": group[0].radius_end,
            "runs": len(group),
        }
        for field in (
            "training_seconds",
            "quantization_error",
            "occupied_units",
            "dead_units",
            "dead_units_pct",
            "maximum_hits",
            "maximum_hits_pct",
            "mean_u_distance",
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
    metrics: Sequence[RadiusRunMetrics],
    aggregate: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "corridas.csv",
        list(RadiusRunMetrics.__dataclass_fields__),
        (vars(metric) for metric in metrics),
    )
    _write_csv(
        generated / "resumen_radio.csv",
        list(aggregate[0]),
        aggregate,
    )


def _profile_label(row: dict) -> str:
    start = row["learning_rate_start"]
    end = row["learning_rate_end"]
    return f"η={start:g}" if start == end else f"η={start:g}→{end:g}"


def _save_both(figure, path: Path) -> None:
    _save_figure(figure, path.with_suffix(".svg"))
    figure.savefig(
        path.with_suffix(".png"),
        format="png",
        dpi=160,
        bbox_inches="tight",
    )


def write_quality_curves(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    grid_sizes = sorted({row["grid_size"] for row in aggregate})
    definitions = (
        ("quantization_error", "Error de cuantización", "Distancia a la BMU"),
        ("dead_units_pct", "Unidades muertas", "Porcentaje de la grilla"),
        (
            "maximum_hits_pct",
            "Concentración máxima",
            "Muestras en la neurona más ocupada (%)",
        ),
    )
    figure, axes = plt.subplots(3, 3, figsize=(16, 13), squeeze=False)
    for row_index, grid_size in enumerate(grid_sizes):
        grid_rows = [row for row in aggregate if row["grid_size"] == grid_size]
        profiles = sorted(
            {
                (row["learning_rate_start"], row["learning_rate_end"])
                for row in grid_rows
            }
        )
        for column_index, (field, title, ylabel) in enumerate(definitions):
            axis = axes[row_index, column_index]
            for rate_start, rate_end in profiles:
                rows = [
                    row
                    for row in grid_rows
                    if row["learning_rate_start"] == rate_start
                    and row["learning_rate_end"] == rate_end
                ]
                axis.errorbar(
                    [100 * row["radius_fraction"] for row in rows],
                    [row[f"{field}_mean"] for row in rows],
                    yerr=[row[f"{field}_std"] for row in rows],
                    marker="o",
                    capsize=3,
                    label=_profile_label(rows[0]),
                )
            axis.set(
                title=f"{title} — k={grid_size}",
                xlabel="Radio inicial: recorrido entre 1 y diámetro (%)",
                ylabel=ylabel,
                xticks=(0, 25, 50, 75, 100),
            )
            axis.grid(alpha=0.25)
            if column_index == 2:
                axis.legend(fontsize=8)
    figure.suptitle("Efecto del radio inicial — media ± desvío de cinco semillas")
    figure.tight_layout()
    _save_both(figure, output / "generated" / "curvas_radio")
    plt.close(figure)


def write_runtime_curves(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    grid_sizes = sorted({row["grid_size"] for row in aggregate})
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for axis, grid_size in zip(axes, grid_sizes):
        grid_rows = [row for row in aggregate if row["grid_size"] == grid_size]
        profiles = sorted(
            {
                (row["learning_rate_start"], row["learning_rate_end"])
                for row in grid_rows
            }
        )
        for rate_start, rate_end in profiles:
            rows = [
                row
                for row in grid_rows
                if row["learning_rate_start"] == rate_start
                and row["learning_rate_end"] == rate_end
            ]
            axis.errorbar(
                [100 * row["radius_fraction"] for row in rows],
                [1000 * row["training_seconds_mean"] for row in rows],
                yerr=[1000 * row["training_seconds_std"] for row in rows],
                marker="o",
                capsize=3,
                label=_profile_label(rows[0]),
            )
        axis.set(
            title=f"k={grid_size}",
            xlabel="Radio inicial: recorrido entre 1 y diámetro (%)",
            ylabel="Tiempo de entrenamiento (ms)",
            xticks=(0, 25, 50, 75, 100),
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Costo computacional observado — media ± desvío")
    figure.tight_layout()
    _save_both(figure, output / "generated" / "tiempos_radio")
    plt.close(figure)


def _minimum_error_rows(aggregate: Sequence[dict]) -> list[dict]:
    profiles = sorted(
        {
            (
                row["grid_size"],
                row["learning_rate_start"],
                row["learning_rate_end"],
            )
            for row in aggregate
        }
    )
    return [
        min(
            (
                row
                for row in aggregate
                if row["grid_size"] == grid_size
                and row["learning_rate_start"] == rate_start
                and row["learning_rate_end"] == rate_end
            ),
            key=lambda row: row["quantization_error_mean"],
        )
        for grid_size, rate_start, rate_end in profiles
    ]


def write_report(output: Path, aggregate: Sequence[dict]) -> None:
    table_rows = "\n".join(
        "| {grid_size} | {schedule} | {learning_rate_start:g} → "
        "{learning_rate_end:g} | {radius_fraction:.0%} | "
        "{radius_start:.3f} → {radius_end:g} | "
        "{quantization_error_mean:.4f} | {dead_units_pct_mean:.1f} | "
        "{maximum_hits_pct_mean:.1f} | {training_seconds_mean:.4f} |".format(
            **row
        )
        for row in _minimum_error_rows(aggregate)
    )
    minimum_runtime = min(row["training_seconds_mean"] for row in aggregate)
    maximum_runtime = max(row["training_seconds_mean"] for row in aggregate)
    report = f"""# Experimento 4: radio inicial

Se compararon `k = 4`, `k = 5` y `k = 6`, cuatro perfiles de tasa por tamaño,
cinco radios iniciales y cinco semillas: 300 corridas. Los radios recorren 0 %,
25 %, 50 %, 75 % y 100 % del trayecto entre el vecindario local `R = 1` y el
diámetro de cada grilla. Todos terminan en `R = 1` mediante descenso lineal;
el primer nivel permanece constante porque comienza y termina en 1.

![Curvas de calidad por radio](generated/curvas_radio.png)

## Mínimo error dentro de cada perfil de tasa

| k | Schedule | Tasa | Nivel de radio | Radio | Error | Muertas (%) | Máximo de hits (%) | Tiempo (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table_rows}

La tabla no selecciona automáticamente esos mínimos: el radio organiza la
topología del mapa y debe juzgarse también mediante matrices U y mapas de hits.
Esas figuras se generarán sólo para las configuraciones representativas, no
para las 300 corridas.

## Lectura de los resultados

- En `k = 4` las diferencias de error son pequeñas. Dos compromisos especialmente
  estables son la tasa constante `η = 0.1` con `R = 1` (1,25 % de unidades
  muertas) y el descenso `0.3 → 0.03` con radio inicial a mitad del recorrido
  (también 1,25 %). Se conservan ambos para la inspección visual.
- En `k = 5` el radio interactúa con la tasa. Con `η = 0.1`, comenzar a mitad
  del recorrido mantiene prácticamente el mínimo error y reduce las unidades
  muertas de 17,6 % a 10,4 %. Con `η = 0.2`, comenzar en el diámetro completo
  las reduce a 4,8 %. Las alternativas decrecientes también dejan candidatos
  y deben decidirse mirando la estructura del mapa.
- En `k = 6`, los radios locales producen errores bajos a costa de entre 28,9 %
  y 37,8 % de unidades muertas. Los radios amplios bajan esa fracción hasta
  aproximadamente 15,6–22,2 %, aunque el tamaño sigue peor ocupado que `k = 4`
  y `k = 5`. Se conserva sólo como caso de frontera para la comparación visual.

Por lo tanto, el barrido no justifica elegir el mínimo error aislado. Primero
se filtran soluciones con ocupación razonable y estabilidad entre semillas;
luego se comparan sus mapas de hits y matrices U.

## Performance computacional

La consigna usa “performance” principalmente para pedir una justificación de
la calidad del modelo. Además se registró el tiempo de entrenamiento de cada
corrida como dato auxiliar. Es una medición dependiente de la máquina y no se
usa como criterio principal; permite comparar el costo cuando dos alternativas
producen resultados equivalentes.

Los tiempos medios por configuración estuvieron entre
`{1000 * minimum_runtime:.1f}` y `{1000 * maximum_runtime:.1f}` ms. La diferencia
es demasiado pequeña para compensar una peor ocupación o una estructura menos
interpretable. Sí se tendrá en cuenta si dos configuraciones resultan
equivalentes en calidad. Para esta implementación, el trabajo por época crece
principalmente con la cantidad de muestras, variables y neuronas, es decir,
aproximadamente con `N × d × k²`; el radio no cambia ese orden de complejidad.

![Tiempos por radio](generated/tiempos_radio.png)

Los resultados individuales están en `generated/corridas.csv` y los promedios
por configuración en `generated/resumen_radio.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_creator_dataset(args.data)
    runs = load_radius_runs(args.configs)
    metrics = evaluate_runs(dataset, runs)
    aggregate = aggregate_metrics(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(args.output, metrics, aggregate)
    write_quality_curves(args.output, aggregate)
    write_runtime_curves(args.output, aggregate)
    write_report(args.output, aggregate)
    print(f"Informe del radio generado en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
