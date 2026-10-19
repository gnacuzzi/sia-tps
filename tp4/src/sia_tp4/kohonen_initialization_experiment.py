"""Comparar inicializaciones sobre las configuraciones finalistas."""

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
from .kohonen_epoch_experiment import EPOCH_PROFILES, SELECTED_EPOCHS
from .pca_analysis import _load_pyplot, _save_figure


INITIALIZATIONS = ("sample", "uniform")


@dataclass(frozen=True)
class InitializationRunMetrics:
    """Indicadores de una corrida del experimento de inicialización."""

    run: str
    profile: str
    initialization: str
    grid_size: int
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
        default=Path("configs/kohonen/06-initialization"),
        help="Directorio con un JSON por corrida.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/06-initialization"),
        help="Directorio del informe comparativo.",
    )
    return parser.parse_args(argv)


def _radius_fraction(run: KohonenRun) -> float:
    diameter = np.sqrt(2) * (run.config.grid_size - 1)
    return float((run.config.radius_start - 1) / (diameter - 1))


def _profile_for_run(run: KohonenRun) -> str:
    config = run.config
    fraction = _radius_fraction(run)
    matches = [
        name
        for name, grid, rate_start, rate_end, radius_fraction in EPOCH_PROFILES
        if config.grid_size == grid
        and np.isclose(config.learning_rate_start, rate_start)
        and np.isclose(config.learning_rate_end, rate_end)
        and np.isclose(fraction, radius_fraction)
    ]
    if len(matches) != 1:
        raise ValueError(f"'{run.name}' no pertenece a una finalista")
    return matches[0]


def load_initialization_runs(directory: Path | str) -> list[KohonenRun]:
    """Cargar y validar las 50 corridas de inicialización."""

    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise ValueError(f"no hay configuraciones JSON en {directory}")
    runs = [load_run(path) for path in paths]
    combinations = set()
    for run in runs:
        config = run.config
        profile = _profile_for_run(run)
        if (
            config.epochs != SELECTED_EPOCHS
            or config.initialization not in INITIALIZATIONS
            or config.radius_end != 1
            or not config.shuffle
            or config.seed not in range(5)
        ):
            raise ValueError(f"'{run.name}' cambia parámetros controlados")
        combination = (profile, config.initialization, config.seed)
        if combination in combinations:
            raise ValueError(f"combinación repetida en '{run.name}'")
        combinations.add(combination)
    expected = {
        (profile[0], initialization, seed)
        for profile in EPOCH_PROFILES
        for initialization in INITIALIZATIONS
        for seed in range(5)
    }
    if combinations != expected:
        raise ValueError("el diseño no cubre los perfiles, inicializaciones y semillas")
    return runs


def evaluate_runs(
    dataset: CreatorDataset, runs: Sequence[KohonenRun]
) -> list[InitializationRunMetrics]:
    """Entrenar cada corrida y registrar sus métricas finales."""

    metrics = []
    sample_count = dataset.values.shape[0]
    for run in runs:
        started = perf_counter()
        result = run_kohonen(dataset, run)
        training_seconds = perf_counter() - started
        total_units = result.hit_map.size
        metrics.append(
            InitializationRunMetrics(
                run=run.name,
                profile=_profile_for_run(run),
                initialization=run.config.initialization,
                grid_size=run.config.grid_size,
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


def aggregate_metrics(metrics: Sequence[InitializationRunMetrics]) -> list[dict]:
    """Promediar las cinco semillas por perfil e inicialización."""

    rows = []
    for profile, initialization in sorted(
        {(metric.profile, metric.initialization) for metric in metrics}
    ):
        group = [
            metric
            for metric in metrics
            if metric.profile == profile
            and metric.initialization == initialization
        ]
        if len(group) != 5:
            raise ValueError("cada grupo debe contener cinco semillas")
        row = {
            "profile": profile,
            "initialization": initialization,
            "grid_size": group[0].grid_size,
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


def initialization_differences(
    metrics: Sequence[InitializationRunMetrics],
) -> list[dict]:
    """Calcular diferencias pareadas uniform menos sample por semilla."""

    rows = []
    fields = (
        "training_seconds",
        "quantization_error",
        "dead_units_pct",
        "maximum_hits_pct",
        "mean_u_distance",
    )
    for profile, *_ in EPOCH_PROFILES:
        sample = {
            metric.seed: metric
            for metric in metrics
            if metric.profile == profile and metric.initialization == "sample"
        }
        uniform = {
            metric.seed: metric
            for metric in metrics
            if metric.profile == profile and metric.initialization == "uniform"
        }
        if sample.keys() != uniform.keys() or len(sample) != 5:
            raise ValueError(f"'{profile}' no tiene semillas pareadas")
        row = {"profile": profile, "runs": len(sample)}
        for field in fields:
            differences = np.asarray(
                [
                    getattr(uniform[seed], field) - getattr(sample[seed], field)
                    for seed in sorted(sample)
                ]
            )
            row[f"{field}_delta_mean"] = float(np.mean(differences))
            row[f"{field}_delta_std"] = float(np.std(differences, ddof=1))
        rows.append(row)
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    metrics: Sequence[InitializationRunMetrics],
    aggregate: Sequence[dict],
    differences: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "corridas.csv",
        list(InitializationRunMetrics.__dataclass_fields__),
        (vars(metric) for metric in metrics),
    )
    _write_csv(
        generated / "resumen_inicializacion.csv",
        list(aggregate[0]),
        aggregate,
    )
    _write_csv(
        generated / "diferencias_uniform_menos_sample.csv",
        list(differences[0]),
        differences,
    )


def _save_both(figure, path: Path) -> None:
    _save_figure(figure, path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")


def write_plot(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    profiles = [profile[0] for profile in EPOCH_PROFILES]
    positions = np.arange(len(profiles))
    definitions = (
        ("quantization_error", "Error de cuantización", "Distancia a la BMU"),
        ("dead_units_pct", "Unidades muertas", "Porcentaje de la grilla"),
        (
            "maximum_hits_pct",
            "Concentración máxima",
            "Muestras en la neurona más ocupada (%)",
        ),
        ("training_seconds", "Tiempo", "Segundos por entrenamiento"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    for axis, (field, title, ylabel) in zip(axes.flat, definitions):
        for offset, initialization in zip((-0.08, 0.08), INITIALIZATIONS):
            rows = [
                next(
                    row
                    for row in aggregate
                    if row["profile"] == profile
                    and row["initialization"] == initialization
                )
                for profile in profiles
            ]
            axis.errorbar(
                positions + offset,
                [row[f"{field}_mean"] for row in rows],
                yerr=[row[f"{field}_std"] for row in rows],
                marker="o",
                linestyle="none",
                capsize=4,
                label=initialization,
            )
        axis.set(title=title, ylabel=ylabel, xticks=positions)
        axis.set_xticklabels(profiles, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Inicialización sample vs. uniform — media ± desvío")
    figure.tight_layout()
    _save_both(figure, output / "generated" / "comparacion_inicializacion")
    plt.close(figure)


def write_report(
    output: Path,
    aggregate: Sequence[dict],
    differences: Sequence[dict],
) -> None:
    table_rows = "\n".join(
        "| {profile} | {initialization} | {quantization_error_mean:.4f} ± "
        "{quantization_error_std:.4f} | {dead_units_pct_mean:.1f} ± "
        "{dead_units_pct_std:.1f} | {maximum_hits_pct_mean:.1f} ± "
        "{maximum_hits_pct_std:.1f} | {training_seconds_mean:.3f} |".format(**row)
        for row in aggregate
    )
    difference_rows = "\n".join(
        "| {profile} | {quantization_error_delta_mean:+.4f} | "
        "{dead_units_pct_delta_mean:+.1f} | "
        "{maximum_hits_pct_delta_mean:+.1f} |".format(**row)
        for row in differences
    )
    report = f"""# Experimento 6: inicialización

Se compararon inicializaciones `sample` y `uniform` en las cinco configuraciones
finalistas, manteniendo 150 épocas y cinco semillas: 50 corridas. `sample`
comienza cada neurona en un creador real; `uniform` sortea cada peso entre el
mínimo y el máximo estandarizado de su variable.

![Comparación de inicializaciones](generated/comparacion_inicializacion.png)

| Configuración | Inicialización | Error | Muertas (%) | Máximo de hits (%) | Tiempo (s) |
| --- | --- | ---: | ---: | ---: | ---: |
{table_rows}

## Diferencia pareada: uniform menos sample

Los valores negativos favorecen a `uniform` en las tres métricas de calidad.

| Configuración | Δ error | Δ muertas (pp) | Δ máximo de hits (pp) |
| --- | ---: | ---: | ---: |
{difference_rows}

Las diferencias se calcularon por semilla antes de promediar. Si `uniform`
resulta mejor de forma consistente, este experimento no basta para fijarla:
habrá que revalidar su interacción con tasa y radio en una región acotada.

## Decisión

Se conserva **`sample`**. `uniform` no mejora conjuntamente las métricas en
ninguna de las cinco configuraciones:

- en `k = 4` mantiene o aumenta las unidades muertas y sube la concentración
  máxima en 1,6 puntos porcentuales;
- en `k = 5` constante empeora error, ocupación y concentración;
- en `k = 5` decreciente reduce el error en 0,0464, pero agrega 10,4 puntos de
  unidades muertas y presenta mucha más variabilidad;
- en `k = 6` reduce el error en 0,0222 y la concentración en 0,4 puntos, pero
  agrega 5,6 puntos de unidades muertas y también aumenta la dispersión.

Las reducciones aisladas del error con `uniform` provienen de mapas que usan
menos neuronas, por lo que no justifican reabrir ahora los barridos de tasa y
radio. `sample` además comienza en regiones donde existen datos reales, una
ventaja pertinente para sólo 50 creadores.

Los resultados individuales están en `generated/corridas.csv`, los promedios
en `generated/resumen_inicializacion.csv` y las diferencias pareadas en
`generated/diferencias_uniform_menos_sample.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_creator_dataset(args.data)
    runs = load_initialization_runs(args.configs)
    metrics = evaluate_runs(dataset, runs)
    aggregate = aggregate_metrics(metrics)
    differences = initialization_differences(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(args.output, metrics, aggregate, differences)
    write_plot(args.output, aggregate)
    write_report(args.output, aggregate, differences)
    print(f"Informe de inicialización generado en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
