"""Comparar cantidades de épocas sobre las configuraciones finalistas."""

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


EXPECTED_EPOCHS = (25, 50, 100, 150, 250, 400)
SELECTED_EPOCHS = 150
EPOCH_PROFILES = (
    ("k4-constant-local", 4, 0.1, 0.1, 0.0),
    ("k4-decay-middle", 4, 0.3, 0.03, 0.5),
    ("k5-constant-global", 5, 0.2, 0.2, 1.0),
    ("k5-decay-wide", 5, 0.7, 0.07, 0.75),
    ("k6-decay-global", 6, 0.7, 0.07, 1.0),
)


@dataclass(frozen=True)
class EpochRunMetrics:
    """Indicadores finales y costo de una corrida del barrido de épocas."""

    run: str
    profile: str
    grid_size: int
    epochs: int
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
        default=Path("configs/kohonen/05-epochs"),
        help="Directorio con un JSON por corrida.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-experimentos/05-epochs"),
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


def load_epoch_runs(directory: Path | str) -> list[KohonenRun]:
    """Cargar y validar las 150 corridas del diseño de épocas."""

    paths = sorted(Path(directory).glob("*.json"))
    if not paths:
        raise ValueError(f"no hay configuraciones JSON en {directory}")
    runs = [load_run(path) for path in paths]
    combinations = set()
    for run in runs:
        config = run.config
        profile = _profile_for_run(run)
        if config.epochs not in EXPECTED_EPOCHS:
            raise ValueError(f"'{run.name}' usa una cantidad de épocas no prevista")
        if (
            config.radius_end != 1
            or config.initialization != "sample"
            or not config.shuffle
            or config.seed not in range(5)
        ):
            raise ValueError(f"'{run.name}' cambia parámetros controlados")
        combination = (profile, config.epochs, config.seed)
        if combination in combinations:
            raise ValueError(f"combinación repetida en '{run.name}'")
        combinations.add(combination)
    expected = {
        (profile[0], epochs, seed)
        for profile in EPOCH_PROFILES
        for epochs in EXPECTED_EPOCHS
        for seed in range(5)
    }
    if combinations != expected:
        raise ValueError("el diseño no cubre todas las finalistas, épocas y semillas")
    return runs


def evaluate_runs(
    dataset: CreatorDataset, runs: Sequence[KohonenRun]
) -> list[EpochRunMetrics]:
    """Entrenar cada corrida y registrar calidad final y tiempo."""

    metrics = []
    sample_count = dataset.values.shape[0]
    for run in runs:
        started = perf_counter()
        result = run_kohonen(dataset, run)
        training_seconds = perf_counter() - started
        total_units = result.hit_map.size
        metrics.append(
            EpochRunMetrics(
                run=run.name,
                profile=_profile_for_run(run),
                grid_size=run.config.grid_size,
                epochs=run.config.epochs,
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


def aggregate_metrics(metrics: Sequence[EpochRunMetrics]) -> list[dict]:
    """Calcular media y desvío de las cinco semillas por punto."""

    rows = []
    for profile, epochs in sorted(
        {(metric.profile, metric.epochs) for metric in metrics}
    ):
        group = [
            metric
            for metric in metrics
            if metric.profile == profile and metric.epochs == epochs
        ]
        if len(group) != 5:
            raise ValueError("cada punto del barrido debe tener cinco semillas")
        first = group[0]
        row = {
            "profile": profile,
            "grid_size": first.grid_size,
            "epochs": epochs,
            "learning_rate_start": first.learning_rate_start,
            "learning_rate_end": first.learning_rate_end,
            "radius_fraction": first.radius_fraction,
            "radius_start": first.radius_start,
            "radius_end": first.radius_end,
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
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    metrics: Sequence[EpochRunMetrics],
    aggregate: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "corridas.csv",
        list(EpochRunMetrics.__dataclass_fields__),
        (vars(metric) for metric in metrics),
    )
    _write_csv(generated / "resumen_epocas.csv", list(aggregate[0]), aggregate)


def _save_both(figure, path: Path) -> None:
    _save_figure(figure, path.with_suffix(".svg"))
    figure.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")


def write_curves(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
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
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for axis, (field, title, ylabel) in zip(axes.flat, definitions):
        for profile, *_ in EPOCH_PROFILES:
            rows = [row for row in aggregate if row["profile"] == profile]
            axis.errorbar(
                [row["epochs"] for row in rows],
                [row[f"{field}_mean"] for row in rows],
                yerr=[row[f"{field}_std"] for row in rows],
                marker="o",
                capsize=3,
                label=profile,
            )
        axis.set(
            title=title,
            xlabel="Épocas",
            ylabel=ylabel,
            xticks=EXPECTED_EPOCHS,
        )
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Barrido de épocas — media ± desvío de cinco semillas")
    figure.tight_layout()
    _save_both(figure, output / "generated" / "curvas_epocas")
    plt.close(figure)


def write_report(output: Path, aggregate: Sequence[dict]) -> None:
    comparison = [row for row in aggregate if row["epochs"] in (150, 250, 400)]
    table_rows = "\n".join(
        "| {profile} | {epochs} | {quantization_error_mean:.4f} ± "
        "{quantization_error_std:.4f} | {dead_units_pct_mean:.1f} ± "
        "{dead_units_pct_std:.1f} | {maximum_hits_pct_mean:.1f} ± "
        "{maximum_hits_pct_std:.1f} | {training_seconds_mean:.3f} |".format(**row)
        for row in comparison
    )
    report = f"""# Experimento 5: cantidad de épocas

Se compararon cinco configuraciones finalistas con `25`, `50`, `100`, `150`,
`250` y `400` épocas y cinco semillas: 150 corridas independientes. En cada
corrida, la tasa y el radio recorren su schedule completo durante la cantidad
de épocas indicada; los puntos no son checkpoints de un mismo entrenamiento.

![Curvas por cantidad de épocas](generated/curvas_epocas.png)

## Región de 150 a 400 épocas

| Configuración | Épocas | Error | Muertas (%) | Máximo de hits (%) | Tiempo (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
{table_rows}

La cantidad de épocas no se elige por el menor error aislado. Se busca una
meseta razonable en calidad y estabilidad, evitando pagar más tiempo por una
mejora pequeña o por una ocupación peor.

## Decisión

Se conservan **150 épocas**. Es el primer presupuesto donde las cinco
configuraciones muestran simultáneamente una ocupación razonable: las dos
variantes `k = 4` promedian 1,2 % de unidades muertas; las dos `k = 5`, 4,8 %;
y el control `k = 6`, 15,6 %. También mantienen concentraciones máximas entre
7,2 % y 10,8 %.

Entrenar hasta 400 épocas cuesta aproximadamente 2,6–2,7 veces más. En `k = 4`
el error ya está en una meseta y la ocupación no mejora consistentemente. En
`k = 5` aparecen pequeñas mejoras aisladas, pero con mayor variabilidad o sin
una reducción consistente del error. En `k = 6`, el error baja de 2,0725 a
1,9931, pero las unidades muertas suben de 15,6 % a 26,1 %: representa mejor a
las neuronas usadas mientras desaprovecha una parte mayor de la grilla.

Por lo tanto, extender el entrenamiento no aporta una mejora conjunta que
justifique cambiar el valor usado hasta ahora. El siguiente experimento puede
comparar inicializaciones manteniendo 150 épocas y los demás hiperparámetros
seleccionados.

Los resultados individuales están en `generated/corridas.csv` y los promedios
en `generated/resumen_epocas.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_creator_dataset(args.data)
    runs = load_epoch_runs(args.configs)
    metrics = evaluate_runs(dataset, runs)
    aggregate = aggregate_metrics(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(args.output, metrics, aggregate)
    write_curves(args.output, aggregate)
    write_report(args.output, aggregate)
    print(f"Informe de épocas generado en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
