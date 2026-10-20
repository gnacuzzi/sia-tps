"""Comparar tasas constantes y decrecientes para la regla de Oja."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from .data import CreatorDataset, load_creator_dataset
from .oja import OjaConfig, align_sign, direction_similarity, fit_oja
from .pca import PCAResult, fit_pca
from .pca_analysis import _load_pyplot, _save_figure


SCHEDULE_END_RATIOS = {"constant": 1.0, "linear-10pct": 0.1}


@dataclass(frozen=True)
class OjaRateDesign:
    """Diseño compacto de la primera etapa experimental."""

    name: str
    epochs: int
    learning_rates: Tuple[float, ...]
    schedules: Tuple[str, ...]
    seeds: Tuple[int, ...]
    shuffle: bool


@dataclass(frozen=True)
class OjaRunMetrics:
    """Resultado final de una tasa, schedule y semilla."""

    run: str
    learning_rate: float
    schedule: str
    seed: int
    success: bool
    direction_similarity: float
    score_correlation: float
    weight_norm: float
    norm_error: float
    failure_reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datos/creadores.csv"),
        help="CSV fuente (por defecto, datos/creadores.csv).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/oja/01-rate.json"),
        help="Diseño JSON del experimento.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/oja-experimentos/01-rate"),
        help="Directorio del informe comparativo.",
    )
    return parser.parse_args(argv)


def load_design(path: Path | str) -> OjaRateDesign:
    """Leer el diseño y rechazar campos desconocidos o combinaciones ambiguas."""

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {"name", "epochs", "learning_rates", "schedules", "seeds", "shuffle"}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"campos faltantes: {missing}; campos desconocidos: {unknown}")

    rates = tuple(float(rate) for rate in raw["learning_rates"])
    schedules = tuple(raw["schedules"])
    seeds = tuple(raw["seeds"])
    if not raw["name"] or not isinstance(raw["name"], str):
        raise ValueError("name debe ser un texto no vacío")
    if not isinstance(raw["epochs"], int) or isinstance(raw["epochs"], bool):
        raise ValueError("epochs debe ser un entero")
    if raw["epochs"] < 1:
        raise ValueError("epochs debe ser positivo")
    if not rates or len(rates) != len(set(rates)):
        raise ValueError("learning_rates debe contener tasas únicas")
    if any(not 0 < rate <= 0.5 for rate in rates):
        raise ValueError("las tasas deben pertenecer a (0, 0.5]")
    if not schedules or len(schedules) != len(set(schedules)):
        raise ValueError("schedules debe contener valores únicos")
    invalid_schedules = sorted(set(schedules) - set(SCHEDULE_END_RATIOS))
    if invalid_schedules:
        raise ValueError(f"schedules desconocidos: {invalid_schedules}")
    if len(seeds) < 10 or len(seeds) != len(set(seeds)):
        raise ValueError("se requieren al menos 10 semillas únicas")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("las semillas deben ser enteros")
    if not isinstance(raw["shuffle"], bool):
        raise ValueError("shuffle debe ser booleano")

    return OjaRateDesign(
        name=raw["name"],
        epochs=raw["epochs"],
        learning_rates=tuple(sorted(rates)),
        schedules=schedules,
        seeds=seeds,
        shuffle=raw["shuffle"],
    )


def run_count(design: OjaRateDesign) -> int:
    return len(design.learning_rates) * len(design.schedules) * len(design.seeds)


def _history_metrics(
    weights_history: np.ndarray,
    standardized: np.ndarray,
    pca: PCAResult,
) -> list[dict]:
    rows = []
    reference = pca.components[:, 0]
    reference_scores = pca.scores[:, 0]
    for epoch, weights in enumerate(weights_history):
        aligned = align_sign(weights, reference)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = standardized @ aligned
        correlation = float(np.corrcoef(scores, reference_scores)[0, 1])
        norm = float(np.linalg.norm(weights))
        rows.append(
            {
                "epoch": epoch,
                "direction_similarity": direction_similarity(weights, reference),
                "score_correlation": correlation,
                "weight_norm": norm,
                "norm_error": abs(norm - 1.0),
            }
        )
    return rows


def evaluate_design(
    dataset: CreatorDataset,
    design: OjaRateDesign,
) -> tuple[list[OjaRunMetrics], list[dict], PCAResult]:
    """Ejecutar todas las combinaciones y conservar métricas, no modelos."""

    pca = fit_pca(dataset.values, dataset.feature_names)
    standardized = pca.standardized_values
    metrics = []
    histories = []
    for schedule in design.schedules:
        ratio = SCHEDULE_END_RATIOS[schedule]
        for rate in design.learning_rates:
            for seed in design.seeds:
                name = f"eta-{rate:g}-{schedule}-seed-{seed:02d}"
                config = OjaConfig(
                    epochs=design.epochs,
                    learning_rate_start=rate,
                    learning_rate_end=rate * ratio,
                    shuffle=design.shuffle,
                    seed=seed,
                )
                try:
                    model = fit_oja(dataset.values, dataset.feature_names, config)
                    history = _history_metrics(
                        model.weight_history, standardized, pca
                    )
                except (FloatingPointError, OverflowError) as error:
                    metrics.append(
                        OjaRunMetrics(
                            name,
                            rate,
                            schedule,
                            seed,
                            False,
                            float("nan"),
                            float("nan"),
                            float("nan"),
                            float("nan"),
                            str(error),
                        )
                    )
                    continue

                for row in history:
                    histories.append(
                        {
                            "run": name,
                            "learning_rate": rate,
                            "schedule": schedule,
                            "seed": seed,
                            **row,
                        }
                    )
                final = history[-1]
                metrics.append(
                    OjaRunMetrics(
                        name,
                        rate,
                        schedule,
                        seed,
                        True,
                        final["direction_similarity"],
                        final["score_correlation"],
                        final["weight_norm"],
                        final["norm_error"],
                        "",
                    )
                )
    return metrics, histories, pca


def _summary(values: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def aggregate_runs(metrics: Sequence[OjaRunMetrics]) -> list[dict]:
    """Resumir las 10 semillas y mantener visibles las divergencias."""

    rows = []
    combinations = sorted(
        {(metric.schedule, metric.learning_rate) for metric in metrics}
    )
    for schedule, rate in combinations:
        group = [
            metric
            for metric in metrics
            if metric.schedule == schedule and metric.learning_rate == rate
        ]
        successful = [metric for metric in group if metric.success]
        row = {
            "schedule": schedule,
            "learning_rate": rate,
            "runs": len(group),
            "successful_runs": len(successful),
            "divergent_runs": len(group) - len(successful),
        }
        for field in (
            "direction_similarity",
            "score_correlation",
            "weight_norm",
            "norm_error",
        ):
            if successful:
                values = np.asarray([getattr(metric, field) for metric in successful])
                row.update(_summary(values, field))
            else:
                row.update(
                    {
                        f"{field}_{suffix}": float("nan")
                        for suffix in ("mean", "std", "median", "min", "max")
                    }
                )
        rows.append(row)
    return rows


def aggregate_history(histories: Sequence[dict]) -> list[dict]:
    """Agregar las trayectorias por configuración y época."""

    rows = []
    keys = sorted(
        {
            (row["schedule"], row["learning_rate"], row["epoch"])
            for row in histories
        }
    )
    for schedule, rate, epoch in keys:
        group = [
            row
            for row in histories
            if row["schedule"] == schedule
            and row["learning_rate"] == rate
            and row["epoch"] == epoch
        ]
        aggregate = {
            "schedule": schedule,
            "learning_rate": rate,
            "epoch": epoch,
            "runs": len(group),
        }
        for field in (
            "direction_similarity",
            "score_correlation",
            "weight_norm",
            "norm_error",
        ):
            values = np.asarray([row[field] for row in group])
            aggregate[f"{field}_mean"] = float(np.mean(values))
            aggregate[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
            aggregate[f"{field}_min"] = float(np.min(values))
            aggregate[f"{field}_max"] = float(np.max(values))
        rows.append(aggregate)
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    metrics: Sequence[OjaRunMetrics],
    aggregate: Sequence[dict],
    history: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "corridas.csv",
        list(OjaRunMetrics.__dataclass_fields__),
        (vars(metric) for metric in metrics),
    )
    _write_csv(generated / "resumen_tasas.csv", list(aggregate[0]), aggregate)
    _write_csv(
        generated / "evolucion_agregada.csv", list(history[0]), history
    )


def write_boxplots(output: Path, metrics: Sequence[OjaRunMetrics]) -> None:
    plt = _load_pyplot()
    schedules = tuple(SCHEDULE_END_RATIOS)
    rates = sorted({metric.learning_rate for metric in metrics})
    figure, axes = plt.subplots(1, len(schedules), figsize=(15, 5), sharey=True)
    for axis, schedule in zip(axes, schedules):
        groups = [
            [
                metric.direction_similarity
                for metric in metrics
                if metric.success
                and metric.schedule == schedule
                and metric.learning_rate == rate
            ]
            for rate in rates
        ]
        axis.boxplot(groups, tick_labels=[f"{rate:g}" for rate in rates])
        axis.set(
            title=schedule,
            xlabel="Tasa inicial η",
            ylabel="Similitud absoluta con PC1",
        )
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    figure.suptitle("Estabilidad final de Oja en 10 semillas")
    figure.tight_layout()
    _save_figure(figure, output / "generated" / "similitud_por_tasa.svg")
    plt.close(figure)


def write_metric_curves(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    definitions = (
        ("direction_similarity", "Similitud con PC1", "Mayor es mejor"),
        ("score_correlation", "Correlación de índices", "Mayor es mejor"),
        ("norm_error", "Error de norma", "|‖w‖ - 1|; menor es mejor"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, (field, title, ylabel) in zip(axes, definitions):
        for schedule in SCHEDULE_END_RATIOS:
            rows = [row for row in aggregate if row["schedule"] == schedule]
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
                label=schedule,
            )
        axis.set_xscale("log")
        axis.set(title=title, xlabel="Tasa inicial η", ylabel=ylabel)
        axis.grid(alpha=0.25)
    axes[-1].legend(title="Schedule")
    figure.suptitle("Resultado final medio y desvío de 10 semillas")
    figure.tight_layout()
    _save_figure(figure, output / "generated" / "metricas_por_tasa.svg")
    plt.close(figure)


def write_full_convergence(output: Path, history: Sequence[dict]) -> None:
    plt = _load_pyplot()
    schedules = tuple(SCHEDULE_END_RATIOS)
    rates = sorted({row["learning_rate"] for row in history})
    figure, axes = plt.subplots(1, len(schedules), figsize=(16, 5), sharey=True)
    for axis, schedule in zip(axes, schedules):
        for rate in rates:
            rows = [
                row
                for row in history
                if row["schedule"] == schedule
                and row["learning_rate"] == rate
            ]
            if not rows:
                continue
            epochs = np.asarray([row["epoch"] for row in rows])
            means = np.asarray([row["direction_similarity_mean"] for row in rows])
            deviations = np.asarray([row["direction_similarity_std"] for row in rows])
            line = axis.plot(epochs, means, linewidth=1.2, label=f"η={rate:g}")[0]
            axis.fill_between(
                epochs,
                np.clip(means - deviations, 0.0, 1.0),
                np.clip(means + deviations, 0.0, 1.0),
                color=line.get_color(),
                alpha=0.08,
            )
        axis.set(
            title=schedule,
            xlabel="Época",
            ylabel="Similitud absoluta con PC1",
        )
        axis.grid(alpha=0.25)
    axes[-1].legend(title="Tasa inicial", bbox_to_anchor=(1.03, 1), loc="upper left")
    figure.suptitle("Evolución completa de la dirección aprendida")
    figure.tight_layout()
    _save_figure(figure, output / "generated" / "convergencia_completa.svg")
    plt.close(figure)


def write_report(
    output: Path,
    design: OjaRateDesign,
    aggregate: Sequence[dict],
    history: Sequence[dict],
    pca: PCAResult,
) -> None:
    rows = []
    for row in sorted(aggregate, key=lambda item: (item["schedule"], item["learning_rate"])):
        if row["successful_runs"]:
            similarity = (
                f'{row["direction_similarity_mean"]:.6f} ± '
                f'{row["direction_similarity_std"]:.6f}'
            )
            correlation = (
                f'{row["score_correlation_mean"]:.6f} ± '
                f'{row["score_correlation_std"]:.6f}'
            )
            norm_error = (
                f'{row["norm_error_mean"]:.6f} ± {row["norm_error_std"]:.6f}'
            )
        else:
            similarity = correlation = norm_error = "sin corridas convergentes"
        rows.append(
            f'| {row["schedule"]} | {row["learning_rate"]:g} | '
            f'{row["successful_runs"]}/{row["runs"]} | {similarity} | '
            f'{correlation} | {norm_error} |'
        )

    finalists = (
        ("linear-10pct", 0.001, "mayor precisión y menor error de norma"),
        (
            "linear-10pct",
            0.005,
            "convergencia temprana con precisión final casi perfecta",
        ),
        ("constant", 0.001, "control estable para aislar el efecto del schedule"),
    )
    checkpoints = (5, 25, 50, 100, design.epochs)
    finalist_rows = []
    for schedule, rate, reason in finalists:
        values = []
        for epoch in checkpoints:
            match = next(
                row
                for row in history
                if row["schedule"] == schedule
                and row["learning_rate"] == rate
                and row["epoch"] == epoch
            )
            values.append(f'{match["direction_similarity_mean"]:.6f}')
        finalist_rows.append(
            f"| {schedule} | {rate:g} | {' | '.join(values)} | {reason} |"
        )

    report = f"""# Oja: etapa 1 - tasa de aprendizaje

## Protocolo

- Datos: las 15 variables base de `docs/decisiones.md`.
- Preprocesamiento: el mismo z-score muestral (`ddof=1`) que PCA.
- Épocas fijas: **{design.epochs}**.
- Tasas iniciales: {', '.join(f'`{rate:g}`' for rate in design.learning_rates)}.
- Schedules: tasa constante y descenso lineal hasta el 10 %.
- Semillas: {', '.join(str(seed) for seed in design.seeds)} ({len(design.seeds)} por configuración).
- Orden: barajado en cada época mediante el generador de cada semilla.
- Total: **{run_count(design)} corridas**.

PC1 se calcula sólo como referencia de evaluación. Su primer autovalor es
{pca.eigenvalues[0]:.6f} y explica {pca.explained_variance_ratio[0] * 100:.2f} %
de la varianza. No interviene en las actualizaciones de Oja.

## Resultados finales

| Schedule | η inicial | Exitosas | Similitud con PC1 | Correlación de índices | Error de norma |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

La similitud es el coseno absoluto entre el peso de Oja y el autovector de PC1,
por lo que trata como equivalentes ambos signos. La correlación se calcula luego
de alinear el signo. El error de norma es `abs(||w|| - 1)`.

![Boxplots de similitud](generated/similitud_por_tasa.svg)

Los boxplots comparan las semillas sin conectarlas como si fueran una serie
temporal.

![Métricas por tasa](generated/metricas_por_tasa.svg)

Cada punto es el promedio de las semillas y sus barras muestran un desvío
estándar. Las configuraciones divergentes permanecen contabilizadas en las
tablas y no se convierten en resultados numéricos artificiales.

## Lectura y recorte para la etapa 2

No aparece un máximo interior seguido por una caída: con 250 épocas, la menor
tasa evaluada ya alcanza el techo de la similitud y la calidad empeora al
aumentar `eta`. Esto no deja un extremo del rango sin explicar. `eta = 0.001`
ya supera 0.99999 alrededor de la época 50; probar tasas todavía menores sólo
evaluaría una convergencia más lenta hacia una métrica que no puede superar 1.

Las tasas constantes entre 0.005 y 0.05 permanecen finitas, pero terminan con
menor similitud, mayor dispersión y una norma más alejada de 1. Las constantes
desde 0.1 divergen en las 10 semillas. El descenso lineal estabiliza tasas de
hasta 0.05, pero con 0.1 sólo convergen 2 de 10 semillas, y desde 0.2 divergen
todas. Por lo tanto, el schedule ayuda a reducir la oscilación final, pero no
rescata de forma confiable una tasa inicial excesiva.

Se proponen tres finalistas para estudiar épocas:

| Schedule | η inicial | Época 5 | Época 25 | Época 50 | Época 100 | Época {design.epochs} | Motivo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(finalist_rows)}

El descenso con `eta = 0.001` representa la mejor aproximación final; el
descenso con `eta = 0.005` permite estudiar si se puede llegar antes a una
solución equivalente; y la constante `eta = 0.001` funciona como control para
medir si el descenso aporta una mejora material.

![Convergencia completa](generated/convergencia_completa.svg)

Se muestra la trayectoria completa antes de considerar cualquier ampliación de
las últimas épocas. La elección de finalistas para el barrido de épocas debe
considerar conjuntamente nivel final, velocidad, dispersión y divergencias; no
se cierra automáticamente en este informe.

## Archivos generados

- `generated/corridas.csv`: resultado final de cada corrida;
- `generated/resumen_tasas.csv`: agregados por tasa y schedule;
- `generated/evolucion_agregada.csv`: media, desvío y extremos por época;
- tres SVG comparativos, sin imágenes redundantes por semilla.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def run_experiment(data: Path, config: Path, output: Path) -> list[dict]:
    dataset = load_creator_dataset(data)
    design = load_design(config)
    metrics, histories, pca = evaluate_design(dataset, design)
    aggregate = aggregate_runs(metrics)
    history = aggregate_history(histories)
    write_tables(output, metrics, aggregate, history)
    write_boxplots(output, metrics)
    write_metric_curves(output, aggregate)
    write_full_convergence(output, history)
    write_report(output, design, aggregate, history, pca)
    return aggregate


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    aggregate = run_experiment(args.data, args.config, args.output)
    divergent = sum(row["divergent_runs"] for row in aggregate)
    total = sum(row["runs"] for row in aggregate)
    print(
        f"Etapa 1 de Oja completa: {total} corridas, {divergent} divergentes; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
