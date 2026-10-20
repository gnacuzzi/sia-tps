"""Comparar cantidades de épocas para los perfiles finalistas de Oja."""

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


@dataclass(frozen=True)
class OjaProfile:
    """Tasa y schedule que avanzaron desde la primera etapa."""

    name: str
    learning_rate_start: float
    learning_rate_end_ratio: float


@dataclass(frozen=True)
class OjaEpochDesign:
    """Diseño compacto del barrido de épocas."""

    name: str
    epochs: Tuple[int, ...]
    profiles: Tuple[OjaProfile, ...]
    seeds: Tuple[int, ...]
    shuffle: bool


@dataclass(frozen=True)
class OjaEpochMetrics:
    """Resultado final de una duración, perfil y semilla."""

    run: str
    profile: str
    epochs: int
    learning_rate_start: float
    learning_rate_end: float
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
        default=Path("configs/oja/02-epochs.json"),
        help="Diseño JSON del experimento.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/oja-experimentos/02-epochs"),
        help="Directorio del informe comparativo.",
    )
    return parser.parse_args(argv)


def _load_profile(raw: object) -> OjaProfile:
    if not isinstance(raw, dict):
        raise ValueError("cada perfil debe ser un objeto JSON")
    expected = {"name", "learning_rate_start", "learning_rate_end_ratio"}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(
            f"perfil con campos faltantes: {missing}; desconocidos: {unknown}"
        )
    name = raw["name"]
    start = float(raw["learning_rate_start"])
    ratio = float(raw["learning_rate_end_ratio"])
    if not isinstance(name, str) or not name:
        raise ValueError("el nombre del perfil debe ser un texto no vacío")
    if not 0 < start <= 0.5:
        raise ValueError("la tasa inicial debe pertenecer a (0, 0.5]")
    if not 0 < ratio <= 1:
        raise ValueError("el cociente de tasa final debe pertenecer a (0, 1]")
    return OjaProfile(name, start, ratio)


def load_design(path: Path | str) -> OjaEpochDesign:
    """Leer y validar el diseño reproducible de la segunda etapa."""

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {"name", "epochs", "profiles", "seeds", "shuffle"}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"campos faltantes: {missing}; campos desconocidos: {unknown}")

    epochs = tuple(raw["epochs"])
    profiles = tuple(_load_profile(profile) for profile in raw["profiles"])
    seeds = tuple(raw["seeds"])
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError("name debe ser un texto no vacío")
    if not epochs or len(epochs) != len(set(epochs)):
        raise ValueError("epochs debe contener duraciones únicas")
    if any(
        isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
        for epoch in epochs
    ):
        raise ValueError("las épocas deben ser enteros positivos")
    if not profiles or len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("los perfiles deben tener nombres únicos")
    if len(seeds) < 10 or len(seeds) != len(set(seeds)):
        raise ValueError("se requieren al menos 10 semillas únicas")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("las semillas deben ser enteros")
    if not isinstance(raw["shuffle"], bool):
        raise ValueError("shuffle debe ser booleano")

    return OjaEpochDesign(
        name=raw["name"],
        epochs=tuple(sorted(epochs)),
        profiles=profiles,
        seeds=seeds,
        shuffle=raw["shuffle"],
    )


def run_count(design: OjaEpochDesign) -> int:
    return len(design.epochs) * len(design.profiles) * len(design.seeds)


def _evaluate_model(
    model,
    standardized: np.ndarray,
    pca: PCAResult,
) -> tuple[float, float, float, float]:
    reference = pca.components[:, 0]
    aligned = align_sign(model.weights, reference)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = standardized @ aligned
    correlation = float(np.corrcoef(scores, pca.scores[:, 0])[0, 1])
    norm = float(np.linalg.norm(model.weights))
    return (
        direction_similarity(model.weights, reference),
        correlation,
        norm,
        abs(norm - 1.0),
    )


def evaluate_design(
    dataset: CreatorDataset,
    design: OjaEpochDesign,
) -> tuple[list[OjaEpochMetrics], PCAResult]:
    """Entrenar cada duración de forma independiente."""

    pca = fit_pca(dataset.values, dataset.feature_names)
    metrics = []
    for profile in design.profiles:
        end = profile.learning_rate_start * profile.learning_rate_end_ratio
        for epochs in design.epochs:
            for seed in design.seeds:
                name = f"{profile.name}-epochs-{epochs:03d}-seed-{seed:02d}"
                config = OjaConfig(
                    epochs=epochs,
                    learning_rate_start=profile.learning_rate_start,
                    learning_rate_end=end,
                    shuffle=design.shuffle,
                    seed=seed,
                )
                try:
                    model = fit_oja(dataset.values, dataset.feature_names, config)
                    similarity, correlation, norm, norm_error = _evaluate_model(
                        model, pca.standardized_values, pca
                    )
                except (FloatingPointError, OverflowError) as error:
                    metrics.append(
                        OjaEpochMetrics(
                            name,
                            profile.name,
                            epochs,
                            profile.learning_rate_start,
                            end,
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
                metrics.append(
                    OjaEpochMetrics(
                        name,
                        profile.name,
                        epochs,
                        profile.learning_rate_start,
                        end,
                        seed,
                        True,
                        similarity,
                        correlation,
                        norm,
                        norm_error,
                        "",
                    )
                )
    return metrics, pca


def _summary(values: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values, ddof=1)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def aggregate_runs(metrics: Sequence[OjaEpochMetrics]) -> list[dict]:
    """Agregar semillas por perfil y duración con desvío muestral."""

    rows = []
    combinations = sorted({(metric.profile, metric.epochs) for metric in metrics})
    for profile, epochs in combinations:
        group = [
            metric
            for metric in metrics
            if metric.profile == profile and metric.epochs == epochs
        ]
        successful = [metric for metric in group if metric.success]
        reference = group[0]
        row = {
            "profile": profile,
            "epochs": epochs,
            "learning_rate_start": reference.learning_rate_start,
            "learning_rate_end": reference.learning_rate_end,
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
            values = np.asarray([getattr(metric, field) for metric in successful])
            if len(values):
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


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    metrics: Sequence[OjaEpochMetrics],
    aggregate: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "corridas.csv",
        list(OjaEpochMetrics.__dataclass_fields__),
        (vars(metric) for metric in metrics),
    )
    _write_csv(generated / "resumen_epocas.csv", list(aggregate[0]), aggregate)


def write_boxplots(output: Path, metrics: Sequence[OjaEpochMetrics]) -> None:
    plt = _load_pyplot()
    profiles = sorted({metric.profile for metric in metrics})
    epochs = sorted({metric.epochs for metric in metrics})
    figure, axes = plt.subplots(1, len(profiles), figsize=(18, 5), sharey=True)
    for axis, profile in zip(axes, profiles):
        groups = [
            [
                metric.direction_similarity
                for metric in metrics
                if metric.success
                and metric.profile == profile
                and metric.epochs == epoch
            ]
            for epoch in epochs
        ]
        axis.boxplot(groups, tick_labels=[str(epoch) for epoch in epochs])
        axis.set(
            title=profile,
            xlabel="Épocas",
            ylabel="Similitud absoluta con PC1",
        )
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=0.8)
    figure.suptitle("Estabilidad de Oja según la duración (10 semillas)")
    figure.tight_layout()
    _save_figure(figure, output / "generated" / "similitud_por_epocas.svg")
    plt.close(figure)


def write_metric_curves(output: Path, aggregate: Sequence[dict]) -> None:
    plt = _load_pyplot()
    definitions = (
        ("direction_similarity", "Similitud con PC1", "Mayor es mejor"),
        ("score_correlation", "Correlación de índices", "Mayor es mejor"),
        ("norm_error", "Error de norma", "|‖w‖ - 1|; menor es mejor"),
    )
    profiles = sorted({row["profile"] for row in aggregate})
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, (field, title, ylabel) in zip(axes, definitions):
        for profile in profiles:
            rows = [row for row in aggregate if row["profile"] == profile]
            epochs = np.asarray([row["epochs"] for row in rows])
            means = np.asarray([row[f"{field}_mean"] for row in rows])
            deviations = np.asarray([row[f"{field}_std"] for row in rows])
            axis.errorbar(
                epochs,
                means,
                yerr=deviations,
                marker="o",
                capsize=3,
                linewidth=1.2,
                label=profile,
            )
        axis.set_xscale("log")
        axis.set(title=title, xlabel="Épocas", ylabel=ylabel)
        axis.set_xticks(sorted({row["epochs"] for row in aggregate}))
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(alpha=0.25)
    axes[-1].legend(title="Perfil", bbox_to_anchor=(1.03, 1), loc="upper left")
    figure.suptitle("Resultado final según la cantidad de épocas")
    figure.tight_layout()
    _save_figure(figure, output / "generated" / "metricas_por_epocas.svg")
    plt.close(figure)


def write_report(
    output: Path,
    design: OjaEpochDesign,
    aggregate: Sequence[dict],
    pca: PCAResult,
) -> None:
    rows = []
    for row in sorted(aggregate, key=lambda item: (item["profile"], item["epochs"])):
        similarity = (
            f'{row["direction_similarity_mean"]:.6f} ± '
            f'{row["direction_similarity_std"]:.6f}'
        )
        correlation = (
            f'{row["score_correlation_mean"]:.6f} ± '
            f'{row["score_correlation_std"]:.6f}'
        )
        norm_error = f'{row["norm_error_mean"]:.6f} ± {row["norm_error_std"]:.6f}'
        rows.append(
            f'| {row["profile"]} | {row["epochs"]} | '
            f'{row["successful_runs"]}/{row["runs"]} | {similarity} | '
            f'{correlation} | {norm_error} |'
        )

    profile_lines = "\n".join(
        f'- `{profile.name}`: `{profile.learning_rate_start:g}` -> '
        f'`{profile.learning_rate_start * profile.learning_rate_end_ratio:g}`.'
        for profile in design.profiles
    )
    selected = next(
        row
        for row in aggregate
        if row["profile"] == "decay-eta-0001" and row["epochs"] == 100
    )
    extended = next(
        row
        for row in aggregate
        if row["profile"] == "decay-eta-0001" and row["epochs"] == 500
    )
    similarity_gain = (
        extended["direction_similarity_mean"]
        - selected["direction_similarity_mean"]
    )
    norm_error_reduction = selected["norm_error_mean"] - extended["norm_error_mean"]
    report = f"""# Oja: etapa 2 - cantidad de épocas

## Protocolo

- Datos y z-score: los mismos 15 atributos y `Standardizer` que PCA.
- Duraciones: {', '.join(f'`{epoch}`' for epoch in design.epochs)} épocas.
- Semillas: {', '.join(str(seed) for seed in design.seeds)} ({len(design.seeds)} por combinación).
- Orden: barajado reproducible en cada época.
- Total: **{run_count(design)} corridas independientes**.

Perfiles provenientes de la etapa 1:

{profile_lines}

Cada duración se entrena desde cero. Esto es necesario porque en los perfiles
decrecientes la pendiente depende de la cantidad total de épocas.

PC1 explica {pca.explained_variance_ratio[0] * 100:.2f} % de la varianza y se
usa sólo como referencia de evaluación.

## Resultados

| Perfil | Épocas | Exitosas | Similitud con PC1 | Correlación de índices | Error de norma |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

![Boxplots por épocas](generated/similitud_por_epocas.svg)

Cada caja resume 10 semillas. Las semillas no se conectan entre sí porque no
constituyen una secuencia temporal.

![Métricas por épocas](generated/metricas_por_epocas.svg)

Los puntos representan promedios y las barras un desvío estándar. La elección
final debe buscar el comienzo de una meseta estable, no el mayor decimal de una
corrida aislada.

## Decisión

Se elige **`decay-eta-0001` con 100 épocas**, es decir, un descenso lineal de
`0.001` a `0.0001`. En las 10 semillas obtuvo una similitud media de
{selected["direction_similarity_mean"]:.9f}, con mínimo
{selected["direction_similarity_min"]:.9f}, y una correlación media entre el
índice de Oja y PC1 de {selected["score_correlation_mean"]:.9f}. El error medio
de norma fue {selected["norm_error_mean"]:.9f}.

Las 50 épocas todavía muestran una peor semilla de 0.997114, mientras que a las
100 todas superan 0.9999996. Extender el mismo perfil de 100 a 500 épocas mejora
la similitud media sólo {similarity_gain:.2e} y reduce el error de norma en
{norm_error_reduction:.6f}. Esa diferencia no justifica multiplicar por cinco
el entrenamiento.

El perfil `decay-eta-0005` converge antes, pero mantiene una norma más alejada
de 1. La tasa constante `0.001` entra en una meseta desde 50 épocas, aunque su
error de norma permanece alrededor de 0.00231, frente a 0.00045 para la opción
elegida. La decisión se apoya así en estabilidad, dirección, proyecciones y
norma, no en el mejor valor de una semilla aislada.

## Archivos generados

- `generated/corridas.csv`: las 240 corridas individuales;
- `generated/resumen_epocas.csv`: promedios, desvíos, medianas y extremos;
- dos SVG comparativos, sin gráficos redundantes por semilla.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def run_experiment(data: Path, config: Path, output: Path) -> list[dict]:
    dataset = load_creator_dataset(data)
    design = load_design(config)
    metrics, pca = evaluate_design(dataset, design)
    aggregate = aggregate_runs(metrics)
    write_tables(output, metrics, aggregate)
    write_boxplots(output, metrics)
    write_metric_curves(output, aggregate)
    write_report(output, design, aggregate, pca)
    return aggregate


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    aggregate = run_experiment(args.data, args.config, args.output)
    divergent = sum(row["divergent_runs"] for row in aggregate)
    total = sum(row["runs"] for row in aggregate)
    print(
        f"Etapa 2 de Oja completa: {total} corridas, {divergent} divergentes; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
