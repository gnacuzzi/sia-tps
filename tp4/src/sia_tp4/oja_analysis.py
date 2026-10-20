"""Entrenar, comparar e interpretar la configuración final de Oja."""

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
class OjaFinalDesign:
    """Configuración elegida y semillas usadas para verificar estabilidad."""

    name: str
    epochs: int
    learning_rate_start: float
    learning_rate_end: float
    seeds: Tuple[int, ...]
    shuffle: bool


@dataclass(frozen=True)
class OjaSeedResult:
    """Resultado completo de una semilla de la configuración final."""

    seed: int
    weights: np.ndarray
    weight_history: np.ndarray
    indices: np.ndarray
    direction_similarity: float
    score_correlation: float
    weight_norm: float
    norm_error: float


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
        default=Path("configs/oja/final.json"),
        help="Configuración final de Oja.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/oja-creadores"),
        help="Directorio del informe final.",
    )
    return parser.parse_args(argv)


def load_design(path: Path | str) -> OjaFinalDesign:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {
        "name",
        "epochs",
        "learning_rate_start",
        "learning_rate_end",
        "seeds",
        "shuffle",
    }
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"campos faltantes: {missing}; campos desconocidos: {unknown}")

    seeds = tuple(raw["seeds"])
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise ValueError("name debe ser un texto no vacío")
    if (
        isinstance(raw["epochs"], bool)
        or not isinstance(raw["epochs"], int)
        or raw["epochs"] < 1
    ):
        raise ValueError("epochs debe ser un entero positivo")
    start = float(raw["learning_rate_start"])
    end = float(raw["learning_rate_end"])
    if not 0 < end <= start <= 0.5:
        raise ValueError("las tasas deben cumplir 0 < final <= inicial <= 0.5")
    if len(seeds) < 10 or len(seeds) != len(set(seeds)):
        raise ValueError("se requieren al menos 10 semillas únicas")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("las semillas deben ser enteros")
    if not isinstance(raw["shuffle"], bool):
        raise ValueError("shuffle debe ser booleano")
    return OjaFinalDesign(
        raw["name"], raw["epochs"], start, end, seeds, raw["shuffle"]
    )


def fit_final_runs(
    dataset: CreatorDataset,
    design: OjaFinalDesign,
) -> tuple[list[OjaSeedResult], PCAResult]:
    """Entrenar todas las semillas y alinear sus signos con PC1."""

    pca = fit_pca(dataset.values, dataset.feature_names)
    results = []
    for seed in design.seeds:
        model = fit_oja(
            dataset.values,
            dataset.feature_names,
            OjaConfig(
                epochs=design.epochs,
                learning_rate_start=design.learning_rate_start,
                learning_rate_end=design.learning_rate_end,
                shuffle=design.shuffle,
                seed=seed,
            ),
        )
        weights = align_sign(model.weights, pca.components[:, 0])
        indices = pca.standardized_values @ weights
        correlation = float(np.corrcoef(indices, pca.scores[:, 0])[0, 1])
        norm = float(np.linalg.norm(weights))
        results.append(
            OjaSeedResult(
                seed=seed,
                weights=weights,
                weight_history=model.weight_history,
                indices=indices,
                direction_similarity=direction_similarity(
                    weights, pca.components[:, 0]
                ),
                score_correlation=correlation,
                weight_norm=norm,
                norm_error=abs(norm - 1.0),
            )
        )
    return results, pca


def select_representative(results: Sequence[OjaSeedResult]) -> OjaSeedResult:
    """Elegir la corrida más cercana a las medianas, no la de mejor métrica."""

    if not results:
        raise ValueError("se necesita al menos una corrida")
    matrix = np.asarray(
        [
            [
                result.direction_similarity,
                result.score_correlation,
                result.norm_error,
            ]
            for result in results
        ]
    )
    medians = np.median(matrix, axis=0)
    scales = np.std(matrix, axis=0, ddof=1) if len(results) > 1 else np.ones(3)
    scales = np.where(scales > 0, scales, 1.0)
    distances = np.linalg.norm((matrix - medians) / scales, axis=1)
    return results[int(np.argmin(distances))]


def _rank_descending(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    dataset: CreatorDataset,
    results: Sequence[OjaSeedResult],
    representative: OjaSeedResult,
    pca: PCAResult,
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(
        generated / "metricas_semillas.csv",
        [
            "seed",
            "direction_similarity",
            "score_correlation",
            "weight_norm",
            "norm_error",
            "representative",
        ],
        (
            {
                "seed": result.seed,
                "direction_similarity": result.direction_similarity,
                "score_correlation": result.score_correlation,
                "weight_norm": result.weight_norm,
                "norm_error": result.norm_error,
                "representative": result.seed == representative.seed,
            }
            for result in results
        ),
    )
    _write_csv(
        generated / "pesos_comparados.csv",
        ["variable", "oja_weight", "pc1_eigenvector", "difference"],
        (
            {
                "variable": variable,
                "oja_weight": oja_weight,
                "pc1_eigenvector": pc1_weight,
                "difference": oja_weight - pc1_weight,
            }
            for variable, oja_weight, pc1_weight in zip(
                dataset.feature_names,
                representative.weights,
                pca.components[:, 0],
            )
        ),
    )
    oja_ranks = _rank_descending(representative.indices)
    pc1_ranks = _rank_descending(pca.scores[:, 0])
    _write_csv(
        generated / "indices.csv",
        [
            "creator_id",
            "name",
            "followers",
            "oja_index",
            "pc1_score",
            "difference",
            "oja_rank",
            "pc1_rank",
        ],
        (
            {
                "creator_id": creator_id,
                "name": name,
                "followers": followers,
                "oja_index": oja_index,
                "pc1_score": pc1_score,
                "difference": oja_index - pc1_score,
                "oja_rank": oja_rank,
                "pc1_rank": pc1_rank,
            }
            for creator_id, name, followers, oja_index, pc1_score, oja_rank, pc1_rank in zip(
                dataset.creator_ids,
                dataset.names,
                dataset.followers,
                representative.indices,
                pca.scores[:, 0],
                oja_ranks,
                pc1_ranks,
            )
        ),
    )


def write_plots(
    output: Path,
    dataset: CreatorDataset,
    results: Sequence[OjaSeedResult],
    representative: OjaSeedResult,
    pca: PCAResult,
) -> None:
    plt = _load_pyplot()
    generated = output / "generated"

    positions = np.arange(len(dataset.feature_names))
    figure, axis = plt.subplots(figsize=(11, 8))
    width = 0.38
    axis.barh(
        positions - width / 2,
        pca.components[:, 0],
        height=width,
        label="Autovector PC1",
        color="#4C78A8",
    )
    axis.barh(
        positions + width / 2,
        representative.weights,
        height=width,
        label=f"Oja (semilla {representative.seed})",
        color="#F58518",
    )
    axis.set_yticks(positions, dataset.feature_names)
    axis.set(title="Pesos de Oja frente al autovector de PC1", xlabel="Coeficiente")
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, generated / "pesos_vs_pc1.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(pca.scores[:, 0], representative.indices, alpha=0.8)
    limits = np.asarray(
        [
            min(pca.scores[:, 0].min(), representative.indices.min()),
            max(pca.scores[:, 0].max(), representative.indices.max()),
        ]
    )
    axis.plot(limits, limits, linestyle="--", color="#777777", label="y = x")
    for creator_id, x_value, y_value in zip(
        dataset.creator_ids, pca.scores[:, 0], representative.indices
    ):
        axis.annotate(
            creator_id,
            (x_value, y_value),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
            alpha=0.7,
        )
    axis.set(
        title="Índice de Oja frente a los scores de PC1",
        xlabel="Score PC1",
        ylabel="Índice de Oja",
        xlim=limits,
        ylim=limits,
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, generated / "indices_vs_pc1.svg")
    plt.close(figure)

    order = np.argsort(pca.scores[:, 0])
    figure, axis = plt.subplots(figsize=(10, 13))
    rows = np.arange(len(order))
    axis.scatter(
        pca.scores[order, 0], rows, s=22, label="PC1", color="#4C78A8"
    )
    axis.scatter(
        representative.indices[order],
        rows,
        s=16,
        label="Oja",
        color="#F58518",
    )
    axis.set_yticks(
        rows,
        [
            f"{dataset.creator_ids[index]} {dataset.names[index]}"
            for index in order
        ],
        fontsize=7,
    )
    axis.set(
        title="Creadores ordenados sobre la dirección principal",
        xlabel="Proyección estandarizada",
        ylabel="Creador",
    )
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, generated / "indices_ordenados.svg")
    plt.close(figure)

    histories = []
    for result in results:
        similarities = [
            direction_similarity(weights, pca.components[:, 0])
            for weights in result.weight_history
        ]
        norm_errors = np.abs(np.linalg.norm(result.weight_history, axis=1) - 1.0)
        histories.append((similarities, norm_errors))
    similarities = np.asarray([history[0] for history in histories])
    norm_errors = np.asarray([history[1] for history in histories])
    epochs = np.arange(similarities.shape[1])
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, values, title, ylabel in (
        (axes[0], similarities, "Dirección aprendida", "Similitud con PC1"),
        (axes[1], norm_errors, "Norma de los pesos", "|‖w‖ - 1|"),
    ):
        mean = values.mean(axis=0)
        deviation = values.std(axis=0, ddof=1)
        axis.plot(epochs, mean, color="#4C78A8")
        axis.fill_between(
            epochs,
            np.maximum(mean - deviation, 0.0),
            mean + deviation,
            color="#4C78A8",
            alpha=0.2,
            label="media ± 1 desvío",
        )
        axis.set(title=title, xlabel="Época", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Convergencia de la configuración final en 10 semillas")
    figure.tight_layout()
    _save_figure(figure, generated / "convergencia_final.svg")
    plt.close(figure)


def _creator_rows(
    dataset: CreatorDataset,
    representative: OjaSeedResult,
    count: int = 5,
) -> tuple[str, str]:
    order = np.argsort(representative.indices)

    def rows(indices: np.ndarray) -> str:
        return "\n".join(
            f"| {dataset.creator_ids[index]} | {dataset.names[index]} | "
            f"{representative.indices[index]:+.4f} |"
            for index in indices
        )

    return rows(order[-count:][::-1]), rows(order[:count])


def write_report(
    output: Path,
    dataset: CreatorDataset,
    design: OjaFinalDesign,
    results: Sequence[OjaSeedResult],
    representative: OjaSeedResult,
    pca: PCAResult,
) -> None:
    similarities = np.asarray([result.direction_similarity for result in results])
    correlations = np.asarray([result.score_correlation for result in results])
    norm_errors = np.asarray([result.norm_error for result in results])
    weight_difference = representative.weights - pca.components[:, 0]
    score_difference = representative.indices - pca.scores[:, 0]
    high_rows, low_rows = _creator_rows(dataset, representative)
    report = f"""# Índice de creadores mediante la regla de Oja

## Configuración final

- Variables: las mismas 15 entradas y el mismo orden que PCA.
- Estandarización: z-score muestral mediante `Standardizer`.
- Tasa: descenso lineal de `{design.learning_rate_start:g}` a `{design.learning_rate_end:g}`.
- Épocas: **{design.epochs}**.
- Orden: barajado reproducible por época.
- Estabilidad: {len(results)} semillas ({', '.join(str(seed) for seed in design.seeds)}).

La corrida representativa es la semilla **{representative.seed}**, elegida por
ser la más cercana a las medianas de similitud, correlación y error de norma.
No se eligió la semilla con el mejor resultado.

## Convergencia y comparación con PCA

| Evidencia en {len(results)} semillas | Media | Desvío | Mínimo | Máximo |
| --- | ---: | ---: | ---: | ---: |
| Similitud de dirección | {similarities.mean():.9f} | {similarities.std(ddof=1):.9f} | {similarities.min():.9f} | {similarities.max():.9f} |
| Correlación índice-PC1 | {correlations.mean():.9f} | {correlations.std(ddof=1):.9f} | {correlations.min():.9f} | {correlations.max():.9f} |
| Error de norma | {norm_errors.mean():.9f} | {norm_errors.std(ddof=1):.9f} | {norm_errors.min():.9f} | {norm_errors.max():.9f} |

![Convergencia final](generated/convergencia_final.svg)

Oja converge a los **coeficientes del autovector** de PC1. No debe compararse
con la tabla de cargas de PCA, porque las cargas incorporan además la raíz del
autovalor. Después de alinear el signo, la máxima diferencia absoluta entre un
peso representativo de Oja y el autovector es
{np.max(np.abs(weight_difference)):.6f}.

![Pesos comparados](generated/pesos_vs_pc1.svg)

El índice se calcula para cada creador como `Z @ w_Oja`. Su correlación con los
scores de PC1 en la corrida representativa es
{representative.score_correlation:.9f}; la diferencia absoluta máxima entre
ambas proyecciones es {np.max(np.abs(score_difference)):.6f}.

![Índice frente a PC1](generated/indices_vs_pc1.svg)

![Índices ordenados](generated/indices_ordenados.svg)

## ¿Qué representa este índice?

Es la posición de cada creador sobre la dirección de **máxima variabilidad** de
las 15 variables estandarizadas. Resume principalmente la oposición ya
identificada en PC1: valores altos se asocian con mayor alcance, precio,
descubrimiento por no seguidores y entretenimiento; valores bajos se asocian
con más visitas al perfil, tutoriales, reviews, tiempo de reproducción y
contenido de perros.

No representa ventas, adopciones, retorno de inversión ni calidad general. Es
un resumen descriptivo no supervisado. PC1 explica {pca.explained_variance_ratio[0] * 100:.2f} %
de la varianza: usar solamente el índice deja fuera el {100 * (1 - pca.explained_variance_ratio[0]):.2f} %
restante y los ejes temáticos más específicos de otras componentes.

## ¿Cómo se puede interpretar?

El signo se alineó con la orientación usada en el informe de PCA para facilitar
la comparación, pero podría invertirse sin cambiar la solución. Por eso
“positivo” no significa “bueno” y “negativo” no significa “malo”. Los extremos
describen dos perfiles opuestos y los valores cercanos a cero, perfiles menos
extremos respecto de este eje.

Mayores índices de la corrida representativa:

| ID | Creador | Índice Oja |
| --- | --- | ---: |
{high_rows}

Menores índices:

| ID | Creador | Índice Oja |
| --- | --- | ---: |
{low_rows}

## ¿Sería correcto elegir automáticamente el valor más alto?

**No.** El máximo identifica al creador más extremo en la dirección positiva de
PC1, no al mejor para cualquier campaña. En este caso, precio tiene una de las
cargas positivas más importantes: un índice alto puede implicar también mayor
costo. Además, una campaña de perros, fútbol, belleza o arte necesita afinidad
temática y un formato adecuado; una única proyección no conserva toda esa
información ni incorpora un resultado supervisado de ventas o adopciones.

El índice sirve para ordenar y comparar creadores sobre el eje general de
escala/descubrimiento frente a profundidad/interés. La decisión final debe
combinarlo con la región de Kohonen, los ejes temáticos, audiencia argentina,
objetivo de la campaña, costo y saturación comercial. Elegir siempre el máximo
convertiría una descripción de variabilidad en una función de utilidad que el
modelo nunca aprendió.

## Archivos generados

- `generated/metricas_semillas.csv`: estabilidad y corrida representativa;
- `generated/pesos_comparados.csv`: pesos de Oja y autovector de PC1;
- `generated/indices.csv`: índice, score y posiciones de cada creador;
- cuatro SVG para convergencia, pesos y proyecciones.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def run_analysis(data: Path, config: Path, output: Path) -> OjaSeedResult:
    dataset = load_creator_dataset(data)
    design = load_design(config)
    results, pca = fit_final_runs(dataset, design)
    representative = select_representative(results)
    write_tables(output, dataset, results, representative, pca)
    write_plots(output, dataset, results, representative, pca)
    write_report(output, dataset, design, results, representative, pca)
    return representative


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    representative = run_analysis(args.data, args.config, args.output)
    print(
        f"Oja final completo: semilla representativa {representative.seed}; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
