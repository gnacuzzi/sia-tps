"""Comparar PCA con engagement total y con sus cuatro componentes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import (
    BASE_FEATURES,
    ENGAGEMENT_COMPONENT_FEATURES,
    ENGAGEMENT_COMPONENTS,
    CreatorDataset,
    load_creator_dataset,
)
from .pca import PCAResult, fit_pca
from .pca_analysis import _load_pyplot, _save_figure, write_plots, write_tables


@dataclass(frozen=True)
class EngagementComparison:
    base_dataset: CreatorDataset
    component_dataset: CreatorDataset
    base_result: PCAResult
    component_result: PCAResult
    threshold: float
    base_component_count: int
    component_component_count: int
    pc1_score_correlation: float
    pc2_score_correlation: float
    two_dimensional_distance_correlation: float
    retained_distance_correlation: float
    score_correlation_matrix: np.ndarray
    engagement_correlation_matrix: np.ndarray


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
        default=Path("docs/pca-engagement"),
        help="Directorio del informe (por defecto, docs/pca-engagement).",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.80,
        help="Umbral exploratorio de varianza acumulada (por defecto, 0.80).",
    )
    return parser.parse_args(argv)


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("las series comparadas deben ser vectores del mismo tamaño")
    correlation = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(correlation):
        raise ValueError("no se puede calcular correlación con una serie constante")
    return correlation


def pairwise_distance_correlation(
    left_scores: np.ndarray, right_scores: np.ndarray
) -> float:
    """Comparar la geometría mediante todas las distancias entre creadores."""

    if left_scores.ndim != 2 or right_scores.ndim != 2:
        raise ValueError("los scores deben ser matrices")
    if left_scores.shape[0] != right_scores.shape[0] or left_scores.shape[0] < 2:
        raise ValueError("las matrices deben tener los mismos creadores")
    upper = np.triu_indices(left_scores.shape[0], 1)
    left_distances = np.linalg.norm(
        left_scores[:, None, :] - left_scores[None, :, :], axis=2
    )[upper]
    right_distances = np.linalg.norm(
        right_scores[:, None, :] - right_scores[None, :, :], axis=2
    )[upper]
    return _pearson(left_distances, right_distances)


def compare_engagement(data_path: Path | str, threshold: float = 0.80) -> EngagementComparison:
    base_dataset = load_creator_dataset(data_path, BASE_FEATURES)
    component_dataset = load_creator_dataset(
        data_path, ENGAGEMENT_COMPONENT_FEATURES
    )
    if base_dataset.creator_ids != component_dataset.creator_ids:
        raise ValueError("las variantes no contienen los mismos creadores y orden")

    base_result = fit_pca(base_dataset.values, base_dataset.feature_names)
    component_result = fit_pca(
        component_dataset.values, component_dataset.feature_names
    )
    base_count = base_result.component_count(threshold)
    component_count = component_result.component_count(threshold)
    score_correlations = np.corrcoef(
        base_result.scores[:, :base_count].T,
        component_result.scores[:, :component_count].T,
    )[:base_count, base_count:]

    engagement_dataset = load_creator_dataset(
        data_path, (*ENGAGEMENT_COMPONENTS, "total_engagement_per_1000")
    )
    return EngagementComparison(
        base_dataset=base_dataset,
        component_dataset=component_dataset,
        base_result=base_result,
        component_result=component_result,
        threshold=threshold,
        base_component_count=base_count,
        component_component_count=component_count,
        pc1_score_correlation=abs(
            _pearson(base_result.scores[:, 0], component_result.scores[:, 0])
        ),
        pc2_score_correlation=abs(
            _pearson(base_result.scores[:, 1], component_result.scores[:, 1])
        ),
        two_dimensional_distance_correlation=pairwise_distance_correlation(
            base_result.scores[:, :2], component_result.scores[:, :2]
        ),
        retained_distance_correlation=pairwise_distance_correlation(
            base_result.scores[:, :base_count],
            component_result.scores[:, :component_count],
        ),
        score_correlation_matrix=score_correlations,
        engagement_correlation_matrix=np.corrcoef(
            engagement_dataset.values, rowvar=False
        ),
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_tables(output: Path, comparison: EngagementComparison) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("base_feature_count", len(comparison.base_dataset.feature_names)),
        ("component_feature_count", len(comparison.component_dataset.feature_names)),
        ("base_components_for_threshold", comparison.base_component_count),
        ("component_components_for_threshold", comparison.component_component_count),
        ("base_pc1_variance", comparison.base_result.explained_variance_ratio[0]),
        ("component_pc1_variance", comparison.component_result.explained_variance_ratio[0]),
        ("base_pc1_pc2_variance", comparison.base_result.cumulative_explained_variance[1]),
        ("component_pc1_pc2_variance", comparison.component_result.cumulative_explained_variance[1]),
        ("pc1_score_correlation", comparison.pc1_score_correlation),
        ("pc2_score_correlation", comparison.pc2_score_correlation),
        ("pc1_pc2_distance_correlation", comparison.two_dimensional_distance_correlation),
        ("retained_distance_correlation", comparison.retained_distance_correlation),
    ]
    _write_csv(
        generated / "metricas_comparacion.csv",
        ["metric", "value"],
        [{"metric": name, "value": value} for name, value in metrics],
    )

    base_names = [f"base_PC{index}" for index in range(1, comparison.base_component_count + 1)]
    component_names = [
        f"components_PC{index}"
        for index in range(1, comparison.component_component_count + 1)
    ]
    _write_csv(
        generated / "correlacion_scores.csv",
        ["component", *component_names],
        [
            {
                "component": base_name,
                **dict(zip(component_names, row)),
            }
            for base_name, row in zip(
                base_names, comparison.score_correlation_matrix
            )
        ],
    )

    engagement_names = [*ENGAGEMENT_COMPONENTS, "total_engagement_per_1000"]
    _write_csv(
        generated / "correlacion_engagement.csv",
        ["variable", *engagement_names],
        [
            {"variable": name, **dict(zip(engagement_names, row))}
            for name, row in zip(
                engagement_names, comparison.engagement_correlation_matrix
            )
        ],
    )


def write_comparison_plots(output: Path, comparison: EngagementComparison) -> None:
    plt = _load_pyplot()
    generated = output / "generated"

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        np.arange(1, len(comparison.base_result.eigenvalues) + 1),
        comparison.base_result.cumulative_explained_variance * 100,
        marker="o",
        label="Total (15 variables)",
        color="#4C78A8",
    )
    axis.plot(
        np.arange(1, len(comparison.component_result.eigenvalues) + 1),
        comparison.component_result.cumulative_explained_variance * 100,
        marker="o",
        label="Componentes (18 variables)",
        color="#E45756",
    )
    axis.axhline(comparison.threshold * 100, color="#555555", linestyle="--")
    axis.set(
        title="Varianza acumulada según representación del engagement",
        xlabel="Cantidad de componentes principales",
        ylabel="Varianza acumulada (%)",
        ylim=(0, 105),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, generated / "varianza_comparada.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for index, (axis, correlation) in enumerate(
        zip(
            axes,
            (comparison.pc1_score_correlation, comparison.pc2_score_correlation),
        )
    ):
        left = comparison.base_result.scores[:, index]
        right = comparison.component_result.scores[:, index]
        axis.scatter(left, right, color="#4C78A8", alpha=0.8)
        coefficients = np.polyfit(left, right, deg=1)
        x_line = np.linspace(left.min(), left.max(), 100)
        axis.plot(x_line, np.polyval(coefficients, x_line), color="#E45756")
        axis.set(
            title=f"PC{index + 1}: |r| = {correlation:.3f}",
            xlabel=f"PC{index + 1} con engagement total",
            ylabel=f"PC{index + 1} con componentes",
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Estabilidad de los scores de los creadores")
    figure.tight_layout()
    _save_figure(figure, generated / "scores_comparados.svg")
    plt.close(figure)


def _engagement_loading_table(comparison: EngagementComparison, count: int = 6) -> str:
    positions = [
        comparison.component_dataset.feature_names.index(name)
        for name in ENGAGEMENT_COMPONENTS
    ]
    header = "| Variable | " + " | ".join(
        f"PC{index}" for index in range(1, count + 1)
    ) + " |"
    separator = "| --- | " + " | ".join("---:" for _ in range(count)) + " |"
    rows = [header, separator]
    for name, position in zip(ENGAGEMENT_COMPONENTS, positions):
        values = comparison.component_result.loadings[position, :count]
        rows.append(
            f"| `{name}` | "
            + " | ".join(f"{value:+.3f}" for value in values)
            + " |"
        )
    return "\n".join(rows)


def write_report(output: Path, comparison: EngagementComparison) -> None:
    base = comparison.base_result
    components = comparison.component_result
    total_position = len(ENGAGEMENT_COMPONENTS)
    total_correlations = comparison.engagement_correlation_matrix[
        total_position, :total_position
    ]
    report = f"""# Comparación de representaciones del engagement

Este experimento compara dos PCA sobre los mismos 50 creadores y con el mismo
z-score por variable:

- **Variante A:** 15 variables y `total_engagement_per_1000`.
- **Variante B:** elimina el total y agrega likes, comentarios, guardados y
  compartidos; contiene 18 variables y se usa sólo como diagnóstico.

No se comparan los porcentajes como si una variante fuera automáticamente
mejor. La variante B otorga cuatro dimensiones al bloque de engagement y, por
eso, cambia el peso relativo de esa familia.

## Resultado general

| Evidencia | Variante A | Variante B |
| --- | ---: | ---: |
| Variables | {len(comparison.base_dataset.feature_names)} | {len(comparison.component_dataset.feature_names)} |
| Varianza de PC1 | {base.explained_variance_ratio[0] * 100:.2f} % | {components.explained_variance_ratio[0] * 100:.2f} % |
| Varianza acumulada PC1-PC2 | {base.cumulative_explained_variance[1] * 100:.2f} % | {components.cumulative_explained_variance[1] * 100:.2f} % |
| Componentes para superar {comparison.threshold * 100:.0f} % | {comparison.base_component_count} | {comparison.component_component_count} |

![Varianza acumulada comparada](generated/varianza_comparada.svg)

La variante B necesita {comparison.component_component_count} componentes para
alcanzar {components.cumulative_explained_variance[comparison.component_component_count - 1] * 100:.2f} %.
La variante A necesita {comparison.base_component_count} para alcanzar
{base.cumulative_explained_variance[comparison.base_component_count - 1] * 100:.2f} %.
Esto no define una ganadora porque parten de 18 y 15 dimensiones distintas.

## Estabilidad de los perfiles

- Correlación absoluta entre scores de PC1: **{comparison.pc1_score_correlation:.3f}**.
- Correlación absoluta entre scores de PC2: **{comparison.pc2_score_correlation:.3f}**.
- Correlación de distancias en el plano PC1-PC2: **{comparison.two_dimensional_distance_correlation:.3f}**.
- Correlación de distancias usando las componentes que superan el umbral:
  **{comparison.retained_distance_correlation:.3f}**.

![Scores comparados](generated/scores_comparados.svg)

Los valores cercanos a 1 indican que los perfiles globales y las relaciones de
proximidad entre creadores cambian poco al separar el engagement. Se usa valor
absoluto para los scores porque el signo de una componente es arbitrario.

## Información adicional de la variante B

{_engagement_loading_table(comparison)}

La variante B muestra una oposición importante en PC1: likes tiene carga
positiva ({components.loadings[comparison.component_dataset.feature_names.index('likes_per_1000'), 0]:+.3f})
y guardados tiene carga negativa
({components.loadings[comparison.component_dataset.feature_names.index('saves_per_1000'), 0]:+.3f}).
El total no puede mostrar esa composición. Compartidos y comentarios también
aparecen en componentes posteriores, pero esas componentes explican menos
varianza y no alteran mucho la geometría general de los creadores.

En los datos originales, el engagement total correlaciona
{total_correlations[0]:+.3f} con likes, {total_correlations[1]:+.3f} con
comentarios, {total_correlations[2]:+.3f} con guardados y
{total_correlations[3]:+.3f} con compartidos. Por eso el total refleja
principalmente la variación de likes y compartidos, mientras que comentarios y
guardados aportan información de composición que conviene conservar como
auxiliar.

El PCA completo de la variante B se encuentra en
`variante-componentes/generated/`, con cargas, scores, autovectores y biplot.

## Recomendación para discutir

La evidencia favorece conservar **engagement total** como entrada principal:

1. mantiene el conjunto en 15 variables, dentro del rango sugerido en clase;
2. PC1, PC2 y las distancias entre creadores permanecen casi iguales;
3. evita que engagement tenga cuatro veces más columnas que otras familias;
4. facilita interpretar Kohonen y Oja.

Los cuatro componentes deben permanecer como variables auxiliares. Sirven para
describir regiones de Kohonen o revisar una campaña que valore específicamente
guardados, comentarios o compartidos. Si una necesidad comercial futura exige
esa distinción, se puede reabrir la selección.

Esta es una recomendación del análisis y queda pendiente de confirmación antes
de fijar las entradas definitivas de Kohonen y Oja.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def run_comparison(
    data: Path, output: Path, threshold: float
) -> EngagementComparison:
    comparison = compare_engagement(data, threshold)
    variant_output = output / "variante-componentes"
    write_tables(
        variant_output,
        comparison.component_dataset,
        comparison.component_result,
    )
    write_plots(
        variant_output,
        comparison.component_dataset,
        comparison.component_result,
        comparison.component_component_count,
    )
    write_comparison_tables(output, comparison)
    write_comparison_plots(output, comparison)
    write_report(output, comparison)
    return comparison


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    comparison = run_comparison(
        args.data, args.output, args.variance_threshold
    )
    print(
        "Comparación completa: "
        f"r(PC1)={comparison.pc1_score_correlation:.3f}, "
        f"r(PC2)={comparison.pc2_score_correlation:.3f}; "
        f"informe en {args.output / 'README.md'}"
    )


if __name__ == "__main__":
    main()
