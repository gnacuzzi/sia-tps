"""Validar Kohonen con cuatro grupos sintéticos de estructura conocida."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from .kohonen import KohonenModel, fit_kohonen
from .kohonen_analysis import KohonenRun, load_run
from .pca_analysis import _load_pyplot, _save_figure


CLUSTER_NAMES: Tuple[str, ...] = (
    "inferior-izquierdo",
    "superior-izquierdo",
    "inferior-derecho",
    "superior-derecho",
)
CLUSTER_CENTERS = np.asarray(
    [(-3.0, -3.0), (-3.0, 3.0), (3.0, -3.0), (3.0, 3.0)],
    dtype=np.float64,
)


@dataclass(frozen=True)
class SyntheticValidation:
    """Datos, modelo y verificaciones del caso controlado."""

    values: np.ndarray
    labels: np.ndarray
    model: KohonenModel
    assignments: np.ndarray
    hit_map: np.ndarray
    u_matrix: np.ndarray
    reproducible: bool
    purity: float
    mean_within_cluster_grid_distance: float
    mean_between_cluster_grid_distance: float
    maximum_center_error: float

    @property
    def all_samples_assigned(self) -> bool:
        return int(self.hit_map.sum()) == len(self.values)

    @property
    def known_structure_recovered(self) -> bool:
        return (
            self.purity >= 0.95
            and self.mean_within_cluster_grid_distance
            < self.mean_between_cluster_grid_distance
            and self.maximum_center_error <= 1.0
        )

    @property
    def passed(self) -> bool:
        return (
            self.all_samples_assigned
            and self.reproducible
            and np.isfinite(self.model.weights).all()
            and np.isfinite(self.u_matrix).all()
            and self.known_structure_recovered
        )


def generate_synthetic_data(
    seed: int = 2026,
    samples_per_cluster: int = 30,
    standard_deviation: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Crear cuatro nubes gaussianas separadas y mezclarlas."""

    if samples_per_cluster < 2:
        raise ValueError("se necesitan al menos dos muestras por grupo")
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("el desvío debe ser positivo y finito")
    generator = np.random.default_rng(seed)
    blocks = [
        generator.normal(center, standard_deviation, (samples_per_cluster, 2))
        for center in CLUSTER_CENTERS
    ]
    values = np.vstack(blocks)
    labels = np.repeat(np.arange(len(CLUSTER_CENTERS)), samples_per_cluster)
    order = generator.permutation(len(values))
    return values[order], labels[order]


def _cluster_purity(
    assignments: np.ndarray, labels: np.ndarray, grid_size: int
) -> float:
    correct = 0
    for row in range(grid_size):
        for column in range(grid_size):
            cell_labels = labels[
                (assignments[:, 0] == row) & (assignments[:, 1] == column)
            ]
            if len(cell_labels):
                correct += int(np.bincount(cell_labels).max())
    return correct / len(labels)


def _pairwise_grid_distances(
    assignments: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    distances = np.linalg.norm(
        assignments[:, None, :] - assignments[None, :, :], axis=2
    )
    upper = np.triu_indices(len(assignments), 1)
    same_cluster = labels[upper[0]] == labels[upper[1]]
    pair_distances = distances[upper]
    return (
        float(pair_distances[same_cluster].mean()),
        float(pair_distances[~same_cluster].mean()),
    )


def _maximum_center_error(model: KohonenModel) -> float:
    prototypes = model.prototypes(original_scale=True).reshape(-1, 2)
    distances = np.linalg.norm(
        CLUSTER_CENTERS[:, None, :] - prototypes[None, :, :], axis=2
    )
    return float(distances.min(axis=1).max())


def validate_synthetic(run: KohonenRun) -> SyntheticValidation:
    """Entrenar dos veces y contrastar el mapa con la estructura conocida."""

    values, labels = generate_synthetic_data()
    first = fit_kohonen(values, ("x", "y"), run.config)
    second = fit_kohonen(values, ("x", "y"), run.config)
    assignments = first.map_samples(values)
    within, between = _pairwise_grid_distances(assignments, labels)
    return SyntheticValidation(
        values=values,
        labels=labels,
        model=first,
        assignments=assignments,
        hit_map=first.hit_map(values),
        u_matrix=first.u_matrix(),
        reproducible=bool(np.allclose(first.weights, second.weights)),
        purity=_cluster_purity(assignments, labels, run.config.grid_size),
        mean_within_cluster_grid_distance=within,
        mean_between_cluster_grid_distance=between,
        maximum_center_error=_maximum_center_error(first),
    )


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _dominant_clusters(validation: SyntheticValidation) -> np.ndarray:
    size = validation.model.config.grid_size
    dominant = np.full((size, size), -1, dtype=int)
    for row in range(size):
        for column in range(size):
            labels = validation.labels[
                (validation.assignments[:, 0] == row)
                & (validation.assignments[:, 1] == column)
            ]
            if len(labels):
                dominant[row, column] = int(np.bincount(labels).argmax())
    return dominant


def write_tables(output: Path, validation: SyntheticValidation) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("all_samples_assigned", validation.all_samples_assigned),
        ("reproducible", validation.reproducible),
        ("finite_weights", bool(np.isfinite(validation.model.weights).all())),
        ("finite_u_matrix", bool(np.isfinite(validation.u_matrix).all())),
        ("cluster_purity", validation.purity),
        (
            "mean_within_cluster_grid_distance",
            validation.mean_within_cluster_grid_distance,
        ),
        (
            "mean_between_cluster_grid_distance",
            validation.mean_between_cluster_grid_distance,
        ),
        ("maximum_center_error", validation.maximum_center_error),
        ("known_structure_recovered", validation.known_structure_recovered),
        ("validation_passed", validation.passed),
    )
    _write_csv(
        generated / "metricas.csv",
        ("metric", "value"),
        ({"metric": name, "value": value} for name, value in metrics),
    )
    _write_csv(
        generated / "muestras.csv",
        ("sample", "cluster", "x", "y", "bmu_row", "bmu_column"),
        (
            {
                "sample": index,
                "cluster": CLUSTER_NAMES[label],
                "x": value[0],
                "y": value[1],
                "bmu_row": assignment[0],
                "bmu_column": assignment[1],
            }
            for index, (value, label, assignment) in enumerate(
                zip(
                    validation.values,
                    validation.labels,
                    validation.assignments,
                )
            )
        ),
    )
    prototypes = validation.model.prototypes(original_scale=True)
    dominant = _dominant_clusters(validation)
    _write_csv(
        generated / "prototipos.csv",
        ("row", "column", "x", "y", "hits", "dominant_cluster"),
        (
            {
                "row": row,
                "column": column,
                "x": prototypes[row, column, 0],
                "y": prototypes[row, column, 1],
                "hits": validation.hit_map[row, column],
                "dominant_cluster": (
                    "empty"
                    if dominant[row, column] < 0
                    else CLUSTER_NAMES[dominant[row, column]]
                ),
            }
            for row in range(validation.model.config.grid_size)
            for column in range(validation.model.config.grid_size)
        ),
    )


def _annotated_heatmap(plt, matrix, title, label, value_format):
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="viridis", aspect="equal")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                format(matrix[row, column], value_format),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.mean() else "black",
            )
    axis.set(
        title=title,
        xlabel="Columna de la grilla",
        ylabel="Fila de la grilla",
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
    )
    figure.colorbar(image, ax=axis, label=label)
    return figure


def write_plots(output: Path, validation: SyntheticValidation) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756")

    figure, axis = plt.subplots(figsize=(8, 7))
    for label, name, color in zip(range(4), CLUSTER_NAMES, colors):
        points = validation.values[validation.labels == label]
        axis.scatter(points[:, 0], points[:, 1], label=name, color=color, alpha=0.75)
    prototypes = validation.model.prototypes(original_scale=True)
    size = validation.model.config.grid_size
    for row in range(size):
        axis.plot(prototypes[row, :, 0], prototypes[row, :, 1], color="#555555")
    for column in range(size):
        axis.plot(prototypes[:, column, 0], prototypes[:, column, 1], color="#555555")
    axis.scatter(
        prototypes[:, :, 0],
        prototypes[:, :, 1],
        marker="s",
        color="black",
        s=35,
        label="prototipos",
    )
    axis.set(
        title="Datos sintéticos y grilla aprendida",
        xlabel="x",
        ylabel="y",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    _save_figure(figure, generated / "grilla_aprendida.svg")
    plt.close(figure)

    figures = (
        (
            _annotated_heatmap(
                plt,
                validation.hit_map,
                "Mapa de hits sintético",
                "Cantidad de muestras",
                "d",
            ),
            "mapa_hits.svg",
        ),
        (
            _annotated_heatmap(
                plt,
                validation.u_matrix,
                "Matriz U sintética",
                "Distancia euclídea media",
                ".2f",
            ),
            "matriz_u.svg",
        ),
    )
    for figure, filename in figures:
        _save_figure(figure, generated / filename)
        plt.close(figure)


def write_report(
    output: Path, run: KohonenRun, validation: SyntheticValidation
) -> None:
    config = run.config
    state = "APROBADA" if validation.passed else "FALLIDA"
    report = f"""# Validación sintética de Kohonen

**Resultado: {state}.**

Se generaron cuatro nubes gaussianas bidimensionales, con 30 muestras por
grupo y centros en los cuatro cuadrantes. Las etiquetas se usan únicamente
para verificar la implementación; no participan del entrenamiento.

## Configuración

- Grilla: `{config.grid_size} x {config.grid_size}`.
- Épocas: `{config.epochs}`.
- Tasa: `{config.learning_rate_start:g}` → `{config.learning_rate_end:g}`.
- Radio: `{config.radius_start:g}` → `{config.radius_end:g}`.
- Inicialización: `{config.initialization}`.
- Semilla de Kohonen: `{config.seed}`.

## Comprobaciones

| Comprobación | Resultado |
| --- | ---: |
| Todas las muestras tienen BMU | {validation.all_samples_assigned} ({int(validation.hit_map.sum())}/{len(validation.values)}) |
| Misma semilla reproduce los pesos | {validation.reproducible} |
| Pesos y matriz U finitos | {bool(np.isfinite(validation.model.weights).all() and np.isfinite(validation.u_matrix).all())} |
| Pureza de las neuronas ocupadas | {validation.purity:.3f} |
| Distancia media dentro del mismo grupo | {validation.mean_within_cluster_grid_distance:.3f} |
| Distancia media entre grupos | {validation.mean_between_cluster_grid_distance:.3f} |
| Máximo error entre un centro y su prototipo más cercano | {validation.maximum_center_error:.3f} |

La pureza sólo se usa en este caso porque conocemos las etiquetas verdaderas.
No será una métrica disponible para seleccionar la red sobre los creadores. La
separación se considera recuperada cuando la pureza es al menos 0,95, las
muestras del mismo grupo quedan más cerca en la grilla que las de grupos
distintos y cada centro tiene un prototipo a distancia máxima 1.

![Datos y grilla aprendida](generated/grilla_aprendida.svg)

![Mapa de hits](generated/mapa_hits.svg)

![Matriz U](generated/matriz_u.svg)

## Conclusión

El caso controlado {'confirma' if validation.passed else 'no confirma'} que la
implementación asigna las muestras de forma reproducible y recupera una
estructura conocida antes de aplicarla al dataset real. Los valores completos
están en `generated/metricas.csv`, `generated/muestras.csv` y
`generated/prototipos.csv`.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/kohonen/synthetic.json"),
        help="Configuración única de validación.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/kohonen-sintetico"),
        help="Directorio del informe de validación.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run = load_run(args.config)
    validation = validate_synthetic(run)
    args.output.mkdir(parents=True, exist_ok=True)
    write_tables(args.output, validation)
    write_plots(args.output, validation)
    write_report(args.output, run, validation)
    if not validation.passed:
        raise RuntimeError("la validación sintética de Kohonen no fue superada")
    print(f"Validación sintética aprobada; informe en {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
