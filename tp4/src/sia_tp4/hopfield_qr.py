"""Carga y representación bipolar de los códigos QR del ejercicio de Hopfield."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hopfield import HopfieldModel, RecallResult, fit_hopfield
from .pca_analysis import _load_pyplot, _save_figure


QR_FILENAMES: Tuple[str, ...] = (
    "beverages_menu_qr.png",
    "dessert_menu_qr.png",
    "food_menu_qr.png",
    "specials_menu_qr.png",
    "wine_menu_qr.png",
)
QR_SHAPE = (37, 37)
NOISE_TYPES = ("uniform_flip", "uniform_erasure", "edge_erasure")
NOISE_LABELS = {
    "uniform_flip": "Flip uniforme",
    "uniform_erasure": "Borrado uniforme",
    "edge_erasure": "Borrado hacia bordes",
}


@dataclass(frozen=True)
class QRPatternSet:
    """Los QR conservando sus formas matricial y vectorial."""

    names: Tuple[str, ...]
    images: np.ndarray
    vectors: np.ndarray


@dataclass(frozen=True)
class QRExperimentDesign:
    """Configuración reproducible del análisis de ruido con los cinco QR."""

    noise_types: Tuple[str, ...]
    noise_levels: Tuple[float, ...]
    seeds: Tuple[int, ...]
    max_iterations: int


def _binary_grayscale(image: np.ndarray, path: Path) -> np.ndarray:
    """Validar una imagen binaria y llevar sus valores al intervalo 0--1."""

    values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"{path.name} debe ser una imagen en escala de grises")
    if values.shape != QR_SHAPE:
        raise ValueError(
            f"{path.name} debe medir {QR_SHAPE[0]}x{QR_SHAPE[1]}, "
            f"pero mide {values.shape}"
        )
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{path.name} debe contener valores numéricos")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contiene valores no finitos")

    unique = np.unique(values)
    if np.isin(unique, (0, 1)).all():
        return values.astype(np.uint8)
    if np.isin(unique, (0, 255)).all():
        return (values // 255).astype(np.uint8)
    raise ValueError(
        f"{path.name} debe ser binaria: sólo se permiten blanco y negro"
    )


def _load_bipolar_image(path: Path) -> np.ndarray:
    """Cargar un QR asignando +1 al negro y -1 al blanco."""

    grayscale = _binary_grayscale(_load_pyplot().imread(path), path)
    return np.where(grayscale == 0, 1, -1).astype(np.int8)


def load_qr_patterns(directory: Path | str) -> QRPatternSet:
    """Cargar los cinco QR del enunciado en un orden fijo y reproducible."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"no existe el directorio de QR: {root}")

    paths = tuple(root / filename for filename in QR_FILENAMES)
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"faltan archivos QR: {missing}")

    images = np.asarray([_load_bipolar_image(path) for path in paths])
    names = tuple(path.stem.removesuffix("_menu_qr") for path in paths)
    vectors = images.reshape(len(images), -1)
    return QRPatternSet(names=names, images=images, vectors=vectors)


def load_design(path: Path | str) -> QRExperimentDesign:
    """Leer y validar el diseño experimental de ruido."""

    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {"noise_types", "noise_levels", "seeds", "max_iterations"}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"campos faltantes: {missing}; desconocidos: {unknown}")

    noise_types = tuple(raw["noise_types"])
    if not noise_types or len(noise_types) != len(set(noise_types)):
        raise ValueError("noise_types debe contener tipos únicos")
    invalid_types = sorted(set(noise_types) - set(NOISE_TYPES))
    if invalid_types:
        raise ValueError(f"tipos de ruido desconocidos: {invalid_types}")

    noise_levels = tuple(float(level) for level in raw["noise_levels"])
    if (
        not noise_levels
        or tuple(sorted(set(noise_levels))) != noise_levels
        or noise_levels[0] != 0.0
        or any(not 0.0 <= level <= 1.0 for level in noise_levels)
    ):
        raise ValueError(
            "noise_levels debe ser creciente, sin repetidos, comenzar en 0 "
            "y quedar entre 0 y 1"
        )

    seeds = tuple(raw["seeds"])
    if len(seeds) < 10 or len(seeds) != len(set(seeds)):
        raise ValueError("se requieren al menos 10 semillas únicas")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("las semillas deben ser enteros")

    maximum = raw["max_iterations"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("max_iterations debe ser un entero positivo")
    return QRExperimentDesign(noise_types, noise_levels, seeds, maximum)


def gram_matrices(patterns: QRPatternSet) -> Tuple[np.ndarray, np.ndarray]:
    """Calcular las matrices de Gram cruda y normalizada por píxel."""

    vectors = patterns.vectors.astype(np.int64)
    gram = vectors @ vectors.T
    normalized = gram / vectors.shape[1]
    return gram, normalized


def pairwise_similarity_rows(patterns: QRPatternSet) -> List[dict]:
    """Describir cada par a partir de su producto interno bipolar."""

    gram, normalized = gram_matrices(patterns)
    pixel_count = patterns.vectors.shape[1]
    rows = []
    for first in range(len(patterns.names)):
        for second in range(first + 1, len(patterns.names)):
            inner_product = int(gram[first, second])
            different_pixels = (pixel_count - inner_product) // 2
            rows.append(
                {
                    "first": patterns.names[first],
                    "second": patterns.names[second],
                    "inner_product": inner_product,
                    "absolute_inner_product": abs(inner_product),
                    "normalized_inner_product": float(normalized[first, second]),
                    "different_pixels": different_pixels,
                    "different_pixel_fraction": different_pixels / pixel_count,
                }
            )
    return rows


def pixel_balance_rows(patterns: QRPatternSet) -> List[dict]:
    """Contar módulos negros y blancos de cada patrón."""

    rows = []
    pixel_count = patterns.vectors.shape[1]
    for name, vector in zip(patterns.names, patterns.vectors):
        black = int(np.count_nonzero(vector == 1))
        rows.append(
            {
                "name": name,
                "black_pixels": black,
                "white_pixels": pixel_count - black,
                "black_fraction": black / pixel_count,
            }
        )
    return rows


def _recall_outcome(
    model: HopfieldModel,
    target_index: int,
    result: RecallResult,
) -> Tuple[str, Optional[int]]:
    """Clasificar el destino de una consulta respecto de los patrones guardados."""

    if result.status == "cycle":
        return "cycle", None
    if result.status == "max_iterations":
        return "max_iterations", None
    matched_index = model.matching_pattern_index(result.final_state)
    if matched_index == target_index:
        return "correct", matched_index
    if matched_index is not None:
        return "other_stored", matched_index
    return "spurious_fixed", None


def run_zero_noise_recall(
    patterns: QRPatternSet,
    max_iterations: int = 100,
) -> Tuple[List[dict], List[dict]]:
    """Consultar cada QR original en una red que almacena los cinco juntos."""

    model = fit_hopfield(patterns.vectors)
    rows = []
    cases = []
    for target_index, (name, target) in enumerate(
        zip(patterns.names, patterns.vectors)
    ):
        result = model.recall(target, max_iterations=max_iterations)
        outcome, matched_index = _recall_outcome(
            model, target_index, result
        )
        final_error = int(np.count_nonzero(result.final_state != target))
        remains_stable = bool(
            result.status == "fixed_point"
            and len(result.states) >= 2
            and np.array_equal(result.states[0], result.states[1])
        )
        row = {
            "target": name,
            "remains_stable": remains_stable,
            "outcome": outcome,
            "status": result.status,
            "matched_pattern": (
                patterns.names[matched_index]
                if matched_index is not None
                else ""
            ),
            "iterations": result.iterations,
            "cycle_length": result.cycle_length or "",
            "final_error_pixels": final_error,
            "final_error_fraction": final_error / len(target),
            "initial_energy": float(result.energies[0]),
            "final_energy": float(result.energies[-1]),
            "energy_change": float(
                result.energies[-1] - result.energies[0]
            ),
        }
        rows.append(row)
        cases.append(
            {
                "target_index": target_index,
                "target": target,
                "row": row,
                "result": result,
            }
        )
    return rows, cases


def energy_trajectory_rows(cases: Sequence[dict]) -> List[dict]:
    """Convertir las trayectorias de energía en una tabla larga."""

    rows = []
    for case in cases:
        target = case["target"]
        name = case["row"]["target"]
        result = case["result"]
        for iteration, (state, energy) in enumerate(
            zip(result.states, result.energies)
        ):
            rows.append(
                {
                    "target": name,
                    "iteration": iteration,
                    "energy": float(energy),
                    "error_pixels": int(np.count_nonzero(state != target)),
                }
            )
    return rows


def edge_sampling_weights(shape: Tuple[int, int] = QR_SHAPE) -> np.ndarray:
    """Dar peso linealmente mayor a los píxeles cercanos al borde."""

    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("shape debe contener dos dimensiones positivas")
    rows, columns = np.indices(shape)
    distance = np.minimum.reduce(
        (rows, columns, shape[0] - 1 - rows, shape[1] - 1 - columns)
    )
    weights = distance.max() + 1 - distance
    flattened = weights.astype(np.float64).reshape(-1)
    return flattened / flattened.sum()


def add_qr_noise(
    pattern: np.ndarray,
    noise_type: str,
    level: float,
    seed: int,
) -> Tuple[np.ndarray, int, int]:
    """Construir una consulta QR deteriorada según una función explícita."""

    values = np.asarray(pattern, dtype=np.int8)
    if values.ndim != 1 or len(values) != QR_SHAPE[0] * QR_SHAPE[1]:
        raise ValueError("pattern debe ser un vector QR de 1369 posiciones")
    if not np.isin(values, (-1, 1)).all():
        raise ValueError("pattern debe ser bipolar")
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"tipo de ruido desconocido: {noise_type}")
    if not np.isfinite(level) or not 0.0 <= level <= 1.0:
        raise ValueError("level debe quedar entre 0 y 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed debe ser un entero")

    selected_count = int(round(level * len(values)))
    noisy = values.copy()
    if selected_count == 0:
        return noisy, 0, 0

    probabilities = (
        edge_sampling_weights()
        if noise_type == "edge_erasure"
        else None
    )
    indices = np.random.default_rng(seed).choice(
        len(values),
        size=selected_count,
        replace=False,
        p=probabilities,
    )
    if noise_type == "uniform_flip":
        noisy[indices] *= -1
    else:
        noisy[indices] = -1
    changed_count = int(np.count_nonzero(noisy != values))
    return noisy, selected_count, changed_count


def run_noise_experiment(
    patterns: QRPatternSet,
    design: QRExperimentDesign,
) -> Tuple[List[dict], List[dict]]:
    """Evaluar las consultas ruidosas sobre una red con los cinco QR."""

    model = fit_hopfield(patterns.vectors)
    baseline_spurious = model.recall(
        patterns.vectors[0], max_iterations=design.max_iterations
    ).final_state
    rows = []
    cases = []
    for noise_type in design.noise_types:
        for target_index, (name, target) in enumerate(
            zip(patterns.names, patterns.vectors)
        ):
            for level in design.noise_levels:
                if level == 0.0:
                    continue
                for seed in design.seeds:
                    query, selected_count, changed_count = add_qr_noise(
                        target, noise_type, level, seed
                    )
                    result = model.recall(
                        query, max_iterations=design.max_iterations
                    )
                    outcome, matched_index = _recall_outcome(
                        model, target_index, result
                    )
                    final_error = int(
                        np.count_nonzero(result.final_state != target)
                    )
                    energy_non_increasing = bool(
                        np.all(np.diff(result.energies) <= 1e-9)
                    )
                    if np.array_equal(result.final_state, baseline_spurious):
                        attractor = "baseline_spurious"
                    elif np.array_equal(
                        result.final_state, -baseline_spurious
                    ):
                        attractor = "inverse_baseline_spurious"
                    elif matched_index is not None:
                        attractor = f"stored_{patterns.names[matched_index]}"
                    elif result.status == "fixed_point":
                        attractor = "other_spurious"
                    else:
                        attractor = result.status
                    row = {
                        "noise_type": noise_type,
                        "target": name,
                        "noise_level": level,
                        "seed": seed,
                        "selected_pixels": selected_count,
                        "changed_pixels": changed_count,
                        "outcome": outcome,
                        "attractor": attractor,
                        "status": result.status,
                        "matched_pattern": (
                            patterns.names[matched_index]
                            if matched_index is not None
                            else ""
                        ),
                        "iterations": result.iterations,
                        "cycle_length": result.cycle_length or "",
                        "final_error_pixels": final_error,
                        "final_error_fraction": final_error / len(target),
                        "initial_energy": float(result.energies[0]),
                        "final_energy": float(result.energies[-1]),
                        "energy_change": float(
                            result.energies[-1] - result.energies[0]
                        ),
                        "energy_non_increasing": energy_non_increasing,
                    }
                    rows.append(row)
                    cases.append(
                        {
                            "target_index": target_index,
                            "target": target,
                            "query": query,
                            "row": row,
                            "result": result,
                        }
                    )
    return rows, cases


def noise_energy_rows(cases: Sequence[dict]) -> List[dict]:
    """Guardar energía y error de cada estado de las corridas con ruido."""

    rows = []
    for case in cases:
        metadata = case["row"]
        for iteration, (state, energy) in enumerate(
            zip(case["result"].states, case["result"].energies)
        ):
            rows.append(
                {
                    "noise_type": metadata["noise_type"],
                    "target": metadata["target"],
                    "noise_level": metadata["noise_level"],
                    "seed": metadata["seed"],
                    "iteration": iteration,
                    "energy": float(energy),
                    "error_pixels": int(
                        np.count_nonzero(state != case["target"])
                    ),
                }
            )
    return rows


def _mean_and_sample_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    deviation = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return float(np.mean(array)), deviation


def _seed_noise_metrics(rows: Sequence[dict]) -> dict:
    count = len(rows)
    return {
        "recovery_rate": sum(row["outcome"] == "correct" for row in rows)
        / count,
        "other_stored_rate": sum(
            row["outcome"] == "other_stored" for row in rows
        )
        / count,
        "spurious_rate": sum(
            row["outcome"] == "spurious_fixed" for row in rows
        )
        / count,
        "baseline_spurious_rate": sum(
            row["attractor"] == "baseline_spurious" for row in rows
        )
        / count,
        "inverse_spurious_rate": sum(
            row["attractor"] == "inverse_baseline_spurious" for row in rows
        )
        / count,
        "other_spurious_rate": sum(
            row["attractor"] == "other_spurious" for row in rows
        )
        / count,
        "cycle_rate": sum(row["outcome"] == "cycle" for row in rows)
        / count,
        "max_iterations_rate": sum(
            row["outcome"] == "max_iterations" for row in rows
        )
        / count,
        "changed_pixels": float(
            np.mean([row["changed_pixels"] for row in rows])
        ),
        "final_error_pixels": float(
            np.mean([row["final_error_pixels"] for row in rows])
        ),
        "iterations": float(np.mean([row["iterations"] for row in rows])),
        "energy_non_increasing_rate": sum(
            row["energy_non_increasing"] for row in rows
        )
        / count,
    }


def summarize_noise_runs(
    rows: Sequence[dict],
    zero_noise_rows: Sequence[dict],
    design: QRExperimentDesign,
) -> List[dict]:
    """Agregar primero por semilla y luego entre al menos diez semillas."""

    metric_names = (
        "recovery_rate",
        "other_stored_rate",
        "spurious_rate",
        "baseline_spurious_rate",
        "inverse_spurious_rate",
        "other_spurious_rate",
        "cycle_rate",
        "max_iterations_rate",
        "changed_pixels",
        "final_error_pixels",
        "iterations",
        "energy_non_increasing_rate",
    )
    baseline = _seed_noise_metrics(
        [
            {
                **row,
                "changed_pixels": 0,
                "energy_non_increasing": True,
                "attractor": "baseline_spurious",
            }
            for row in zero_noise_rows
        ]
    )
    summary = []
    for noise_type in design.noise_types:
        baseline_row = {
            "noise_type": noise_type,
            "noise_level": 0.0,
            "seed_count": 0,
        }
        for metric in metric_names:
            baseline_row[f"{metric}_mean"] = baseline[metric]
            baseline_row[f"{metric}_std"] = 0.0
        summary.append(baseline_row)

        for level in design.noise_levels:
            if level == 0.0:
                continue
            seed_metrics = []
            for seed in design.seeds:
                selected = [
                    row
                    for row in rows
                    if row["noise_type"] == noise_type
                    and row["noise_level"] == level
                    and row["seed"] == seed
                ]
                if len(selected) != 5:
                    raise ValueError(
                        "cada ruido, nivel y semilla debe incluir los cinco QR"
                    )
                seed_metrics.append(_seed_noise_metrics(selected))
            summary_row = {
                "noise_type": noise_type,
                "noise_level": level,
                "seed_count": len(seed_metrics),
            }
            for metric in metric_names:
                mean, deviation = _mean_and_sample_std(
                    [values[metric] for values in seed_metrics]
                )
                summary_row[f"{metric}_mean"] = mean
                summary_row[f"{metric}_std"] = deviation
            summary.append(summary_row)
    return summary


def summarize_noise_by_target(
    rows: Sequence[dict],
    zero_noise_rows: Sequence[dict],
    design: QRExperimentDesign,
) -> List[dict]:
    """Agregar las métricas sin ocultar diferencias entre los cinco QR."""

    metric_names = (
        "recovery_rate",
        "other_stored_rate",
        "spurious_rate",
        "baseline_spurious_rate",
        "inverse_spurious_rate",
        "other_spurious_rate",
        "cycle_rate",
        "max_iterations_rate",
        "changed_pixels",
        "final_error_pixels",
        "iterations",
        "energy_non_increasing_rate",
    )
    summary = []
    for noise_type in design.noise_types:
        for baseline_source in zero_noise_rows:
            baseline = _seed_noise_metrics(
                [
                    {
                        **baseline_source,
                        "changed_pixels": 0,
                        "energy_non_increasing": True,
                        "attractor": "baseline_spurious",
                    }
                ]
            )
            baseline_row = {
                "noise_type": noise_type,
                "target": baseline_source["target"],
                "noise_level": 0.0,
                "seed_count": 0,
            }
            for metric in metric_names:
                baseline_row[f"{metric}_mean"] = baseline[metric]
                baseline_row[f"{metric}_std"] = 0.0
            summary.append(baseline_row)

        for target in sorted({row["target"] for row in rows}):
            for level in design.noise_levels:
                if level == 0.0:
                    continue
                selected = [
                    row
                    for row in rows
                    if row["noise_type"] == noise_type
                    and row["target"] == target
                    and row["noise_level"] == level
                ]
                if len(selected) != len(design.seeds):
                    raise ValueError(
                        "cada ruido, QR y nivel debe incluir todas las semillas"
                    )
                seed_metrics = [_seed_noise_metrics([row]) for row in selected]
                summary_row = {
                    "noise_type": noise_type,
                    "target": target,
                    "noise_level": level,
                    "seed_count": len(selected),
                }
                for metric in metric_names:
                    mean, deviation = _mean_and_sample_std(
                        [values[metric] for values in seed_metrics]
                    )
                    summary_row[f"{metric}_mean"] = mean
                    summary_row[f"{metric}_std"] = deviation
                summary.append(summary_row)
    return sorted(
        summary,
        key=lambda row: (
            design.noise_types.index(row["noise_type"]),
            row["target"],
            row["noise_level"],
        ),
    )


def select_controlled_noise_cases(
    cases: Sequence[dict],
    design: QRExperimentDesign,
    target_name: str,
) -> List[dict]:
    """Elegir una comparación visual sin seleccionar según el resultado."""

    level = design.noise_levels[len(design.noise_levels) // 2]
    if level == 0.0:
        raise ValueError("se necesita al menos un nivel positivo central")
    seed = min(design.seeds)
    selected = []
    for noise_type in design.noise_types:
        match = next(
            case
            for case in cases
            if case["row"]["noise_type"] == noise_type
            and case["row"]["target"] == target_name
            and case["row"]["noise_level"] == level
            and case["row"]["seed"] == seed
        )
        selected.append(match)
    return selected


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_orthogonality_tables(output: Path, patterns: QRPatternSet) -> None:
    """Guardar Gram, similitudes por par y balances de píxeles."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    gram, normalized = gram_matrices(patterns)

    matrix_rows = []
    normalized_rows = []
    for name, raw_row, normalized_row in zip(patterns.names, gram, normalized):
        matrix_rows.append(
            {"name": name, **dict(zip(patterns.names, map(int, raw_row)))}
        )
        normalized_rows.append(
            {
                "name": name,
                **dict(zip(patterns.names, map(float, normalized_row))),
            }
        )
    fields = ["name", *patterns.names]
    _write_csv(generated / "matriz_gram.csv", fields, matrix_rows)
    _write_csv(
        generated / "matriz_gram_normalizada.csv", fields, normalized_rows
    )
    pair_rows = pairwise_similarity_rows(patterns)
    _write_csv(generated / "similitud_pares.csv", list(pair_rows[0]), pair_rows)
    balance_rows = pixel_balance_rows(patterns)
    _write_csv(
        generated / "balance_pixeles.csv", list(balance_rows[0]), balance_rows
    )


def write_zero_noise_tables(
    output: Path,
    rows: Sequence[dict],
    cases: Sequence[dict],
) -> None:
    """Guardar el resumen y la energía completa de las consultas sin ruido."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "recuperacion_sin_ruido.csv", list(rows[0]), rows)
    energy_rows = energy_trajectory_rows(cases)
    _write_csv(
        generated / "energia_sin_ruido.csv",
        list(energy_rows[0]),
        energy_rows,
    )


def write_noise_tables(
    output: Path,
    rows: Sequence[dict],
    cases: Sequence[dict],
    summary: Sequence[dict],
    target_summary: Sequence[dict],
    controlled_cases: Sequence[dict],
) -> None:
    """Guardar corridas, trayectorias, agregados y casos visualizados."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "corridas_ruido.csv", list(rows[0]), rows)
    energy_rows = noise_energy_rows(cases)
    _write_csv(
        generated / "energia_corridas_ruido.csv",
        list(energy_rows[0]),
        energy_rows,
    )
    _write_csv(
        generated / "resumen_ruido.csv", list(summary[0]), summary
    )
    _write_csv(
        generated / "resumen_ruido_por_qr.csv",
        list(target_summary[0]),
        target_summary,
    )
    controlled_rows = [case["row"] for case in controlled_cases]
    _write_csv(
        generated / "casos_ruido_controlado.csv",
        list(controlled_rows[0]),
        controlled_rows,
    )


def write_orthogonality_plot(output: Path, patterns: QRPatternSet) -> None:
    """Graficar la matriz normalizada con sus valores explícitos."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _, normalized = gram_matrices(patterns)
    plt = _load_pyplot()
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized, cmap="RdYlBu_r", vmin=-1.0, vmax=1.0)
    indices = np.arange(len(patterns.names))
    axis.set_xticks(indices, patterns.names, rotation=35, ha="right")
    axis.set_yticks(indices, patterns.names)
    axis.set_title("Productos internos normalizados entre QR")
    for row in indices:
        for column in indices:
            axis.text(
                column,
                row,
                f"{normalized[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Producto interno / 1369")
    figure.tight_layout()
    _save_figure(figure, generated / "matriz_gram_normalizada.svg")
    plt.close(figure)


def write_zero_noise_plots(output: Path, cases: Sequence[dict]) -> None:
    """Comparar objetivos, consultas y destinos, y mostrar sus energías."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()

    figure, axes = plt.subplots(
        len(cases), 4, figsize=(9.4, 2.15 * len(cases)), squeeze=False
    )
    titles = ("Objetivo", "Consulta S(0)", "Estado final", "Diferencias")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row_index, case in enumerate(cases):
        target = case["target"].reshape(QR_SHAPE)
        initial = case["result"].states[0].reshape(QR_SHAPE)
        final = case["result"].final_state.reshape(QR_SHAPE)
        difference = target != final
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].imshow(target, cmap="binary", vmin=-1, vmax=1)
        axes[row_index, 1].imshow(initial, cmap="binary", vmin=-1, vmax=1)
        axes[row_index, 2].imshow(final, cmap="binary", vmin=-1, vmax=1)
        axes[row_index, 3].imshow(
            difference, cmap="Reds", vmin=0, vmax=1
        )
        row = case["row"]
        axes[row_index, 0].set_ylabel(
            f"{row['target']}\n{row['outcome']} · "
            f"{row['final_error_pixels']} errores",
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
            fontsize=8,
        )
    figure.suptitle("Recuperación de los cinco QR sin ruido", y=1.0)
    figure.tight_layout()
    _save_figure(figure, generated / "recuperacion_sin_ruido.svg")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(11, 6.4), squeeze=False)
    for axis, case in zip(axes.reshape(-1), cases):
        energies = case["result"].energies
        iterations = np.arange(len(energies))
        axis.plot(iterations, energies, marker="o", color="#3b82f6")
        axis.set_title(case["row"]["target"])
        axis.set_xlabel("Iteración")
        axis.set_ylabel("Energía")
        axis.set_xticks(iterations)
        axis.margins(x=0.12, y=0.15)
        axis.grid(alpha=0.25)
    for axis in axes.reshape(-1)[len(cases) :]:
        axis.axis("off")
    figure.suptitle("Energía durante la recuperación sin ruido")
    figure.tight_layout()
    _save_figure(figure, generated / "energia_sin_ruido.svg")
    plt.close(figure)


def _bounded_error(
    means: np.ndarray,
    deviations: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    return np.vstack(
        (
            np.minimum(deviations, means - lower),
            np.minimum(deviations, upper - means),
        )
    )


def _noise_summary_by_type(
    summary: Sequence[dict], noise_type: str
) -> List[dict]:
    return sorted(
        (row for row in summary if row["noise_type"] == noise_type),
        key=lambda row: row["noise_level"],
    )


def write_noise_plots(
    output: Path,
    summary: Sequence[dict],
    controlled_cases: Sequence[dict],
) -> None:
    """Graficar resultados agregados y una comparación visual controlada."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    colors = {
        "uniform_flip": "#2563eb",
        "uniform_erasure": "#f59e0b",
        "edge_erasure": "#dc2626",
    }
    line_styles = {
        "uniform_flip": "-",
        "uniform_erasure": "--",
        "edge_erasure": ":",
    }
    markers = {
        "uniform_flip": "o",
        "uniform_erasure": "s",
        "edge_erasure": "^",
    }

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for noise_type in NOISE_TYPES:
        series = _noise_summary_by_type(summary, noise_type)
        if not series:
            continue
        levels = np.asarray([row["noise_level"] * 100 for row in series])
        means = np.asarray(
            [row["recovery_rate_mean"] * 100 for row in series]
        )
        deviations = np.asarray(
            [row["recovery_rate_std"] * 100 for row in series]
        )
        axis.errorbar(
            levels,
            means,
            yerr=_bounded_error(means, deviations, 0.0, 100.0),
            marker=markers[noise_type],
            linestyle=line_styles[noise_type],
            capsize=5,
            linewidth=1.8,
            label=NOISE_LABELS[noise_type],
            color=colors[noise_type],
        )
    axis.set_xlabel("Nivel nominal de ruido (%)")
    axis.set_ylabel("Recuperación correcta (%)")
    axis.set_ylim(-2, 102)
    axis.margins(x=0.04)
    axis.grid(alpha=0.25)
    axis.legend()
    axis.text(
        0.5,
        0.08,
        "Las tres series coinciden en 0 %",
        transform=axis.transAxes,
        ha="center",
        color="#475569",
    )
    figure.tight_layout()
    _save_figure(figure, generated / "recuperacion_vs_ruido.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for noise_type in NOISE_TYPES:
        series = _noise_summary_by_type(summary, noise_type)
        if not series:
            continue
        levels = np.asarray([row["noise_level"] * 100 for row in series])
        means = np.asarray(
            [row["final_error_pixels_mean"] for row in series]
        )
        deviations = np.asarray(
            [row["final_error_pixels_std"] for row in series]
        )
        axis.errorbar(
            levels,
            means,
            yerr=_bounded_error(means, deviations, 0.0, 1369.0),
            marker=markers[noise_type],
            linestyle=line_styles[noise_type],
            capsize=5,
            linewidth=1.8,
            label=NOISE_LABELS[noise_type],
            color=colors[noise_type],
        )
    axis.set_xlabel("Nivel nominal de ruido (%)")
    axis.set_ylabel("Error final medio (píxeles)")
    axis.set_ylim(bottom=0)
    axis.margins(x=0.04)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, generated / "error_final_vs_ruido.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for noise_type in NOISE_TYPES:
        series = _noise_summary_by_type(summary, noise_type)
        if not series:
            continue
        levels = np.asarray([row["noise_level"] * 100 for row in series])
        means = np.asarray([row["changed_pixels_mean"] for row in series])
        deviations = np.asarray(
            [row["changed_pixels_std"] for row in series]
        )
        axis.errorbar(
            levels,
            means,
            yerr=_bounded_error(means, deviations, 0.0, 1369.0),
            marker=markers[noise_type],
            linestyle=line_styles[noise_type],
            capsize=5,
            linewidth=1.8,
            label=NOISE_LABELS[noise_type],
            color=colors[noise_type],
        )
    axis.set_xlabel("Nivel nominal de ruido (%)")
    axis.set_ylabel("Píxeles efectivamente modificados")
    axis.set_ylim(bottom=0)
    axis.margins(x=0.04)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, generated / "error_entrada_vs_ruido.svg")
    plt.close(figure)

    outcomes = (
        ("recovery_rate_mean", "Correcto", "#16a34a"),
        ("other_stored_rate_mean", "Otro almacenado", "#eab308"),
        ("baseline_spurious_rate_mean", "Espurio base", "#f97316"),
        ("inverse_spurious_rate_mean", "Espurio inverso", "#a855f7"),
        ("other_spurious_rate_mean", "Otro espurio", "#fb7185"),
        ("cycle_rate_mean", "Ciclo", "#dc2626"),
        ("max_iterations_rate_mean", "Máx. iteraciones", "#64748b"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.1), sharey=True)
    for axis, noise_type in zip(axes, NOISE_TYPES):
        series = _noise_summary_by_type(summary, noise_type)
        levels = np.asarray([row["noise_level"] * 100 for row in series])
        bottom = np.zeros(len(series))
        for field, label, color in outcomes:
            values = np.asarray([row[field] * 100 for row in series])
            axis.bar(
                levels,
                values,
                width=3.4,
                bottom=bottom,
                label=label,
                color=color,
            )
            bottom += values
        axis.set_title(NOISE_LABELS[noise_type])
        axis.set_xlabel("Ruido nominal (%)")
        axis.set_xlim(-3, 53)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Proporción de resultados (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.86))
    _save_figure(figure, generated / "resultados_vs_ruido.svg")
    plt.close(figure)

    figure, axes = plt.subplots(
        len(controlled_cases), 4, figsize=(9.4, 6.8), squeeze=False
    )
    titles = ("Objetivo", "Consulta", "Daño de entrada", "Error final")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)
    for row_index, case in enumerate(controlled_cases):
        target = case["target"].reshape(QR_SHAPE)
        query = case["query"].reshape(QR_SHAPE)
        final = case["result"].final_state.reshape(QR_SHAPE)
        panels = (
            (target, "binary", -1, 1),
            (query, "binary", -1, 1),
            (target != query, "Reds", 0, 1),
            (target != final, "Reds", 0, 1),
        )
        for axis, (values, cmap, lower, upper) in zip(
            axes[row_index], panels
        ):
            axis.imshow(values, cmap=cmap, vmin=lower, vmax=upper)
            axis.set_xticks([])
            axis.set_yticks([])
        row = case["row"]
        axes[row_index, 0].set_ylabel(
            f"{NOISE_LABELS[row['noise_type']]}\n"
            f"{row['noise_level'] * 100:g} % · semilla {row['seed']}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
            fontsize=8,
        )
    figure.suptitle("Comparación controlada de las funciones de ruido", y=1.0)
    figure.tight_layout()
    _save_figure(figure, generated / "consultas_ruido_controlado.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), squeeze=False)
    for axis, case in zip(axes.reshape(-1), controlled_cases):
        energies = case["result"].energies
        iterations = np.arange(len(energies))
        axis.plot(iterations, energies, marker="o", color="#3b82f6")
        axis.set_title(NOISE_LABELS[case["row"]["noise_type"]])
        axis.set_xlabel("Iteración")
        axis.set_ylabel("Energía")
        axis.set_xticks(iterations)
        axis.margins(x=0.12, y=0.15)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, generated / "energia_ruido_controlado.svg")
    plt.close(figure)


def write_report(
    output: Path,
    patterns: QRPatternSet,
    zero_noise_rows: Sequence[dict],
    zero_noise_cases: Sequence[dict],
    design: QRExperimentDesign,
    noise_rows: Sequence[dict],
    noise_summary: Sequence[dict],
    controlled_cases: Sequence[dict],
) -> None:
    """Documentar los resultados descriptivos del análisis."""

    pairs = pairwise_similarity_rows(patterns)
    balance = pixel_balance_rows(patterns)
    similarities = np.asarray(
        [row["normalized_inner_product"] for row in pairs]
    )
    absolute_similarities = np.abs(similarities)
    pattern_size = patterns.vectors.shape[1]
    theoretical_capacity_reference = round(0.15 * pattern_size)
    closest = max(pairs, key=lambda row: row["absolute_inner_product"])
    minimum = min(row["absolute_inner_product"] for row in pairs)
    least_similar = [
        row for row in pairs if row["absolute_inner_product"] == minimum
    ]
    least_similar_names = ", ".join(
        f"**{row['first']}–{row['second']}**" for row in least_similar
    )
    unique_final_states = {
        case["result"].final_state.tobytes() for case in zero_noise_cases
    }
    correct_count = sum(
        row["outcome"] == "correct" for row in zero_noise_rows
    )
    stable_count = sum(row["remains_stable"] for row in zero_noise_rows)
    transitions = ", ".join(
        f"**{row['target']}→{row['matched_pattern'] or 'espurio'}**"
        for row in zero_noise_rows
    )
    comparison_lines = []
    for noise_type in design.noise_types:
        series = _noise_summary_by_type(noise_summary, noise_type)
        last = series[-1]
        positive = [row for row in series if row["noise_level"] > 0]
        best = max(positive, key=lambda row: row["recovery_rate_mean"])
        recovery_sentence = (
            "No hubo recuperaciones correctas en ningún nivel positivo."
            if best["recovery_rate_mean"] == 0.0
            else (
                f"Su mayor recuperación media fue "
                f"{best['recovery_rate_mean'] * 100:.1f} % a "
                f"{best['noise_level'] * 100:g} %."
            )
        )
        comparison_lines.append(
            f"- **{NOISE_LABELS[noise_type]}**: a "
            f"{last['noise_level'] * 100:g} % nominal modifica en promedio "
            f"{last['changed_pixels_mean']:.1f} píxeles; recuperación "
            f"{last['recovery_rate_mean'] * 100:.1f} %, error final "
            f"{last['final_error_pixels_mean']:.1f} píxeles. "
            f"{recovery_sentence}"
        )
    energy_violations = sum(
        not row["energy_non_increasing"] for row in noise_rows
    )
    baseline_attractor_count = sum(
        row["attractor"] == "baseline_spurious" for row in noise_rows
    )
    inverse_attractor_count = sum(
        row["attractor"] == "inverse_baseline_spurious"
        for row in noise_rows
    )
    other_attractor_count = len(noise_rows) - (
        baseline_attractor_count + inverse_attractor_count
    )
    inverse_groups: Dict[Tuple[str, float], int] = {}
    for row in noise_rows:
        if row["attractor"] == "inverse_baseline_spurious":
            key = (row["noise_type"], row["noise_level"])
            inverse_groups[key] = inverse_groups.get(key, 0) + 1
    inverse_description = ", ".join(
        f"{NOISE_LABELS[noise_type]} al {level * 100:g} % "
        f"({count} de "
        f"{sum(row['noise_type'] == noise_type and row['noise_level'] == level for row in noise_rows)} corridas)"
        for (noise_type, level), count in sorted(inverse_groups.items())
    )
    controlled = controlled_cases[0]["row"]
    report = f"""# Análisis inicial de los QR para Hopfield

## Representación

Los cinco códigos QR se cargan como matrices binarias de 37×37 y se convierten
a representación bipolar: negro `+1` y blanco `-1`. Cada patrón se aplana en
un vector de 1369 componentes. La carga es determinista y no usa semillas ni
hiperparámetros.

## Matriz de Gram

Se calcula `G = XXᵀ` usando enteros de 64 bits y se normaliza dividiendo por
1369. La diagonal normalizada vale 1 porque cada patrón se compara consigo
mismo. Para patrones bipolares, un valor fuera de la diagonal cercano a cero
indicaría mayor cercanía a la ortogonalidad.

- producto interno normalizado medio fuera de la diagonal: **{similarities.mean():.3f}**;
- mínimo absoluto: **{least_similar[0]['normalized_inner_product']:.3f}** en
  {least_similar_names};
- máximo absoluto: **{closest['normalized_inner_product']:.3f}** entre
  **{closest['first']}** y **{closest['second']}**.

![Matriz de Gram normalizada](generated/matriz_gram_normalizada.svg)

Todos los productos internos son positivos y altos. Por lo tanto, estos QR no
son cercanos a ser ortogonales: comparten muchos más píxeles con el mismo signo
que con signo opuesto. Esta observación anticipa interferencia al almacenar los
cinco, pero por sí sola no demuestra el comportamiento dinámico; la prueba de
recuperación que aparece debajo comprueba esa hipótesis.

## Balance de píxeles

{chr(10).join(f"- **{row['name']}**: {row['black_pixels']} negros, {row['white_pixels']} blancos ({row['black_fraction'] * 100:.1f} % negros)." for row in balance)}

El balance es cercano a 50 % en los cinco casos. Por eso, los productos
internos altos no se explican simplemente por una mayoría global de blanco o
negro, sino por posiciones coincidentes entre los QR, incluidas sus estructuras
compartidas. No se modifica ni se centra la entrada: se conserva la
codificación bipolar pedida para Hopfield.

## Recuperación sin ruido

Se entrenó una única red mediante la regla de Hebb almacenando simultáneamente
los cinco vectores. Luego cada QR original se usó como consulta, sin modificar
ningún píxel. El límite de {design.max_iterations} iteraciones funciona sólo como
resguardo y no fue alcanzado. Esta prueba tampoco usa semillas.

- patrones que permanecieron estables: **{stable_count}/5**;
- recuperaciones correctas: **{correct_count}/5**;
- estados finales distintos encontrados: **{len(unique_final_states)}**;
- destinos: {transitions}.

La primera actualización lleva cada consulta al estado espurio y la segunda
confirma que ese estado no cambia; por eso todas las filas registran dos
iteraciones.

{chr(10).join(f"- **{row['target']}**: `{row['outcome']}`, {row['iterations']} iteraciones, {row['final_error_pixels']} píxeles de error, energía {row['initial_energy']:.1f} → {row['final_energy']:.1f}." for row in zero_noise_rows)}

En los cinco recorridos la energía disminuye durante la primera actualización
y permanece constante al confirmar el punto fijo. Esta observación describe
estas trayectorias concretas; no se usa para cambiar la regla de actualización.

![Recuperación sin ruido](generated/recuperacion_sin_ruido.svg)

![Energía sin ruido](generated/energia_sin_ruido.svg)

El chequeo reproducible confirma que ninguno de los QR originales es un punto
fijo cuando se almacenan los cinco juntos. Todos convergen al mismo estado
fijo espurio. Por lo tanto, un barrido de ruido sobre esta red no puede
interpretarse aisladamente como robustez frente a perturbaciones: ya existe
interferencia con 0 % de ruido. El barrido siguiente conserva igualmente los
cinco patrones, como exige la consigna, y esta línea de base se mantiene visible
al interpretar sus resultados.

## Consultas deterioradas

Se comparan tres funciones sugeridas en clase sobre los mismos niveles y
semillas:

1. **Flip uniforme:** se eligen exactamente `round(nivel × 1369)` posiciones
   sin reemplazo y se invierte su signo.
2. **Borrado uniforme:** se elige la misma cantidad de posiciones de manera
   uniforme y se las lleva a blanco (`-1`). Un píxel ya blanco no cambia.
3. **Borrado hacia bordes:** también lleva las posiciones elegidas a blanco,
   pero su peso de selección decrece linealmente con la distancia al borde. En
   una imagen 37×37 el peso sin normalizar es `19 - distancia_al_borde`.

El diseño usa niveles de {', '.join(f'{level * 100:g} %' for level in design.noise_levels)},
semillas {', '.join(map(str, design.seeds))} y un máximo de
{design.max_iterations} iteraciones como resguardo. El 0 % corresponde a las
cinco consultas deterministas ya analizadas; para cada nivel positivo se
evalúan los cinco QR con {len(design.seeds)} semillas. En total se ejecutan
{len(noise_rows)} consultas aleatorias más las cinco consultas base.

El nivel es **nominal**: indica cuántas posiciones se seleccionan. En el flip
todas cambian; en los borrados sólo cambian efectivamente las que eran negras.
Por eso también se informa el error real de entrada y no se supone que tres
ruidos con el mismo porcentaje nominal tengan igual cantidad de alteraciones.

Para las curvas se calcula primero cada métrica sobre los cinco QR de una
semilla y después se muestran media y desvío estándar entre semillas. Las
barras están acotadas al rango válido de cada métrica. No se conectan semillas
como si fueran una secuencia temporal y no se elige un nivel de ruido óptimo.
Cuando el desvío es cero, la barra coincide con el marcador y no tiene longitud
visible; esto ocurre en toda la curva de recuperación.

{chr(10).join(comparison_lines)}

En **{baseline_attractor_count}** corridas el destino fue el mismo espurio de
0 %; en **{inverse_attractor_count}** fue su inverso y en
**{other_attractor_count}** apareció otro destino. El inverso apareció en:
**{inverse_description or 'ninguna condición'}**. En las demás condiciones se
conserva el espurio base. Esto explica cualquier salto del error final sin
confundirlo con una recuperación.

![Recuperación frente al ruido](generated/recuperacion_vs_ruido.svg)

![Error final frente al ruido](generated/error_final_vs_ruido.svg)

![Daño efectivo de entrada](generated/error_entrada_vs_ruido.svg)

![Tipos de resultado](generated/resultados_vs_ruido.svg)

Las {len(noise_rows)} corridas presentaron **{energy_violations}** trayectorias
con algún aumento de energía. Las trayectorias completas quedan en CSV para no
seleccionar conclusiones a partir de un único ejemplo.

## Comparación visual controlada

Para mostrar cómo actúa cada función sin elegir casos por su resultado, se usa
la primera etiqueta en el orden del dataset (**{controlled['target']}**), el
nivel positivo central configurado (**{controlled['noise_level'] * 100:g} %**)
y la menor semilla (**{controlled['seed']}**). La selección se fija antes de
mirar el estado final.

![Consultas con ruido controlado](generated/consultas_ruido_controlado.svg)

![Energía de los casos controlados](generated/energia_ruido_controlado.svg)

Como la recuperación base es 0/5, una eventual mejora al agregar ruido no
significa que la red sea robusta: significa que alguna perturbación desplazó
la consulta desde el atractor espurio hacia otro atractor. El análisis de
capacidad mediante subconjuntos se reserva para la sección de limitaciones,
después de completar el caso obligatorio con los cinco QR.

## Limitaciones observadas e hipótesis de capacidad

### Qué muestran los resultados actuales

La red se entrenó con los cinco QR simultáneamente, tal como pide la consigna.
Sin embargo, ninguno de los cinco patrones originales permanece estable al ser
usado como consulta sin ruido: todos convergen al mismo punto fijo espurio.
Además, ninguna de las {len(noise_rows)} consultas deterioradas recupera su QR
objetivo. La mayor parte converge al mismo espurio encontrado sin ruido y, en
los casos restantes, a su inverso.

Este comportamiento es coherente con la interferencia anticipada por la matriz
de Gram. Los productos internos normalizados fuera de la diagonal están entre
{absolute_similarities.min():.3f} y {absolute_similarities.max():.3f}, con
media {absolute_similarities.mean():.3f}: los QR comparten una fracción
importante de sus píxeles y, por lo tanto, no constituyen patrones cercanos a
la ortogonalidad. Al sumar sus productos externos, la regla de Hebb introduce
términos cruzados que modifican los mínimos de energía de la red.

Que la energía no aumente durante las actualizaciones confirma que la dinámica
implementada desciende —o se mantiene— sobre la función de energía. No
garantiza, en cambio, que el mínimo alcanzado sea una memoria deseada. En este
caso, el punto fijo espurio es un atractor válido de la dinámica, aunque no
coincida con ninguno de los cinco QR almacenados.

### Capacidad teórica frente a capacidad efectiva

En clase se presentó como referencia una capacidad del orden de 0.15 N
patrones para una red de N neuronas. Para estos QR, N = {pattern_size}, lo que
daría aproximadamente {theoretical_capacity_reference} patrones bajo
condiciones favorables. Ese valor es una guía teórica, no una garantía
independiente de los datos: supone patrones suficientemente separados y no
contempla por sí solo la fuerte correlación observada aquí.

Por eso, que la red falle con solo cinco patrones no contradice esa referencia.
Muestra que la **capacidad efectiva** para este conjunto particular de QR es
menor que cinco cuando se usa la regla de Hebb implementada. Con los resultados
actuales todavía no puede determinarse cuántos QR sí puede almacenar ni si el
resultado depende principalmente de la cantidad o de la combinación elegida.

### Contraste de capacidad sin ruido

El [barrido exhaustivo de capacidad](../hopfield-qr-capacidad/README.md)
entrena una red para cada uno de los 31 subconjuntos no vacíos de los cinco QR.
Los cinco patrones individuales y los diez pares son completamente estables;
en cambio, ninguna combinación de tres, cuatro o cinco conserva uno de sus
originales como punto fijo. La capacidad efectiva observada es entonces de dos
QR para esta familia, codificación, regla de aprendizaje y dinámica, no una
capacidad universal de una red de 1369 neuronas.

El par más correlacionado también es estable, por lo que el peor producto
interno aislado no explica el corte. Los 15 subconjuntos válidos se evalúan en
el [barrido de robustez](../hopfield-qr-robustez/README.md), sin confundir el
efecto del ruido con una falla que ya existe en la memoria sin perturbar.

## Archivos generados

- `generated/matriz_gram.csv`: productos internos crudos;
- `generated/matriz_gram_normalizada.csv`: productos internos divididos por 1369;
- `generated/similitud_pares.csv`: los diez pares y sus píxeles diferentes;
- `generated/balance_pixeles.csv`: cantidad de módulos negros y blancos;
- `generated/recuperacion_sin_ruido.csv`: resultado de cada consulta original;
- `generated/energia_sin_ruido.csv`: energía y error en cada iteración;
- `generated/corridas_ruido.csv`: una fila por consulta deteriorada;
- `generated/energia_corridas_ruido.csv`: energía de todas las iteraciones;
- `generated/resumen_ruido.csv`: medias y desvíos entre semillas;
- `generated/resumen_ruido_por_qr.csv`: agregados sin ocultar diferencias entre QR;
- `generated/casos_ruido_controlado.csv`: casos de la comparación visual;
- `generated/matriz_gram_normalizada.svg`: visualización del Gram;
- `generated/recuperacion_sin_ruido.svg`: objetivos, consultas y estados finales;
- `generated/energia_sin_ruido.svg`: trayectorias de energía;
- cuatro gráficos agregados para comparar ruido;
- `generated/consultas_ruido_controlado.svg` y
  `generated/energia_ruido_controlado.svg`: casos fijados sin selección por
  resultado.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datos/hopfield"),
        help="Directorio que contiene los cinco QR.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hopfield/qr.json"),
        help="Configuración reproducible del experimento.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/hopfield-qr"),
        help="Directorio del informe reproducible.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    design = load_design(args.config)
    patterns = load_qr_patterns(args.data)
    zero_noise_rows, zero_noise_cases = run_zero_noise_recall(
        patterns, max_iterations=design.max_iterations
    )
    noise_rows, noise_cases = run_noise_experiment(patterns, design)
    noise_summary = summarize_noise_runs(noise_rows, zero_noise_rows, design)
    target_noise_summary = summarize_noise_by_target(
        noise_rows, zero_noise_rows, design
    )
    controlled_cases = select_controlled_noise_cases(
        noise_cases, design, patterns.names[0]
    )
    write_orthogonality_tables(args.output, patterns)
    write_zero_noise_tables(args.output, zero_noise_rows, zero_noise_cases)
    write_noise_tables(
        args.output,
        noise_rows,
        noise_cases,
        noise_summary,
        target_noise_summary,
        controlled_cases,
    )
    write_orthogonality_plot(args.output, patterns)
    write_zero_noise_plots(args.output, zero_noise_cases)
    write_noise_plots(args.output, noise_summary, controlled_cases)
    write_report(
        args.output,
        patterns,
        zero_noise_rows,
        zero_noise_cases,
        design,
        noise_rows,
        noise_summary,
        controlled_cases,
    )
    similarities = [
        row["normalized_inner_product"]
        for row in pairwise_similarity_rows(patterns)
    ]
    print(
        f"Informe generado en {args.output}; "
        f"similitud media={np.mean(similarities):.3f}; "
        f"máxima={np.max(np.abs(similarities)):.3f}; "
        f"recuperaciones sin ruido="
        f"{sum(row['outcome'] == 'correct' for row in zero_noise_rows)}/5; "
        f"consultas con ruido={len(noise_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
