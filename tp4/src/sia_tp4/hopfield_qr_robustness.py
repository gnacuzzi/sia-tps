"""Robustez frente a ruido de los subconjuntos estables de uno y dos QR."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hopfield import HopfieldModel, fit_hopfield
from .hopfield_qr import (
    NOISE_LABELS,
    QRExperimentDesign,
    QRPatternSet,
    add_qr_noise,
    load_design,
    load_qr_patterns,
    seeds_for_noise_level,
)
from .pca_analysis import _load_pyplot, _save_figure


def robustness_subsets(pattern_count: int) -> List[Tuple[int, ...]]:
    """Enumerar los individuales y pares, válidos en el barrido sin ruido."""

    if pattern_count < 2:
        raise ValueError("se necesitan al menos dos patrones")
    return [
        subset
        for size in (1, 2)
        for subset in combinations(range(pattern_count), size)
    ]


def _classify_result(
    model: HopfieldModel,
    target_index: int,
    result,
) -> Tuple[str, Optional[int]]:
    if result.status == "cycle":
        return "cycle", None
    if result.status == "max_iterations":
        return "max_iterations", None
    matched = model.matching_pattern_index(result.final_state)
    if matched == target_index:
        return "correct", matched
    if matched is not None:
        return "other_stored", matched
    if np.array_equal(result.final_state, -model.stored_patterns[target_index]):
        return "inverse_target", None
    return "spurious_fixed", None


def _pair_similarity(vectors: np.ndarray) -> float | str:
    if len(vectors) == 1:
        return ""
    product = int(
        vectors[0].astype(np.int64) @ vectors[1].astype(np.int64)
    )
    return abs(product) / vectors.shape[1]


def _baseline_row(
    subset_id: str,
    names: Tuple[str, ...],
    pair_similarity: float | str,
    model: HopfieldModel,
    target_index: int,
    target: np.ndarray,
    max_iterations: int,
) -> dict:
    result = model.recall(target, max_iterations=max_iterations)
    outcome, matched = _classify_result(model, target_index, result)
    remains_stable = bool(
        result.status == "fixed_point"
        and len(result.states) >= 2
        and np.array_equal(result.states[0], result.states[1])
    )
    return {
        "subset_id": subset_id,
        "storage_size": len(names),
        "stored_patterns": "+".join(names),
        "pair_similarity": pair_similarity,
        "target": names[target_index],
        "remains_stable": remains_stable,
        "outcome": outcome,
        "status": result.status,
        "matched_pattern": names[matched] if matched is not None else "",
        "iterations": result.iterations,
        "final_error_pixels": int(
            np.count_nonzero(result.final_state != target)
        ),
        "initial_energy": float(result.energies[0]),
        "final_energy": float(result.energies[-1]),
        "energy_non_increasing": bool(
            np.all(np.diff(result.energies) <= 1e-9)
        ),
    }


def run_robustness_sweep(
    patterns: QRPatternSet,
    design: QRExperimentDesign,
) -> Tuple[List[dict], List[dict]]:
    """Evaluar ruido pareado en los 5 individuales y las 10 duplas."""

    subsets = robustness_subsets(len(patterns.names))
    baseline_rows: List[dict] = []
    noise_rows: List[dict] = []
    query_cache: Dict[Tuple[int, str, float, int], Tuple[np.ndarray, int, int]] = {}

    for subset_number, indices in enumerate(subsets, start=1):
        subset_id = f"S{subset_number:02d}"
        names = tuple(patterns.names[index] for index in indices)
        stored = patterns.vectors[list(indices)]
        similarity = _pair_similarity(stored)
        model = fit_hopfield(stored)
        for local_index, global_index in enumerate(indices):
            target = patterns.vectors[global_index]
            baseline = _baseline_row(
                subset_id,
                names,
                similarity,
                model,
                local_index,
                target,
                design.max_iterations,
            )
            if not baseline["remains_stable"]:
                raise ValueError(
                    f"{subset_id} no es estable sin ruido; no corresponde "
                    "medir robustez"
                )
            baseline_rows.append(baseline)

            for noise_type in design.noise_types:
                for level in design.noise_levels:
                    if level == 0.0:
                        continue
                    for seed in seeds_for_noise_level(design, level):
                        key = (global_index, noise_type, level, seed)
                        if key not in query_cache:
                            query_cache[key] = add_qr_noise(
                                target, noise_type, level, seed
                            )
                        query, selected_count, changed_count = query_cache[key]
                        result = model.recall(
                            query, max_iterations=design.max_iterations
                        )
                        outcome, matched = _classify_result(
                            model, local_index, result
                        )
                        final_error = int(
                            np.count_nonzero(result.final_state != target)
                        )
                        noise_rows.append(
                            {
                                "subset_id": subset_id,
                                "storage_size": len(indices),
                                "stored_patterns": "+".join(names),
                                "pair_similarity": similarity,
                                "target": patterns.names[global_index],
                                "noise_type": noise_type,
                                "noise_level": level,
                                "seed": seed,
                                "selected_pixels": selected_count,
                                "changed_pixels": changed_count,
                                "outcome": outcome,
                                "status": result.status,
                                "matched_pattern": (
                                    names[matched] if matched is not None else ""
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
                                "energy_non_increasing": bool(
                                    np.all(np.diff(result.energies) <= 1e-9)
                                ),
                            }
                        )
    return baseline_rows, noise_rows


def _metrics(rows: Sequence[dict]) -> dict:
    count = len(rows)
    return {
        "recovery_rate": sum(row["outcome"] == "correct" for row in rows)
        / count,
        "other_stored_rate": sum(
            row["outcome"] == "other_stored" for row in rows
        )
        / count,
        "inverse_target_rate": sum(
            row["outcome"] == "inverse_target" for row in rows
        )
        / count,
        "spurious_rate": sum(
            row["outcome"] == "spurious_fixed" for row in rows
        )
        / count,
        "cycle_rate": sum(row["outcome"] == "cycle" for row in rows) / count,
        "max_iterations_rate": sum(
            row["outcome"] == "max_iterations" for row in rows
        )
        / count,
        "changed_pixels": float(
            np.mean([row.get("changed_pixels", 0) for row in rows])
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


def _mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    deviation = 0.0 if len(array) == 1 else float(np.std(array, ddof=1))
    return float(np.mean(array)), deviation


METRICS = (
    "recovery_rate",
    "other_stored_rate",
    "inverse_target_rate",
    "spurious_rate",
    "cycle_rate",
    "max_iterations_rate",
    "changed_pixels",
    "final_error_pixels",
    "iterations",
    "energy_non_increasing_rate",
)


def _summary_row(prefix: dict, per_seed: Sequence[dict]) -> dict:
    row = {**prefix, "seed_count": len(per_seed)}
    for metric in METRICS:
        mean, deviation = _mean_and_std(
            [values[metric] for values in per_seed]
        )
        row[f"{metric}_mean"] = mean
        row[f"{metric}_std"] = deviation
    return row


def _baseline_summary(prefix: dict, rows: Sequence[dict]) -> dict:
    values = _metrics(rows)
    result = {**prefix, "seed_count": 0}
    for metric in METRICS:
        result[f"{metric}_mean"] = values[metric]
        result[f"{metric}_std"] = 0.0
    return result


def summarize_by_storage_size(
    baseline_rows: Sequence[dict],
    noise_rows: Sequence[dict],
    design: QRExperimentDesign,
) -> List[dict]:
    """Comparar uno y dos patrones, agregando primero dentro de cada semilla."""

    summary = []
    for noise_type in design.noise_types:
        for storage_size in (1, 2):
            baseline = [
                row
                for row in baseline_rows
                if row["storage_size"] == storage_size
            ]
            summary.append(
                _baseline_summary(
                    {
                        "noise_type": noise_type,
                        "storage_size": storage_size,
                        "noise_level": 0.0,
                    },
                    baseline,
                )
            )
            for level in design.noise_levels:
                if level == 0.0:
                    continue
                per_seed = []
                for seed in seeds_for_noise_level(design, level):
                    selected = [
                        row
                        for row in noise_rows
                        if row["noise_type"] == noise_type
                        and row["storage_size"] == storage_size
                        and row["noise_level"] == level
                        and row["seed"] == seed
                    ]
                    expected = len(baseline)
                    if len(selected) != expected:
                        raise ValueError(
                            "faltan consultas para una condición y semilla"
                        )
                    per_seed.append(_metrics(selected))
                summary.append(
                    _summary_row(
                        {
                            "noise_type": noise_type,
                            "storage_size": storage_size,
                            "noise_level": level,
                        },
                        per_seed,
                    )
                )
    return summary


def summarize_by_subset(
    baseline_rows: Sequence[dict],
    noise_rows: Sequence[dict],
    design: QRExperimentDesign,
) -> List[dict]:
    """Conservar diferencias entre los quince subconjuntos válidos."""

    summary = []
    subset_ids = sorted({row["subset_id"] for row in baseline_rows})
    for subset_id in subset_ids:
        baseline = [row for row in baseline_rows if row["subset_id"] == subset_id]
        metadata = baseline[0]
        for noise_type in design.noise_types:
            prefix = {
                "subset_id": subset_id,
                "storage_size": metadata["storage_size"],
                "stored_patterns": metadata["stored_patterns"],
                "pair_similarity": metadata["pair_similarity"],
                "noise_type": noise_type,
            }
            summary.append(
                _baseline_summary(
                    {**prefix, "noise_level": 0.0}, baseline
                )
            )
            for level in design.noise_levels:
                if level == 0.0:
                    continue
                per_seed = []
                for seed in seeds_for_noise_level(design, level):
                    selected = [
                        row
                        for row in noise_rows
                        if row["subset_id"] == subset_id
                        and row["noise_type"] == noise_type
                        and row["noise_level"] == level
                        and row["seed"] == seed
                    ]
                    if len(selected) != metadata["storage_size"]:
                        raise ValueError(
                            "faltan objetivos de un subconjunto y semilla"
                        )
                    per_seed.append(_metrics(selected))
                summary.append(
                    _summary_row(
                        {**prefix, "noise_level": level}, per_seed
                    )
                )
    return summary


def summarize_pair_robustness(subset_summary: Sequence[dict]) -> List[dict]:
    """Resumir cada dupla sin seleccionar un único nivel de ruido."""

    rows = []
    pair_ids = sorted(
        {
            row["subset_id"]
            for row in subset_summary
            if row["storage_size"] == 2
        }
    )
    noise_types = tuple(
        dict.fromkeys(row["noise_type"] for row in subset_summary)
    )
    for subset_id in pair_ids:
        for noise_type in noise_types:
            selected = sorted(
                (
                    row
                    for row in subset_summary
                    if row["subset_id"] == subset_id
                    and row["noise_type"] == noise_type
                ),
                key=lambda row: row["noise_level"],
            )
            positive = [row for row in selected if row["noise_level"] > 0]
            below_half = [
                row
                for row in positive
                if row["recovery_rate_mean"] < 0.5
            ]
            levels = {row["noise_level"]: row for row in selected}
            def recovery_at(level: float) -> float | str:
                return (
                    levels[level]["recovery_rate_mean"]
                    if level in levels
                    else ""
                )

            rows.append(
                {
                    "subset_id": subset_id,
                    "stored_patterns": selected[0]["stored_patterns"],
                    "pair_similarity": selected[0]["pair_similarity"],
                    "noise_type": noise_type,
                    "mean_positive_recovery": float(
                        np.mean(
                            [row["recovery_rate_mean"] for row in positive]
                        )
                    ),
                    "recovery_at_10_percent": recovery_at(0.1),
                    "recovery_at_30_percent": recovery_at(0.3),
                    "recovery_at_50_percent": recovery_at(0.5),
                    "recovery_at_75_percent": recovery_at(0.75),
                    "recovery_at_100_percent": recovery_at(1.0),
                    "first_level_below_50_percent": (
                        below_half[0]["noise_level"] if below_half else ""
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    baseline_rows: Sequence[dict],
    noise_rows: Sequence[dict],
    storage_summary: Sequence[dict],
    subset_summary: Sequence[dict],
    pair_summary: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    _write_csv(generated / "consultas_sin_ruido.csv", baseline_rows)
    _write_csv(generated / "corridas_ruido.csv", noise_rows)
    _write_csv(generated / "resumen_por_cantidad.csv", storage_summary)
    _write_csv(generated / "resumen_por_subconjunto.csv", subset_summary)
    _write_csv(generated / "resumen_robustez_duplas.csv", pair_summary)


def _series(
    rows: Sequence[dict], noise_type: str, storage_size: int
) -> List[dict]:
    return sorted(
        (
            row
            for row in rows
            if row["noise_type"] == noise_type
            and row["storage_size"] == storage_size
        ),
        key=lambda row: row["noise_level"],
    )


def _bounded_error(means: np.ndarray, deviations: np.ndarray) -> np.ndarray:
    return np.vstack(
        (
            np.minimum(deviations, means),
            np.minimum(deviations, 100.0 - means),
        )
    )


def write_plots(
    output: Path,
    design: QRExperimentDesign,
    storage_summary: Sequence[dict],
    subset_summary: Sequence[dict],
    pair_summary: Sequence[dict],
) -> None:
    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    colors = {1: "#2563eb", 2: "#dc2626"}
    markers = {1: "o", 2: "s"}

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for axis, noise_type in zip(axes, design.noise_types):
        for storage_size in (1, 2):
            rows = _series(storage_summary, noise_type, storage_size)
            levels = np.asarray([row["noise_level"] * 100 for row in rows])
            means = np.asarray(
                [row["recovery_rate_mean"] * 100 for row in rows]
            )
            deviations = np.asarray(
                [row["recovery_rate_std"] * 100 for row in rows]
            )
            axis.errorbar(
                levels,
                means,
                yerr=_bounded_error(means, deviations),
                color=colors[storage_size],
                marker=markers[storage_size],
                linewidth=1.8,
                capsize=4,
                label=f"{storage_size} QR",
            )
        axis.set_title(NOISE_LABELS[noise_type])
        axis.set_xlabel("Ruido nominal (%)")
        axis.set_xlim(-2, 102)
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Recuperación correcta (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    _save_figure(figure, generated / "recuperacion_uno_vs_dos.svg")
    plt.close(figure)

    pairs = sorted(
        {
            (row["subset_id"], row["stored_patterns"])
            for row in subset_summary
            if row["storage_size"] == 2
        }
    )
    levels = list(design.noise_levels)
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 6.0), sharey=True)
    image = None
    for axis, noise_type in zip(axes, design.noise_types):
        matrix = np.asarray(
            [
                [
                    next(
                        row["recovery_rate_mean"]
                        for row in subset_summary
                        if row["subset_id"] == subset_id
                        and row["noise_type"] == noise_type
                        and row["noise_level"] == level
                    )
                    for level in levels
                ]
                for subset_id, _ in pairs
            ]
        )
        image = axis.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        axis.set_title(NOISE_LABELS[noise_type])
        tick_positions = np.arange(0, len(levels), 2)
        axis.set_xticks(
            tick_positions,
            [f"{levels[index] * 100:g}" for index in tick_positions],
            rotation=45,
            ha="right",
        )
        axis.set_xlabel("Ruido nominal (%)")
    axes[0].set_yticks(
        np.arange(len(pairs)),
        [f"{subset_id} · {names}" for subset_id, names in pairs],
        fontsize=8,
    )
    figure.subplots_adjust(
        left=0.18,
        right=0.89,
        bottom=0.18,
        top=0.9,
        wspace=0.12,
    )
    colorbar_axis = figure.add_axes((0.92, 0.2, 0.015, 0.64))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Recuperación media")
    _save_figure(figure, generated / "mapa_robustez_duplas.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    for axis, noise_type in zip(axes, design.noise_types):
        rows = [
            row for row in pair_summary if row["noise_type"] == noise_type
        ]
        axis.scatter(
            [row["pair_similarity"] for row in rows],
            [row["mean_positive_recovery"] * 100 for row in rows],
            color="#7c3aed",
            s=52,
            alpha=0.75,
        )
        axis.set_title(NOISE_LABELS[noise_type])
        axis.set_xlabel("Producto interno absoluto normalizado")
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Recuperación media en niveles positivos (%)")
    figure.tight_layout()
    _save_figure(figure, generated / "robustez_vs_similitud.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    noise_colors = {
        "uniform_flip": "#2563eb",
        "uniform_erasure": "#ea580c",
        "edge_erasure": "#16a34a",
    }
    for axis, storage_size in zip(axes, (1, 2)):
        for noise_type in design.noise_types:
            rows = _series(storage_summary, noise_type, storage_size)
            effective_damage = np.asarray(
                [row["changed_pixels_mean"] / 1369 * 100 for row in rows]
            )
            recovery = np.asarray(
                [row["recovery_rate_mean"] * 100 for row in rows]
            )
            deviations = np.asarray(
                [row["recovery_rate_std"] * 100 for row in rows]
            )
            axis.errorbar(
                effective_damage,
                recovery,
                yerr=_bounded_error(recovery, deviations),
                color=noise_colors[noise_type],
                marker="o",
                markersize=3.5,
                linewidth=1.6,
                capsize=3,
                label=NOISE_LABELS[noise_type],
            )
        axis.set_title(f"{storage_size} QR almacenado{'s' if storage_size > 1 else ''}")
        axis.set_xlabel("Píxeles efectivamente modificados (%)")
        axis.set_xlim(-2, 102)
        axis.set_ylim(-3, 103)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Recuperación correcta (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    _save_figure(figure, generated / "recuperacion_vs_dano_efectivo.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    destination_fields = (
        ("recovery_rate_mean", "Objetivo", "#16a34a"),
        ("other_stored_rate_mean", "Otro almacenado", "#2563eb"),
        ("inverse_target_rate_mean", "Inverso del objetivo", "#7c3aed"),
        ("spurious_rate_mean", "Espurio", "#ea580c"),
        ("cycle_rate_mean", "Ciclo", "#dc2626"),
        ("max_iterations_rate_mean", "Máx. iteraciones", "#64748b"),
    )
    for axis, storage_size in zip(axes, (1, 2)):
        rows = _series(storage_summary, "uniform_flip", storage_size)
        levels_array = np.asarray([row["noise_level"] * 100 for row in rows])
        bottom = np.zeros(len(rows))
        for field, label, color in destination_fields:
            values = np.asarray([row[field] * 100 for row in rows])
            axis.fill_between(
                levels_array,
                bottom,
                bottom + values,
                label=label,
                color=color,
                alpha=0.9,
            )
            bottom += values
        axis.set_title(f"{storage_size} QR almacenado{'s' if storage_size > 1 else ''}")
        axis.set_xlabel("Flip uniforme nominal (%)")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, 100)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Destino de la recuperación (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.82))
    _save_figure(figure, generated / "destinos_flip.svg")
    plt.close(figure)


def _find_summary(
    rows: Sequence[dict], noise_type: str, storage_size: int, level: float
) -> dict:
    return next(
        row
        for row in rows
        if row["noise_type"] == noise_type
        and row["storage_size"] == storage_size
        and row["noise_level"] == level
    )


def write_report(
    output: Path,
    design: QRExperimentDesign,
    baseline_rows: Sequence[dict],
    noise_rows: Sequence[dict],
    storage_summary: Sequence[dict],
    subset_summary: Sequence[dict],
    pair_summary: Sequence[dict],
) -> None:
    comparison_lines = []
    for noise_type in design.noise_types:
        single_50 = _find_summary(storage_summary, noise_type, 1, 0.5)
        pair_50 = _find_summary(storage_summary, noise_type, 2, 0.5)
        single_75 = _find_summary(storage_summary, noise_type, 1, 0.75)
        pair_75 = _find_summary(storage_summary, noise_type, 2, 0.75)
        single_100 = _find_summary(storage_summary, noise_type, 1, 1.0)
        pair_100 = _find_summary(storage_summary, noise_type, 2, 1.0)
        comparison_lines.append(
            f"| {NOISE_LABELS[noise_type]} | "
            f"{single_50['recovery_rate_mean'] * 100:.1f} % | "
            f"{pair_50['recovery_rate_mean'] * 100:.1f} % | "
            f"{single_75['recovery_rate_mean'] * 100:.1f} % | "
            f"{pair_75['recovery_rate_mean'] * 100:.1f} % | "
            f"{single_100['recovery_rate_mean'] * 100:.1f} % | "
            f"{pair_100['recovery_rate_mean'] * 100:.1f} % |"
        )
    pair_extremes = []
    for noise_type in design.noise_types:
        selected = [
            row for row in pair_summary if row["noise_type"] == noise_type
        ]
        values = [row["mean_positive_recovery"] for row in selected]
        minimum = min(values)
        maximum = max(values)
        if np.isclose(minimum, maximum):
            sentence = (
                f"las diez duplas obtienen {minimum * 100:.1f} %."
            )
        else:
            best_count = sum(np.isclose(value, maximum) for value in values)
            worst_count = sum(np.isclose(value, minimum) for value in values)
            best_phrase = (
                "1 dupla alcanza"
                if best_count == 1
                else f"{best_count} duplas alcanzan"
            )
            worst_phrase = (
                "1 queda" if worst_count == 1 else f"{worst_count} quedan"
            )
            sentence = (
                f"los promedios quedan entre {minimum * 100:.1f} % y "
                f"{maximum * 100:.1f} %; {best_phrase} el máximo y "
                f"{worst_phrase} en el mínimo."
            )
        pair_extremes.append(
            f"- **{NOISE_LABELS[noise_type]}:** {sentence}"
        )
    correct = sum(row["outcome"] == "correct" for row in noise_rows)
    energy_violations = sum(
        not row["energy_non_increasing"] for row in noise_rows
    )
    flip_pair_40 = _find_summary(storage_summary, "uniform_flip", 2, 0.4)
    flip_pair_45 = _find_summary(storage_summary, "uniform_flip", 2, 0.45)
    flip_pair_50 = _find_summary(storage_summary, "uniform_flip", 2, 0.5)
    flip_single_50 = _find_summary(storage_summary, "uniform_flip", 1, 0.5)
    flip_single_55 = _find_summary(storage_summary, "uniform_flip", 1, 0.55)
    flip_pair_55 = _find_summary(storage_summary, "uniform_flip", 2, 0.55)
    flip_pair_65 = _find_summary(storage_summary, "uniform_flip", 2, 0.65)
    uniform_pair_90 = _find_summary(
        storage_summary, "uniform_erasure", 2, 0.9
    )
    edge_pair_90 = _find_summary(storage_summary, "edge_erasure", 2, 0.9)
    uniform_single_95 = _find_summary(
        storage_summary, "uniform_erasure", 1, 0.95
    )
    uniform_pair_95 = _find_summary(
        storage_summary, "uniform_erasure", 2, 0.95
    )
    edge_single_95 = _find_summary(
        storage_summary, "edge_erasure", 1, 0.95
    )
    edge_pair_95 = _find_summary(storage_summary, "edge_erasure", 2, 0.95)
    uniform_single_100 = _find_summary(
        storage_summary, "uniform_erasure", 1, 1.0
    )
    uniform_pair_100 = _find_summary(
        storage_summary, "uniform_erasure", 2, 1.0
    )
    positive_random_levels = [
        level for level in design.noise_levels if 0.0 < level < 1.0
    ]
    early_pair_failures = [
        row
        for row in subset_summary
        if row["storage_size"] == 2
        and row["noise_type"] == "uniform_flip"
        and row["noise_level"] < 0.5
        and row["recovery_rate_mean"] < 1.0
    ]
    early_failure_description = ", ".join(
        f"{row['stored_patterns']} a {row['noise_level'] * 100:g} % "
        f"({row['recovery_rate_mean'] * 100:.0f} %)"
        for row in early_pair_failures
    )
    report = f"""# Robustez frente al ruido con uno y dos QR

## Diseño experimental

El [barrido de capacidad](../hopfield-qr-capacidad/README.md) mostró que los
cinco conjuntos individuales y las diez duplas son los únicos subconjuntos
completamente estables sin ruido. Sobre esas 15 memorias se aplican los tres
deterioros ya definidos: flip uniforme, borrado uniforme y borrado hacia los
bordes. Se usan los niveles
{', '.join(f'{level * 100:g} %' for level in design.noise_levels)}. Para los
niveles aleatorios se usan las semillas {', '.join(map(str, design.seeds))}.

Para cada QR, tipo, nivel y semilla se reutiliza exactamente la misma consulta
deteriorada en el conjunto individual y en todas las duplas que contienen ese
QR. Este pareo permite que la comparación mida el efecto de guardar un segundo
patrón y no el de haber elegido posiciones de ruido diferentes.

El 0 % corresponde a {len(baseline_rows)} consultas deterministas. Cada uno de
los {len(positive_random_levels)} niveles entre 5 % y 95 % usa diez semillas.
El 100 % se ejecuta una vez porque deteriorar todos los píxeles produce la
misma consulta con cualquier semilla; repetirla diez veces no aportaría
variabilidad. En total se conservan {len(noise_rows)} corridas con ruido. Las
métricas se calculan primero dentro de cada semilla y luego se informa su media
y desvío estándar. No se busca un nivel óptimo: el ruido es una variable de
deterioro, no un hiperparámetro de la red.

## Uno frente a dos patrones almacenados

![Recuperación con uno y dos QR](generated/recuperacion_uno_vs_dos.svg)

| Ruido | 1 QR a 50 % | 2 QR a 50 % | 1 QR a 75 % | 2 QR a 75 % | 1 QR a 100 % | 2 QR a 100 % |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

Las barras representan el desvío entre semillas. Cuando el desvío es cero,
coinciden con el marcador. La comparación incluye los cinco QR individuales y
las veinte apariciones de objetivos dentro de duplas, sin seleccionar una
combinación después de observar su resultado.

Con flip uniforme, almacenar un solo QR mantiene
{flip_single_50['recovery_rate_mean'] * 100:.0f} % de recuperación hasta el
50 %. A partir de 55 %, el {flip_single_55['inverse_target_rate_mean'] * 100:.0f} %
converge al patrón inverso. Esto es esperable: con más de la mitad de los bits
invertidos, la consulta queda más cerca de `-xi` que de `xi`, y la simetría de
la regla de Hebb hace que ambos sean puntos fijos. No es un error de
implementación.

En las duplas la recuperación es 100 % hasta 35 %, baja a
{flip_pair_40['recovery_rate_mean'] * 100:.0f} % al 40 % y a
{flip_pair_45['recovery_rate_mean'] * 100:.0f} % al 45 %, y cae a
{flip_pair_50['recovery_rate_mean'] * 100:.0f} % al 50 %. En ese nivel,
{flip_pair_50['other_stored_rate_mean'] * 100:.0f} % converge al otro QR,
{flip_pair_50['spurious_rate_mean'] * 100:.0f} % a puntos fijos espurios y
{flip_pair_50['cycle_rate_mean'] * 100:.0f} % entra en ciclos. Al 55 %,
{flip_pair_55['inverse_target_rate_mean'] * 100:.0f} % ya converge al inverso;
desde 65 %, ese destino representa
{flip_pair_65['inverse_target_rate_mean'] * 100:.0f} % de las corridas.

![Destino de las recuperaciones con flip](generated/destinos_flip.svg)

El gráfico separa recuperación correcta, otro patrón almacenado, inverso del
objetivo, estado espurio y ciclo. Esa separación es importante por encima de
50 %: informar solamente "falló" ocultaría la transición sistemática al
inverso.

## Ruido nominal y daño efectivo

![Recuperación según daño efectivo](generated/recuperacion_vs_dano_efectivo.svg)

Los porcentajes nominales no representan el mismo daño efectivo. Un flip del
50 % modifica {flip_pair_50['changed_pixels_mean']:.0f} píxeles. En cambio, el
borrado sólo modifica una posición si originalmente era negra; incluso al
100 % cambia en promedio {uniform_single_100['changed_pixels_mean']:.0f} de los
1369 píxeles. Por eso se incluye otra lectura con el porcentaje realmente
modificado en el eje horizontal.

Los borrados permanecen mucho más robustos: con dos QR, a 90 % la recuperación
es {uniform_pair_90['recovery_rate_mean'] * 100:.1f} % para borrado uniforme y
{edge_pair_90['recovery_rate_mean'] * 100:.1f} % para borrado hacia bordes. Al
95 %, los individuales recuperan
{uniform_single_95['recovery_rate_mean'] * 100:.1f} % y
{edge_single_95['recovery_rate_mean'] * 100:.1f} %, mientras las duplas bajan a
{uniform_pair_95['recovery_rate_mean'] * 100:.1f} % y
{edge_pair_95['recovery_rate_mean'] * 100:.1f} %, respectivamente. En 100 % la
consulta queda completamente blanca y deja de contener información sobre qué
QR era el objetivo; por eso la recuperación correcta cae a 0 %. En las duplas,
{uniform_pair_100['inverse_target_rate_mean'] * 100:.0f} % termina en el
inverso, {uniform_pair_100['spurious_rate_mean'] * 100:.0f} % en un punto fijo
espurio y {uniform_pair_100['cycle_rate_mean'] * 100:.0f} % en un ciclo.

## Diferencias entre duplas

![Mapa de robustez de las duplas](generated/mapa_robustez_duplas.svg)

{chr(10).join(pair_extremes)}

Los extremos anteriores se informan como descripción del barrido exhaustivo,
no como selección de una única dupla representativa. El mapa conserva las diez
combinaciones y permite ver si la degradación ocurre en el mismo nivel.

## Similitud y robustez

![Robustez frente a similitud](generated/robustez_vs_similitud.svg)

Cada punto es una de las diez duplas. El eje vertical promedia por igual los
veinte niveles positivos configurados, por lo que sirve como resumen
descriptivo de toda la curva y no como una métrica optimizada. Con sólo diez
pares, el gráfico se usa para inspeccionar la asociación y no para afirmar una
ley general entre producto interno y tolerancia al ruido.

Antes de la caída general del 50 %, las únicas desviaciones fueron:
**{early_failure_description or 'ninguna'}**. Se concentran entre las duplas de
mayor similitud, aunque el orden no es perfectamente monótono. La evidencia es
compatible con una menor robustez al aumentar la interferencia, pero diez pares
no alcanzan para convertir esa observación en una relación general.

## Control de las corridas

Se recuperaron correctamente **{correct} de {len(noise_rows)}** consultas
deterioradas. Hubo **{energy_violations}** corridas con algún aumento de energía.
La tabla completa conserva resultados, errores, iteraciones y energías finales;
no se guardan imágenes ni trayectorias redundantes para cada una de las
{len(noise_rows)} corridas.

## Archivos generados

- `generated/consultas_sin_ruido.csv`: las 25 líneas de base;
- `generated/corridas_ruido.csv`: las {len(noise_rows)} corridas individuales;
- `generated/resumen_por_cantidad.csv`: comparación entre uno y dos QR;
- `generated/resumen_por_subconjunto.csv`: curvas de los 15 subconjuntos;
- `generated/resumen_robustez_duplas.csv`: resumen de las diez duplas;
- `generated/recuperacion_uno_vs_dos.svg`: curvas con diez semillas;
- `generated/mapa_robustez_duplas.svg`: todas las duplas y niveles;
- `generated/robustez_vs_similitud.svg`: relación descriptiva con el Gram;
- `generated/recuperacion_vs_dano_efectivo.svg`: comparación por cambios reales;
- `generated/destinos_flip.svg`: destinos de la dinámica con flip uniforme.
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
        default=Path("configs/hopfield/qr_robustness.json"),
        help="Configuración del barrido extendido de robustez.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/hopfield-qr-robustez"),
        help="Directorio del informe reproducible.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    design = load_design(args.config)
    patterns = load_qr_patterns(args.data)
    baseline_rows, noise_rows = run_robustness_sweep(patterns, design)
    storage_summary = summarize_by_storage_size(
        baseline_rows, noise_rows, design
    )
    subset_summary = summarize_by_subset(baseline_rows, noise_rows, design)
    pair_summary = summarize_pair_robustness(subset_summary)
    write_tables(
        args.output,
        baseline_rows,
        noise_rows,
        storage_summary,
        subset_summary,
        pair_summary,
    )
    write_plots(
        args.output, design, storage_summary, subset_summary, pair_summary
    )
    write_report(
        args.output,
        design,
        baseline_rows,
        noise_rows,
        storage_summary,
        subset_summary,
        pair_summary,
    )
    print(
        f"Informe generado en {args.output}; "
        f"subconjuntos={len(baseline_rows)} consultas base en 15 memorias; "
        f"corridas con ruido={len(noise_rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
