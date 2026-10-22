"""Experimento controlado de Hopfield con letras bipolares de 5 por 5."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .hopfield import HopfieldModel, fit_hopfield
from .pca_analysis import _load_pyplot, _save_figure


LETTER_BITMAPS: Mapping[str, Tuple[str, ...]] = {
    "A": (".###.", "#...#", "#####", "#...#", "#...#"),
    "B": ("####.", "#...#", "####.", "#...#", "####."),
    "C": (".####", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "####."),
    "E": ("#####", "#....", "####.", "#....", "#####"),
    "F": ("#####", "#....", "####.", "#....", "#...."),
    "G": (".###.", "#....", "#.###", "#...#", ".###."),
    "H": ("#...#", "#...#", "#####", "#...#", "#...#"),
    "I": ("#####", "..#..", "..#..", "..#..", "#####"),
    "J": ("#####", "...#.", "...#.", "#..#.", "###.."),
    "K": ("#...#", "#..#.", "###..", "#..#.", "#...#"),
    "L": ("#....", "#....", "#....", "#....", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#...#", "#...#"),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#"),
    "O": (".###.", "#...#", "#...#", "#...#", ".###."),
    "P": ("####.", "#...#", "####.", "#....", "#...."),
    "Q": (".###.", "#...#", "#.#.#", "#..#.", ".##.#"),
    "R": ("####.", "#...#", "####.", "#..#.", "#...#"),
    "S": (".####", "#....", ".###.", "....#", "####."),
    "T": ("#####", "..#..", "..#..", "..#..", "..#.."),
    "U": ("#...#", "#...#", "#...#", "#...#", ".###."),
    "V": ("#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "W": ("#...#", "#...#", "#.#.#", "##.##", "#...#"),
    "X": ("#...#", ".#.#.", "..#..", ".#.#.", "#...#"),
    "Y": ("#...#", ".#.#.", "..#..", "..#..", "..#.."),
    "Z": ("#####", "...#.", "..#..", ".#...", "#####"),
}


def _bitmap_to_pattern(bitmap: Sequence[str]) -> np.ndarray:
    if len(bitmap) != 5 or any(len(row) != 5 for row in bitmap):
        raise ValueError("cada letra debe tener exactamente 5 filas de 5 celdas")
    if any(character not in ".#" for row in bitmap for character in row):
        raise ValueError("las letras sólo pueden usar '.' y '#'")
    return np.asarray(
        [[1 if character == "#" else -1 for character in row] for row in bitmap],
        dtype=np.int8,
    )


LETTER_PATTERNS: Mapping[str, np.ndarray] = {
    name: _bitmap_to_pattern(bitmap) for name, bitmap in LETTER_BITMAPS.items()
}


@dataclass(frozen=True)
class LetterExperimentDesign:
    """Configuración reproducible del experimento de letras."""

    group_size: int
    noise_levels: Tuple[float, ...]
    seeds: Tuple[int, ...]
    max_iterations: int


@dataclass(frozen=True)
class LetterGroupScore:
    """Resumen de ortogonalidad para una combinación de letras."""

    letters: Tuple[str, ...]
    mean_absolute_inner_product: float
    maximum_absolute_inner_product: int
    maximum_pair_count: int


REPRESENTATIVE_CASES = (
    (
        "correct_high_noise",
        "Correcta con mayor ruido",
        "Mayor ruido con recuperación correcta en el grupo favorable",
    ),
    (
        "first_spurious",
        "Primer estado espurio",
        "Menor ruido con estado fijo espurio en el grupo favorable",
    ),
    (
        "first_cycle",
        "Primer ciclo",
        "Menor ruido con ciclo en el grupo favorable",
    ),
    (
        "baseline_interference",
        "Interferencia sin ruido",
        "Primera letra incorrecta con 0 % de ruido en el grupo desfavorable",
    ),
)


def load_design(path: Path | str) -> LetterExperimentDesign:
    """Leer y validar la configuración JSON del experimento."""

    with Path(path).open(encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("la configuración debe ser un objeto JSON")
    expected = {"group_size", "noise_levels", "seeds", "max_iterations"}
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise ValueError(f"campos faltantes: {missing}; desconocidos: {unknown}")

    group_size = raw["group_size"]
    if isinstance(group_size, bool) or group_size != 4:
        raise ValueError("group_size debe ser 4 para el caso pedido")
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
    return LetterExperimentDesign(group_size, noise_levels, seeds, maximum)


def letter_matrix(names: Sequence[str]) -> np.ndarray:
    """Aplanar un conjunto ordenado de letras, una por fila."""

    if not names:
        raise ValueError("se necesita al menos una letra")
    unknown = [name for name in names if name not in LETTER_PATTERNS]
    if unknown:
        raise ValueError(f"letras desconocidas: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("las letras no pueden repetirse")
    return np.asarray([LETTER_PATTERNS[name].reshape(-1) for name in names])


def gram_matrix(names: Sequence[str]) -> np.ndarray:
    """Calcular los productos internos entre los patrones indicados."""

    patterns = letter_matrix(names)
    return patterns @ patterns.T


def rank_letter_groups(group_size: int = 4) -> List[LetterGroupScore]:
    """Ordenar grupos desde menor interferencia media y máxima."""

    if group_size < 2 or group_size > len(LETTER_PATTERNS):
        raise ValueError("group_size queda fuera del rango permitido")
    scores = []
    for names in itertools.combinations(sorted(LETTER_PATTERNS), group_size):
        gram = gram_matrix(names)
        upper = np.abs(gram[np.triu_indices(group_size, 1)])
        maximum = int(np.max(upper))
        scores.append(
            LetterGroupScore(
                letters=names,
                mean_absolute_inner_product=float(np.mean(upper)),
                maximum_absolute_inner_product=maximum,
                maximum_pair_count=int(np.count_nonzero(upper == maximum)),
            )
        )
    return sorted(
        scores,
        key=lambda score: (
            score.mean_absolute_inner_product,
            score.maximum_absolute_inner_product,
            score.maximum_pair_count,
            score.letters,
        ),
    )


def add_flip_noise(pattern: np.ndarray, level: float, seed: int) -> np.ndarray:
    """Invertir exactamente la fracción indicada de posiciones."""

    values = np.asarray(pattern, dtype=np.int8)
    if values.ndim != 1 or not np.isin(values, (-1, 1)).all():
        raise ValueError("pattern debe ser un vector bipolar")
    if not np.isfinite(level) or not 0.0 <= level <= 1.0:
        raise ValueError("level debe quedar entre 0 y 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed debe ser un entero")
    noisy = values.copy()
    amount = int(round(level * len(noisy)))
    if amount:
        indices = np.random.default_rng(seed).choice(
            len(noisy), size=amount, replace=False
        )
        noisy[indices] *= -1
    return noisy


def _outcome(
    model: HopfieldModel,
    target_index: int,
    status: str,
    final_state: np.ndarray,
) -> str:
    if status == "cycle":
        return "cycle"
    if status == "max_iterations":
        return "max_iterations"
    match = model.matching_pattern_index(final_state)
    if match == target_index:
        return "correct"
    if match is not None:
        return "other_stored"
    return "spurious_fixed"


def run_experiment(
    design: LetterExperimentDesign,
) -> Tuple[List[LetterGroupScore], Dict[str, Tuple[str, ...]], List[dict]]:
    """Comparar el mejor y peor grupo bajo perturbaciones equivalentes."""

    ranking = rank_letter_groups(design.group_size)
    groups = {
        "favorable": ranking[0].letters,
        "desfavorable": ranking[-1].letters,
    }
    rows: List[dict] = []
    for group_name, letters in groups.items():
        patterns = letter_matrix(letters)
        model = fit_hopfield(patterns)
        for target_index, (letter, target) in enumerate(zip(letters, patterns)):
            for level in design.noise_levels:
                for seed in design.seeds:
                    query = add_flip_noise(target, level, seed)
                    result = model.recall(
                        query, max_iterations=design.max_iterations
                    )
                    matched_index = model.matching_pattern_index(
                        result.final_state
                    )
                    energy_differences = np.diff(result.energies)
                    rows.append(
                        {
                            "group": group_name,
                            "letters": "".join(letters),
                            "target": letter,
                            "noise_level": level,
                            "seed": seed,
                            "flipped_pixels": int(np.count_nonzero(query != target)),
                            "outcome": _outcome(
                                model,
                                target_index,
                                result.status,
                                result.final_state,
                            ),
                            "matched_letter": (
                                letters[matched_index]
                                if matched_index is not None
                                else ""
                            ),
                            "status": result.status,
                            "iterations": result.iterations,
                            "cycle_length": result.cycle_length or "",
                            "final_pixel_errors": int(
                                np.count_nonzero(result.final_state != target)
                            ),
                            "initial_energy": float(result.energies[0]),
                            "final_energy": float(result.energies[-1]),
                            "energy_drop": float(
                                result.energies[0] - result.energies[-1]
                            ),
                            "non_increasing_energy": bool(
                                np.all(energy_differences <= 1e-10)
                            ),
                        }
                    )
    return ranking, groups, rows


def summarize_runs(rows: Sequence[dict]) -> List[dict]:
    """Agregar primero por semilla y luego resumir las diez repeticiones."""

    per_seed: Dict[Tuple[str, float, int], List[dict]] = {}
    for row in rows:
        key = (row["group"], row["noise_level"], row["seed"])
        per_seed.setdefault(key, []).append(row)

    seed_metrics = []
    outcomes = ("correct", "other_stored", "spurious_fixed", "cycle")
    for (group, noise, seed), seed_rows in sorted(per_seed.items()):
        seed_metrics.append(
            {
                "group": group,
                "noise_level": noise,
                "seed": seed,
                "recovery_rate": np.mean(
                    [row["outcome"] == "correct" for row in seed_rows]
                ),
                "final_pixel_errors": np.mean(
                    [row["final_pixel_errors"] for row in seed_rows]
                ),
                "iterations": np.mean([row["iterations"] for row in seed_rows]),
                **{
                    f"{outcome}_rate": np.mean(
                        [row["outcome"] == outcome for row in seed_rows]
                    )
                    for outcome in outcomes
                },
            }
        )

    grouped: Dict[Tuple[str, float], List[dict]] = {}
    for row in seed_metrics:
        grouped.setdefault((row["group"], row["noise_level"]), []).append(row)
    summary = []
    metrics = (
        "recovery_rate",
        "final_pixel_errors",
        "iterations",
        "correct_rate",
        "other_stored_rate",
        "spurious_fixed_rate",
        "cycle_rate",
    )
    for (group, noise), group_rows in sorted(grouped.items()):
        item = {"group": group, "noise_level": noise, "seed_count": len(group_rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in group_rows], dtype=float)
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_std"] = float(np.std(values, ddof=1))
        summary.append(item)
    return summary


def summarize_by_letter(rows: Sequence[dict]) -> List[dict]:
    """Conservar la heterogeneidad entre letras que oculta el promedio."""

    grouped: Dict[Tuple[str, str, float], List[dict]] = {}
    for row in rows:
        key = (row["group"], row["target"], row["noise_level"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (group, target, noise), group_rows in sorted(grouped.items()):
        outcomes = [row["outcome"] for row in group_rows]
        summary.append(
            {
                "group": group,
                "target": target,
                "noise_level": noise,
                "seed_count": len(group_rows),
                "recovery_rate": float(
                    np.mean([outcome == "correct" for outcome in outcomes])
                ),
                "other_stored_rate": float(
                    np.mean([outcome == "other_stored" for outcome in outcomes])
                ),
                "spurious_fixed_rate": float(
                    np.mean([outcome == "spurious_fixed" for outcome in outcomes])
                ),
                "cycle_rate": float(
                    np.mean([outcome == "cycle" for outcome in outcomes])
                ),
                "mean_final_pixel_errors": float(
                    np.mean([row["final_pixel_errors"] for row in group_rows])
                ),
                "mean_iterations": float(
                    np.mean([row["iterations"] for row in group_rows])
                ),
            }
        )
    return summary


def select_representative_rows(rows: Sequence[dict]) -> List[dict]:
    """Elegir casos por reglas explícitas, sin selección visual posterior."""

    selectors = {
        "correct_high_noise": lambda row: (
            row["group"] == "favorable" and row["outcome"] == "correct"
        ),
        "first_spurious": lambda row: (
            row["group"] == "favorable"
            and row["outcome"] == "spurious_fixed"
        ),
        "first_cycle": lambda row: (
            row["group"] == "favorable" and row["outcome"] == "cycle"
        ),
        "baseline_interference": lambda row: (
            row["group"] == "desfavorable"
            and row["noise_level"] == 0.0
            and row["outcome"] != "correct"
        ),
    }
    selected = []
    for case_id, title, criterion in REPRESENTATIVE_CASES:
        candidates = [row for row in rows if selectors[case_id](row)]
        if not candidates:
            raise ValueError(f"no se encontró un caso para: {case_id}")
        if case_id == "correct_high_noise":
            candidates.sort(
                key=lambda row: (-row["noise_level"], row["target"], row["seed"])
            )
        else:
            candidates.sort(
                key=lambda row: (row["noise_level"], row["target"], row["seed"])
            )
        chosen = dict(candidates[0])
        chosen["case_id"] = case_id
        chosen["case_title"] = title
        chosen["selection_criterion"] = criterion
        selected.append(chosen)
    return selected


def rebuild_representative_cases(
    selected_rows: Sequence[dict],
    groups: Mapping[str, Tuple[str, ...]],
    max_iterations: int,
) -> List[dict]:
    """Reconstruir consultas y trayectorias desde la tabla de corridas."""

    cases = []
    for row in selected_rows:
        letters = groups[row["group"]]
        patterns = letter_matrix(letters)
        model = fit_hopfield(patterns)
        target_index = letters.index(row["target"])
        target = patterns[target_index]
        query = add_flip_noise(target, row["noise_level"], row["seed"])
        result = model.recall(query, max_iterations=max_iterations)
        cases.append(
            {
                "metadata": row,
                "target": target,
                "query": query,
                "result": result,
            }
        )
    return cases


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError("no hay filas para escribir")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(
    output: Path,
    ranking: Sequence[LetterGroupScore],
    rows: Sequence[dict],
    summary: Sequence[dict],
    letter_summary: Sequence[dict],
    representative_rows: Sequence[dict],
) -> None:
    """Guardar resultados suficientes sin versionar las 14.950 combinaciones."""

    generated = output / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    selected = list(ranking[:10]) + list(ranking[-10:])
    ranking_rows = [
        {
            "position": index if index <= 10 else len(ranking) - 20 + index,
            "selection": "mejor" if index <= 10 else "peor",
            "letters": "".join(score.letters),
            "mean_absolute_inner_product": score.mean_absolute_inner_product,
            "maximum_absolute_inner_product": score.maximum_absolute_inner_product,
            "maximum_pair_count": score.maximum_pair_count,
        }
        for index, score in enumerate(selected, start=1)
    ]
    _write_csv(generated / "ranking_extremos.csv", ranking_rows)
    _write_csv(generated / "corridas.csv", rows)
    _write_csv(generated / "resumen.csv", summary)
    _write_csv(generated / "resumen_por_letra.csv", letter_summary)
    representative_fields = (
        "case_id",
        "case_title",
        "selection_criterion",
        "group",
        "letters",
        "target",
        "noise_level",
        "seed",
        "outcome",
        "status",
        "iterations",
        "matched_letter",
        "flipped_pixels",
        "final_pixel_errors",
        "initial_energy",
        "final_energy",
        "energy_drop",
    )
    _write_csv(
        generated / "casos_representativos.csv",
        [
            {field: row[field] for field in representative_fields}
            for row in representative_rows
        ],
    )


def _summary_arrays(
    summary: Sequence[dict], group: str, metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in summary if row["group"] == group]
    return (
        np.asarray([row["noise_level"] * 100 for row in selected]),
        np.asarray([row[f"{metric}_mean"] for row in selected]),
        np.asarray([row[f"{metric}_std"] for row in selected]),
    )


def _bounded_error(
    mean: np.ndarray,
    deviation: np.ndarray,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Acotar barras simétricas al rango válido de una métrica."""

    lower = np.minimum(deviation, mean - lower_bound)
    upper = np.minimum(deviation, upper_bound - mean)
    return np.vstack((np.maximum(lower, 0.0), np.maximum(upper, 0.0)))


def write_plots(output: Path, summary: Sequence[dict]) -> None:
    """Generar tres figuras agregadas y reproducibles."""

    generated = output / "generated"
    plt = _load_pyplot()
    colors = {"favorable": "#3b82f6", "desfavorable": "#ef4444"}

    figure, axis = plt.subplots(figsize=(8, 5))
    for group in colors:
        noise, mean, deviation = _summary_arrays(summary, group, "recovery_rate")
        percentages = mean * 100
        errors = _bounded_error(
            percentages, deviation * 100, lower_bound=0.0, upper_bound=100.0
        )
        axis.errorbar(
            noise,
            percentages,
            yerr=errors,
            marker="o",
            capsize=4,
            elinewidth=1.5,
            linewidth=2,
            color=colors[group],
            label=group.capitalize(),
            zorder=3,
        )
    axis.set(xlabel="Ruido (% de píxeles invertidos)", ylabel="Recuperación correcta (%)")
    axis.set_xlim(-3, 53)
    axis.set_ylim(-3, 103)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, generated / "recuperacion_vs_ruido.svg")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for group in colors:
        noise, mean, deviation = _summary_arrays(
            summary, group, "final_pixel_errors"
        )
        axis.errorbar(
            noise,
            mean,
            yerr=_bounded_error(
                mean, deviation, lower_bound=0.0, upper_bound=25.0
            ),
            marker="o",
            capsize=4,
            elinewidth=1.5,
            linewidth=2,
            color=colors[group],
            label=group.capitalize(),
            zorder=3,
        )
    axis.set(xlabel="Ruido (% de píxeles invertidos)", ylabel="Píxeles finales incorrectos")
    axis.set_xlim(-3, 53)
    axis.set_ylim(-1, 26)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    _save_figure(figure, generated / "error_vs_ruido.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    outcome_metrics = (
        ("correct_rate", "Correcta", "#22c55e"),
        ("other_stored_rate", "Otra letra", "#f59e0b"),
        ("spurious_fixed_rate", "Espurio", "#8b5cf6"),
        ("cycle_rate", "Ciclo", "#ef4444"),
    )
    for axis, group in zip(axes, colors):
        selected = [row for row in summary if row["group"] == group]
        noise = np.asarray([row["noise_level"] * 100 for row in selected])
        bottom = np.zeros(len(selected))
        for metric, label, color in outcome_metrics:
            values = np.asarray([row[f"{metric}_mean"] for row in selected]) * 100
            axis.bar(noise, values, bottom=bottom, width=3.8, label=label, color=color)
            bottom += values
        axis.set_title(group.capitalize())
        axis.set_xlabel("Ruido (%)")
        axis.set_xlim(-3, 53)
        axis.set_ylim(0, 102)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Resultados (%)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    figure.tight_layout()
    _save_figure(figure, generated / "resultados_vs_ruido.svg")
    plt.close(figure)


def _mark_errors(axis, state: np.ndarray, target: np.ndarray) -> None:
    """Enmarcar en rojo las celdas que difieren del patrón objetivo."""

    from matplotlib.patches import Rectangle

    for index in np.flatnonzero(state != target):
        row, column = divmod(int(index), 5)
        axis.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#ef4444",
                linewidth=1.8,
            )
        )


def write_representative_plots(output: Path, cases: Sequence[dict]) -> None:
    """Mostrar evolución visual y energía de los casos preseleccionados."""

    generated = output / "generated"
    plt = _load_pyplot()
    trajectories = []
    for case in cases:
        states = list(case["result"].states)
        if len(states) >= 2 and np.array_equal(states[-1], states[-2]):
            states = states[:-1]
        trajectories.append([case["target"], *states])
    columns = max(len(trajectory) for trajectory in trajectories)
    figure, axes = plt.subplots(
        len(cases), columns, figsize=(2.1 * columns, 2.1 * len(cases)), squeeze=False
    )
    for row_index, (case, trajectory) in enumerate(zip(cases, trajectories)):
        metadata = case["metadata"]
        for column_index, axis in enumerate(axes[row_index]):
            axis.set_xticks([])
            axis.set_yticks([])
            if column_index >= len(trajectory):
                axis.axis("off")
                continue
            state = trajectory[column_index]
            axis.imshow(state.reshape(5, 5), cmap="binary", vmin=-1, vmax=1)
            if column_index == 0:
                title = "Objetivo"
            else:
                title = f"S({column_index - 1})"
                _mark_errors(axis, state, case["target"])
            axis.set_title(title, fontsize=9)
        axes[row_index, 0].set_ylabel(
            f"{metadata['case_title']}\n"
            f"{metadata['target']} · {metadata['noise_level'] * 100:g} % · "
            f"semilla {metadata['seed']}",
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
            fontsize=9,
        )
    figure.suptitle("Evolución de casos seleccionados por reglas reproducibles")
    figure.tight_layout()
    _save_figure(figure, generated / "casos_representativos.svg")
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for axis, case in zip(axes.reshape(-1), cases):
        metadata = case["metadata"]
        energies = case["result"].energies
        axis.plot(range(len(energies)), energies, marker="o", color="#3b82f6")
        axis.set_title(metadata["case_title"])
        axis.set_xlabel("Iteración")
        axis.set_ylabel("Energía")
        axis.set_xticks(range(len(energies)))
        axis.grid(alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, generated / "energia_casos_representativos.svg")
    plt.close(figure)


def write_report(
    output: Path,
    design: LetterExperimentDesign,
    ranking: Sequence[LetterGroupScore],
    groups: Mapping[str, Tuple[str, ...]],
    rows: Sequence[dict],
    summary: Sequence[dict],
    representative_rows: Sequence[dict],
) -> None:
    """Documentar el diseño, los resultados y sus límites."""

    best = ranking[0]
    worst = ranking[-1]
    zero_noise = {
        row["group"]: row
        for row in summary
        if row["noise_level"] == 0.0
    }
    last_noise = max(design.noise_levels)
    last = {
        row["group"]: row
        for row in summary
        if row["noise_level"] == last_noise
    }
    zero_noise_transitions = {}
    for row in rows:
        if row["noise_level"] == 0.0 and row["seed"] == design.seeds[0]:
            zero_noise_transitions[(row["group"], row["target"])] = (
                row["matched_letter"] or row["outcome"]
            )
    unfavorable_transitions = ", ".join(
        f"{letter}→{zero_noise_transitions[('desfavorable', letter)]}"
        for letter in groups["desfavorable"]
    )
    best_repetition = (
        "una vez"
        if best.maximum_pair_count == 1
        else f"{best.maximum_pair_count} veces"
    )
    worst_repetition = (
        "una vez"
        if worst.maximum_pair_count == 1
        else f"{worst.maximum_pair_count} veces"
    )
    report = f"""# Experimento controlado de Hopfield con letras 5×5

## Objetivo

Validar la implementación con patrones pequeños e interpretables y estudiar si
la cercanía a la ortogonalidad se relaciona con una recuperación más robusta
ante ruido. El experimento no ajusta hiperparámetros de Hopfield: el nivel de
ruido y el conjunto almacenado son condiciones de evaluación.

## Diseño reproducible

- patrones: las 26 letras definidas explícitamente como matrices bipolares 5×5;
- tamaño de grupo: {design.group_size};
- combinaciones evaluadas: {len(ranking)};
- niveles de ruido: {', '.join(f'{level * 100:g} %' for level in design.noise_levels)};
- semillas: {', '.join(map(str, design.seeds))};
- máximo de iteraciones: {design.max_iterations}, usado sólo como resguardo;
- ruido: inversión sin reemplazo de exactamente `round(nivel × 25)` posiciones.

Para evitar elegir por una única corrida, cada punto agrega cuatro letras y
{len(design.seeds)} semillas. Las barras representan un desvío estándar entre
semillas, acotado al rango válido de la métrica para evitar extremos fuera de
0–100 % o de 0–25 píxeles. Cuando la variabilidad es cero, la barra coincide
con el marcador y por eso no tiene longitud visible. La semilla no se
interpreta como una variable ordenada.

## Selección de grupos

El orden es lexicográfico: primero se minimiza el producto interno absoluto
medio, luego el máximo y finalmente la cantidad de pares que alcanzan ese
máximo. No se construye un score ponderado.

- grupo favorable **{''.join(groups['favorable'])}**: media
  {best.mean_absolute_inner_product:.2f}, máximo
  {best.maximum_absolute_inner_product}, repetido
  {best_repetition};
- grupo desfavorable **{''.join(groups['desfavorable'])}**: media
  {worst.mean_absolute_inner_product:.2f}, máximo
  {worst.maximum_absolute_inner_product}, repetido
  {worst_repetition}.

Se conservan sólo los diez extremos de cada lado en la tabla generada; el
ranking completo se recalcula de forma determinista al ejecutar el script.

## Resultados

Sin ruido, la recuperación correcta fue
**{zero_noise['favorable']['recovery_rate_mean'] * 100:.1f} %** para el grupo
favorable y **{zero_noise['desfavorable']['recovery_rate_mean'] * 100:.1f} %**
para el desfavorable. Con {last_noise * 100:g} % de ruido fue
**{last['favorable']['recovery_rate_mean'] * 100:.1f} %** y
**{last['desfavorable']['recovery_rate_mean'] * 100:.1f} %**, respectivamente.

La recuperación de 25 % sin ruido del grupo desfavorable no representa un
éxito parcial repartido entre sus cuatro letras: sus transiciones son
**{unfavorable_transitions}**. Por lo tanto, una
sola letra funciona como atractor dominante. Su porcentaje agregado en niveles
altos de ruido no debe interpretarse como una mejora ni como robustez del grupo.

![Recuperación frente al ruido](generated/recuperacion_vs_ruido.svg)

![Error final frente al ruido](generated/error_vs_ruido.svg)

![Tipos de resultado](generated/resultados_vs_ruido.svg)

## Casos representativos

Los casos no se eligieron por su apariencia. Se aplicaron cuatro reglas antes
de graficar: mayor ruido todavía recuperado, menor ruido con estado espurio,
menor ruido con ciclo y primera interferencia sin ruido. Los desempates usan
letra y semilla en orden ascendente.

{chr(10).join(f"- **{row['case_title']}**: {row['target']}, {row['noise_level'] * 100:g} % de ruido, semilla {row['seed']}, resultado `{row['outcome']}`." for row in representative_rows)}

Los bordes rojos señalan celdas distintas del objetivo. `S(0)` es la consulta
ruidosa y los estados siguientes muestran la evolución de la red.

![Evolución de casos representativos](generated/casos_representativos.svg)

![Energía de casos representativos](generated/energia_casos_representativos.svg)

Estas curvas describen robustez y modos de fallo; no se elige un "nivel óptimo"
de ruido. Antes de extrapolar al caso QR se deben inspeccionar los resultados
por letra y las trayectorias de energía de casos representativos.

## Archivos generados

- `generated/ranking_extremos.csv`: diez mejores y diez peores grupos;
- `generated/corridas.csv`: una fila por letra, ruido y semilla;
- `generated/resumen.csv`: medias y desvíos agregados por semilla;
- `generated/resumen_por_letra.csv`: resultados sin ocultar diferencias entre
  patrones;
- `generated/casos_representativos.csv`: reglas y corridas mostradas;
- cinco gráficos SVG agregados, sin imágenes redundantes por corrida.
"""
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(report, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hopfield/letters.json"),
        help="Configuración del experimento.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/hopfield-letras"),
        help="Directorio del informe reproducible.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    design = load_design(args.config)
    ranking, groups, rows = run_experiment(design)
    summary = summarize_runs(rows)
    letter_summary = summarize_by_letter(rows)
    representative_rows = select_representative_rows(rows)
    cases = rebuild_representative_cases(
        representative_rows, groups, design.max_iterations
    )
    write_tables(
        args.output,
        ranking,
        rows,
        summary,
        letter_summary,
        representative_rows,
    )
    write_plots(args.output, summary)
    write_representative_plots(args.output, cases)
    write_report(
        args.output,
        design,
        ranking,
        groups,
        rows,
        summary,
        representative_rows,
    )
    print(
        f"Informe generado en {args.output}; "
        f"grupo favorable={''.join(groups['favorable'])}; "
        f"grupo desfavorable={''.join(groups['desfavorable'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
