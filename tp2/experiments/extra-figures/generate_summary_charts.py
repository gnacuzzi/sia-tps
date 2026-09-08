#!/usr/bin/env python3
"""Generate defense-oriented charts from the committed experiment summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
SUMMARIES = HERE.parent / "results" / "summaries"
BLUE = "#357fb5"
RED = "#db5b68"
ORANGE = "#e99513"
GREEN = "#2b9278"
NAVY = "#17384d"
MUTED = "#607782"


def read_phase(phase: str) -> list[dict[str, str]]:
    with (SUMMARIES / f"{phase}.csv").open(encoding="utf-8", newline="") as stream:
        return [row for row in csv.DictReader(stream) if row["target"] == "flag"]


def values(rows: list[dict[str, str]]) -> tuple[list[float], list[list[float]]]:
    medians = [float(row["median_best_error"]) * 1000 for row in rows]
    q1 = [float(row["q1_best_error"]) * 1000 for row in rows]
    q3 = [float(row["q3_best_error"]) * 1000 for row in rows]
    errors = [
        [median - lower for median, lower in zip(medians, q1)],
        [upper - median for median, upper in zip(medians, q3)],
    ]
    return medians, errors


def finish(axis: plt.Axes, figure: plt.Figure, filename: str) -> None:
    axis.grid(axis="y", alpha=0.24)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    figure.text(
        0.5,
        0.018,
        "Bandera de Colombia · 5 semillas por condición · rayas: 50 % central (IQR)",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=MUTED,
    )
    figure.tight_layout(rect=(0.025, 0.06, 0.985, 0.98))
    figure.savefig(HERE / f"{filename}.png", dpi=220, bbox_inches="tight")
    figure.savefig(HERE / f"{filename}.svg", format="svg", bbox_inches="tight")
    plt.close(figure)


def render_vertical(
    phase: str,
    conditions: list[str],
    labels: list[str],
    title: str,
    filename: str,
) -> None:
    by_condition = {row["condition"]: row for row in read_phase(phase)}
    rows = [by_condition[condition] for condition in conditions]
    medians, errors = values(rows)
    best = min(range(len(medians)), key=medians.__getitem__)
    colors = [BLUE if index == best else RED for index in range(len(medians))]

    figure, axis = plt.subplots(figsize=(12, 7.2))
    bars = axis.bar(
        range(len(rows)),
        medians,
        width=0.58,
        color=colors,
        yerr=errors,
        capsize=6,
        error_kw={"elinewidth": 1.7, "ecolor": NAVY},
    )
    axis.set_title(title, fontsize=20, fontweight="bold", pad=20)
    axis.set_ylabel("NMSE final mediano × 1000 (menor es mejor)", fontsize=13, fontweight="bold")
    axis.set_xticks(range(len(labels)), labels, fontsize=12, fontweight="bold")
    axis.set_ylim(bottom=0)
    for bar, median in zip(bars, medians):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            median,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=NAVY,
        )
    finish(axis, figure, filename)


def render_selection() -> None:
    label_map = {
        "elite": "Elite",
        "roulette": "Ruleta",
        "universal": "Universal (SUS)",
        "ranking": "Ranking",
        "boltzmann": "Boltzmann",
        "tournament_2": "Tournament 2",
        "tournament_5": "Tournament 5",
        "probabilistic_0_6": "Torneo probabilístico 0,6",
    }
    rows = sorted(read_phase("selection"), key=lambda row: float(row["median_best_error"]), reverse=True)
    medians, errors = values(rows)
    best = min(range(len(medians)), key=medians.__getitem__)
    colors = [BLUE if index == best else "#93a4ad" for index in range(len(rows))]

    figure, axis = plt.subplots(figsize=(12, 7.2))
    positions = list(range(len(rows)))
    bars = axis.barh(
        positions,
        medians,
        height=0.58,
        color=colors,
        xerr=errors,
        capsize=4,
        error_kw={"elinewidth": 1.4, "ecolor": NAVY},
    )
    axis.set_title("Comparación de los métodos de selección — Colombia", fontsize=20, fontweight="bold", pad=20)
    axis.set_xlabel("NMSE final mediano × 1000 (menor es mejor)", fontsize=13, fontweight="bold")
    axis.set_yticks(positions, [label_map[row["condition"]] for row in rows], fontsize=11)
    axis.grid(axis="x", alpha=0.24)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, median in zip(bars, medians):
        axis.text(
            median,
            bar.get_y() + bar.get_height() / 2,
            f"  {median:.3f}",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=NAVY,
        )
    figure.text(
        0.5,
        0.018,
        "Bandera de Colombia · 5 semillas por condición · rayas: 50 % central (IQR)",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color=MUTED,
    )
    figure.tight_layout(rect=(0.025, 0.06, 0.985, 0.98))
    figure.savefig(HERE / "03-selection-colombia.png", dpi=220, bbox_inches="tight")
    figure.savefig(HERE / "03-selection-colombia.svg", format="svg", bbox_inches="tight")
    plt.close(figure)


def render_crossover_summary() -> None:
    by_condition = {row["condition"]: row for row in read_phase("crossover")}
    conditions = [
        "tournament_5__uniform",
        "tournament_5__one_point",
        "ranking__uniform",
        "ranking__one_point",
    ]
    rows = [by_condition[condition] for condition in conditions]
    medians, errors = values(rows)
    positions = [0, 1, 2.5, 3.5]
    colors = [BLUE, RED, GREEN, ORANGE]

    figure, axis = plt.subplots(figsize=(12, 7.2))
    bars = axis.bar(
        positions,
        medians,
        width=0.68,
        color=colors,
        yerr=errors,
        capsize=5,
        error_kw={"elinewidth": 1.5, "ecolor": NAVY},
    )
    axis.set_title("Cruza e interacción con la selección — Colombia", fontsize=20, fontweight="bold", pad=20)
    axis.set_ylabel("NMSE final mediano × 1000 (menor es mejor)", fontsize=13, fontweight="bold")
    axis.set_xticks(positions, ["Uniforme", "Un punto", "Uniforme", "Un punto"], fontsize=11)
    axis.text(0.5, -0.11, "Tournament 5", transform=axis.get_xaxis_transform(), ha="center", fontweight="bold")
    axis.text(3.0, -0.11, "Ranking", transform=axis.get_xaxis_transform(), ha="center", fontweight="bold")
    axis.set_ylim(bottom=0)
    for bar, median in zip(bars, medians):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            median,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=NAVY,
        )
    finish(axis, figure, "04-crossover-summary-colombia")


def main() -> None:
    render_vertical(
        "resolution",
        ["side_32", "side_64"],
        ["32 px", "64 px"],
        "Calidad final según la resolución de trabajo",
        "01-resolution-colombia",
    )
    render_vertical(
        "capacity",
        ["triangles_10", "triangles_25", "triangles_50"],
        ["10", "25", "50"],
        "Calidad final según la cantidad de triángulos",
        "02-triangle-count-colombia",
    )
    render_selection()
    render_crossover_summary()
    render_vertical(
        "mutation",
        ["single_local", "multigene_local_balanced"],
        ["Un gen local", "Multigénica balanceada"],
        "Comparación de los métodos de mutación — Colombia",
        "06-mutation-colombia",
    )
    render_vertical(
        "survival",
        ["additive", "exclusive"],
        ["Aditiva", "Exclusiva"],
        "Comparación de las estrategias de supervivencia — Colombia",
        "07-survival-colombia",
    )


if __name__ == "__main__":
    main()
