"""Comparar arquitecturas en la época fija 300."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sia_tp3.digit_experiment import parameter_count


def fixed_row(folder, name, architecture):
    with (Path(folder) / name / "history.csv").open(newline="") as file:
        final = list(csv.DictReader(file))[300]
    return {
        "candidate": name,
        "architecture": "-".join(map(str, architecture)),
        "parameter_count": parameter_count(architecture),
        "epoch": 300,
        "validation_accuracy": float(final["validation_accuracy"]),
        "validation_macro_f1": float(final["validation_macro_f1_present"]),
        "validation_loss": float(final["validation_loss"]),
    }


def run(control_dir, search_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        fixed_row(control_dir, "rotation-4deg-p-05-seed-0", [784, 128, 10]),
        fixed_row(search_dir, "wide-192", [784, 192, 10]),
        fixed_row(search_dir, "wide-256", [784, 256, 10]),
        fixed_row(search_dir, "deep-128-64", [784, 128, 64, 10]),
    ]
    with (output / "architecture-fixed-epoch-300.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["architecture"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, field, title in (
            (axes[0], "validation_accuracy", "Accuracy de validation"),
            (axes[1], "validation_macro_f1", "Macro-F1 de validation")):
        values = [row[field] for row in rows]
        bars = axis.bar(labels, values, color=["#2a6fbb", "#a6a6a6", "#a6a6a6", "#a6a6a6"])
        axis.axhline(values[0], color="#2a6fbb", linestyle="--", alpha=0.55)
        axis.set(title=title, xlabel="Arquitectura", ylabel=title,
                 ylim=(min(values) - 0.003, max(values) + 0.002))
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, fmt="%.4f", padding=3)
    fig.suptitle("Búsqueda de arquitectura · semilla 0, época fija 300")
    fig.savefig(output / "architecture-search.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.control, args.search, args.output)
