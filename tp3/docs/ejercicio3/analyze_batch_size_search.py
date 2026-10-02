"""Comparar tamaños de batch con igual cantidad de épocas y schedule."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAINING_SIZE = 19601


def load_history(folder, run):
    with (Path(folder) / run / "history.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def run(control_dir, search_dir, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = (
        (64, load_history(search_dir, "batch-64")),
        (128, load_history(control_dir, "rotation-4deg-p-05-seed-0")),
        (256, load_history(search_dir, "batch-256")),
    )
    rows = []
    for batch_size, history in candidates:
        final = history[300]
        updates_per_epoch = math.ceil(TRAINING_SIZE / batch_size)
        rows.append({
            "batch_size": batch_size,
            "updates_per_epoch": updates_per_epoch,
            "updates_at_first_decay": updates_per_epoch * 150,
            "updates_at_epoch_300": updates_per_epoch * 300,
            "epoch": 300,
            "training_loss": float(final["loss"]),
            "validation_accuracy": float(final["validation_accuracy"]),
            "validation_macro_f1": float(final["validation_macro_f1_present"]),
            "validation_loss": float(final["validation_loss"]),
        })
    with (output / "batch-size-fixed-epoch-300.csv").open(
            "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    epochs = range(151, 301)
    for batch_size, history in candidates:
        for axis, field, title in (
                (axes[0], "validation_accuracy", "Accuracy de validation"),
                (axes[1], "validation_macro_f1_present", "Macro-F1 de validation")):
            axis.plot(epochs, [float(history[e][field]) for e in epochs],
                      label=f"Batch {batch_size}", alpha=0.9)
            axis.set(title=title, xlabel="Época", ylabel=title)
            axis.grid(alpha=0.2)
            axis.legend()
    axes[0].set_ylim(0.965, 0.985)
    axes[1].set_ylim(0.955, 0.982)
    fig.suptitle("Tamaño de batch con schedule fijo")
    fig.savefig(output / "batch-size-search.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.control, args.search, args.output)
