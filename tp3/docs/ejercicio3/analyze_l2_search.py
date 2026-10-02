"""Consolidar la búsqueda inicial de regularización L2."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def class_metric(metrics, label, name):
    row = next(item for item in metrics["validation"]["per_class"]
               if item["label"] == label)
    return row[name]


def run(input_dir, output):
    input_dir, output = Path(input_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((input_dir / "config.json").read_text())
    rows, histories = [], {}
    for run_config in config["runs"]:
        name = run_config["name"]
        metrics = json.loads((input_dir / name / "metrics.json").read_text())
        with (input_dir / name / "history.csv").open(newline="") as file:
            histories[name] = list(csv.DictReader(file))
        rows.append({
            "run": name,
            "l2_lambda": run_config.get("l2_lambda", 0.0),
            "best_epoch": metrics["best_epoch"],
            "validation_accuracy": metrics["validation"]["accuracy"],
            "validation_macro_f1": metrics["validation"]["macro_f1"],
            "digit_5_f1": class_metric(metrics, 5, "f1"),
            "digit_8_f1": class_metric(metrics, 8, "f1"),
        })
    write_csv(output / "l2-search-summary.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for row in rows:
        history = histories[row["run"]]
        epochs = [int(item["epoch"]) for item in history]
        label = f"lambda={row['l2_lambda']:g}"
        axes[0].plot(epochs, [float(item["validation_macro_f1_present"])
                             for item in history], label=label)
        axes[1].plot(epochs, [float(item["validation_loss"])
                             for item in history], label=label)
    axes[0].set(title="Macro-F1 de validation", xlabel="Época",
                ylabel="Macro-F1", ylim=(0.85, 1.0))
    axes[1].set(title="Entropía cruzada de validation", xlabel="Época",
                ylabel="Cross-entropy")
    for axis in axes:
        axis.legend()
    fig.suptitle("Búsqueda de regularización L2 · semilla 0")
    fig.savefig(output / "l2-search-curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    positions = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar([position - width / 2 for position in positions],
           [row["digit_5_f1"] for row in rows], width, label="F1 del 5")
    ax.bar([position + width / 2 for position in positions],
           [row["digit_8_f1"] for row in rows], width, label="F1 del 8")
    ax.set(title="Clases minoritarias en el mejor checkpoint",
           xlabel="Lambda L2", ylabel="F1", ylim=(0.8, 1.0),
           xticks=positions,
           xticklabels=[f"{row['l2_lambda']:g}" for row in rows])
    ax.legend()
    fig.savefig(output / "l2-search-minority-f1.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
