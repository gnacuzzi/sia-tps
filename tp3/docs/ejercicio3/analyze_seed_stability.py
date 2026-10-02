"""Consolidar las cinco semillas del baseline del ejercicio 3."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREFIX = "baseline-exercise2-new-data-seed-"


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def class_f1(metrics, label):
    return next(row["f1"] for row in metrics["validation"]["per_class"]
                if row["label"] == label)


def run(input_dir, output):
    input_dir, output = Path(input_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    histories = {}
    for seed in range(5):
        run_dir = input_dir / f"{PREFIX}{seed}"
        metrics = json.loads((run_dir / "metrics.json").read_text())
        with (run_dir / "history.csv").open(newline="") as file:
            histories[seed] = list(csv.DictReader(file))
        seed_rows.append({
            "seed": seed,
            "best_epoch": metrics["best_epoch"],
            "validation_accuracy": metrics["validation"]["accuracy"],
            "validation_macro_f1": metrics["validation"]["macro_f1"],
            "digit_5_f1": class_f1(metrics, 5),
            "digit_8_f1": class_f1(metrics, 8),
        })
    write_csv(output / "baseline-five-seeds.csv", seed_rows)

    summary = {}
    for metric in ("best_epoch", "validation_accuracy", "validation_macro_f1",
                   "digit_5_f1", "digit_8_f1"):
        values = [row[metric] for row in seed_rows]
        summary[metric] = {
            "mean": statistics.mean(values),
            "sd": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    summary["seeds_with_zero_digit_8_f1"] = sum(
        row["digit_8_f1"] == 0 for row in seed_rows)
    (output / "baseline-five-seeds-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    write_csv(output / "baseline-five-seeds-summary.csv", [{
        "metric": metric,
        **values,
    } for metric, values in summary.items() if isinstance(values, dict)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    seeds = [row["seed"] for row in seed_rows]
    axes[0].plot(seeds, [row["validation_accuracy"] for row in seed_rows],
                 marker="o", label="Accuracy")
    axes[0].plot(seeds, [row["validation_macro_f1"] for row in seed_rows],
                 marker="o", label="Macro-F1")
    axes[0].set(title="Desempeño en el mejor checkpoint", xlabel="Semilla",
                ylabel="Métrica", xticks=seeds, ylim=(0.8, 1.0))
    axes[0].legend()
    axes[1].plot(seeds, [row["digit_5_f1"] for row in seed_rows],
                 marker="o", label="F1 del 5")
    axes[1].plot(seeds, [row["digit_8_f1"] for row in seed_rows],
                 marker="o", label="F1 del 8")
    axes[1].set(title="Clases minoritarias", xlabel="Semilla", ylabel="F1",
                xticks=seeds, ylim=(-0.05, 1.0))
    axes[1].legend()
    fig.suptitle("Estabilidad del baseline con cinco semillas")
    fig.savefig(output / "baseline-five-seeds.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for seed, history in histories.items():
        epochs = [int(row["epoch"]) for row in history]
        values = [float(row["validation_macro_f1_present"]) for row in history]
        ax.plot(epochs, values, label=f"Semilla {seed}")
    ax.set(title="Macro-F1 de validation por época",
           xlabel="Época", ylabel="Macro-F1", ylim=(0, 1.02))
    ax.legend(ncol=2)
    fig.savefig(output / "baseline-five-seeds-curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
