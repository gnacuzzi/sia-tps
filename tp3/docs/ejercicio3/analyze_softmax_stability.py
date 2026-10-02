"""Consolidar las cinco semillas de los learning rates finalistas."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RATES = (("softmax-ce-lr-0003", 0.003), ("softmax-ce-lr-001", 0.01))
SEEDS = range(5)


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
    rows, histories = [], {}
    for prefix, rate in RATES:
        for seed in SEEDS:
            name = f"{prefix}-seed-{seed}"
            run_dir = input_dir / name
            metrics = json.loads((run_dir / "metrics.json").read_text())
            with (run_dir / "history.csv").open(newline="") as file:
                histories[(rate, seed)] = list(csv.DictReader(file))
            rows.append({
                "learning_rate": rate,
                "seed": seed,
                "best_epoch": metrics["best_epoch"],
                "validation_accuracy": metrics["validation"]["accuracy"],
                "validation_macro_f1": metrics["validation"]["macro_f1"],
                "digit_5_precision": class_metric(metrics, 5, "precision"),
                "digit_5_recall": class_metric(metrics, 5, "recall"),
                "digit_5_f1": class_metric(metrics, 5, "f1"),
                "digit_8_precision": class_metric(metrics, 8, "precision"),
                "digit_8_recall": class_metric(metrics, 8, "recall"),
                "digit_8_f1": class_metric(metrics, 8, "f1"),
            })
    write_csv(output / "softmax-finalists-five-seeds.csv", rows)

    summary_rows = []
    for _, rate in RATES:
        rate_rows = [row for row in rows if row["learning_rate"] == rate]
        summary = {"learning_rate": rate}
        for metric in ("best_epoch", "validation_accuracy",
                       "validation_macro_f1", "digit_5_f1", "digit_8_f1"):
            values = [row[metric] for row in rate_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
            summary[f"{metric}_minimum"] = min(values)
            summary[f"{metric}_maximum"] = max(values)
        summary["seeds_with_zero_digit_8_f1"] = sum(
            row["digit_8_f1"] == 0 for row in rate_rows)
        summary_rows.append(summary)
    write_csv(output / "softmax-finalists-summary.csv", summary_rows)
    (output / "softmax-finalists-summary.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for _, rate in RATES:
        rate_rows = [row for row in rows if row["learning_rate"] == rate]
        seeds = [row["seed"] for row in rate_rows]
        label = f"eta={rate:g}"
        axes[0].plot(seeds, [row["validation_macro_f1"] for row in rate_rows],
                     marker="o", label=label)
        axes[1].plot(seeds, [row["digit_8_f1"] for row in rate_rows],
                     marker="o", label=label)
    axes[0].set(title="Macro-F1 en el mejor checkpoint", xlabel="Semilla",
                ylabel="Macro-F1", xticks=list(SEEDS), ylim=(0.9, 1.0))
    axes[1].set(title="F1 del 8 en el mejor checkpoint", xlabel="Semilla",
                ylabel="F1", xticks=list(SEEDS), ylim=(0.75, 1.0))
    for axis in axes:
        axis.legend()
    fig.suptitle("Estabilidad de softmax + entropía cruzada")
    fig.savefig(output / "softmax-finalists-five-seeds.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for _, rate in RATES:
        for seed in SEEDS:
            history = histories[(rate, seed)]
            epochs = [int(row["epoch"]) for row in history]
            axes[0].plot(epochs,
                         [float(row["validation_macro_f1_present"])
                          for row in history], alpha=0.75,
                         label=f"eta={rate:g}, semilla={seed}")
            axes[1].plot(epochs,
                         [float(row["validation_loss"]) for row in history],
                         alpha=0.75)
    axes[0].set(title="Macro-F1 de validation", xlabel="Época",
                ylabel="Macro-F1", ylim=(0.8, 1.0))
    axes[1].set(title="Entropía cruzada de validation", xlabel="Época",
                ylabel="Cross-entropy")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Curvas de las diez corridas finalistas")
    fig.savefig(output / "softmax-finalists-curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
