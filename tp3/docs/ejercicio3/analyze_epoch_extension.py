"""Comparar 200 contra 300 épocas con el augmentation seleccionado."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def class_f1(metrics, label):
    return next(item["f1"] for item in metrics["validation"]["per_class"]
                if item["label"] == label)


def load_rows(input_dir, prefix, maximum_epochs):
    rows, histories = [], {}
    for seed in range(5):
        folder = input_dir / f"{prefix}{seed}"
        metrics = json.loads((folder / "metrics.json").read_text())
        with (folder / "history.csv").open(newline="") as file:
            histories[seed] = list(csv.DictReader(file))
        rows.append({
            "maximum_epochs": maximum_epochs,
            "seed": seed,
            "best_epoch": metrics["best_epoch"],
            "validation_accuracy": metrics["validation"]["accuracy"],
            "validation_macro_f1": metrics["validation"]["macro_f1"],
            "digit_5_f1": class_f1(metrics, 5),
            "digit_8_f1": class_f1(metrics, 8),
        })
    return rows, histories


def run(epochs_200, epochs_300, output):
    epochs_200, epochs_300, output = map(Path, (epochs_200, epochs_300, output))
    output.mkdir(parents=True, exist_ok=True)
    rows_200, histories_200 = load_rows(
        epochs_200, "translation-shift-1-p-05-seed-", 200)
    rows_300, histories_300 = load_rows(
        epochs_300, "translation-shift-1-p-05-300-seed-", 300)
    rows = rows_200 + rows_300
    write_csv(output / "epoch-extension-five-seeds.csv", rows)

    history_fields = (
        "loss", "validation_loss", "mse", "validation_mse",
        "training_macro_f1_present", "validation_macro_f1_present",
    )
    for seed in range(5):
        assert len(histories_200[seed]) == 201
        for short, extended in zip(histories_200[seed], histories_300[seed][:201]):
            assert short["epoch"] == extended["epoch"]
            assert all(short[field] == extended[field] for field in history_fields)

    metric_names = ("validation_accuracy", "validation_macro_f1",
                    "digit_5_f1", "digit_8_f1")
    summary_rows = []
    for maximum_epochs in (200, 300):
        selected = [row for row in rows
                    if row["maximum_epochs"] == maximum_epochs]
        summary = {"maximum_epochs": maximum_epochs}
        for metric in metric_names:
            values = [row[metric] for row in selected]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
            summary[f"{metric}_minimum"] = min(values)
            summary[f"{metric}_maximum"] = max(values)
        summary_rows.append(summary)
    write_csv(output / "epoch-extension-summary.csv", summary_rows)
    (output / "epoch-extension-summary.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n")

    paired = []
    for seed in range(5):
        short = rows_200[seed]
        extended = rows_300[seed]
        paired.append({
            "seed": seed,
            "best_epoch_200": short["best_epoch"],
            "best_epoch_300": extended["best_epoch"],
            **{f"delta_{metric}": extended[metric] - short[metric]
               for metric in metric_names},
        })
    write_csv(output / "epoch-extension-paired-differences.csv", paired)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for seed in range(5):
        history = histories_300[seed]
        epochs = [int(row["epoch"]) for row in history]
        axes[0].plot(epochs,
                     [float(row["validation_macro_f1_present"])
                      for row in history], label=f"Semilla {seed}")
        axes[1].plot(epochs,
                     [float(row["validation_loss"]) for row in history],
                     label=f"Semilla {seed}")
    axes[0].set(title="Macro-F1 de validation", xlabel="Época",
                ylabel="Macro-F1", ylim=(0.93, 0.98))
    axes[1].set(title="Entropía cruzada de validation", xlabel="Época",
                ylabel="Cross-entropy")
    for axis in axes:
        axis.axvline(200, color="black", linestyle="--", alpha=0.6,
                     label="Límite anterior")
        axis.legend(ncol=2, fontsize=8)
    fig.suptitle("Convergencia con augmentation hasta 300 épocas")
    fig.savefig(output / "epoch-extension-curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs-200", type=Path, required=True)
    parser.add_argument("--epochs-300", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.epochs_200, args.epochs_300, args.output)
