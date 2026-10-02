"""Resumir rotaciones pequeñas en época fija y compararlas con el control."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sia_tp3.experiments import load_digits_development_split
from sia_tp3.metrics import classification_metrics
from sia_tp3.models import MultilayerPerceptron


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_history(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def run(search_dir, rotation_dir, control_dir, control_csv, data,
        additional_data, output):
    search_dir, rotation_dir, control_dir, control_csv, output = map(
        Path, (search_dir, rotation_dir, control_dir, control_csv, output))
    output.mkdir(parents=True, exist_ok=True)
    split = load_digits_development_split(
        data, validation_fraction=0.2, validation_seed=0,
        additional_train_path=additional_data, deduplicate_inputs=True)

    search_rows = []
    for name in ("rotation-control", "rotation-2deg-p-05",
                 "rotation-4deg-p-05", "rotation-6deg-p-05"):
        final = load_history(search_dir / name / "history.csv")[300]
        search_rows.append({
            "candidate": name,
            "seed": 0,
            "epoch": 300,
            "validation_accuracy": float(final["validation_accuracy"]),
            "validation_macro_f1": float(final["validation_macro_f1_present"]),
            "validation_loss": float(final["validation_loss"]),
        })
    write_csv(output / "rotation-search-fixed-epoch-300.csv", search_rows)

    rotation_rows, tail_rows = [], []
    histories = {}
    for seed in range(5):
        folder = rotation_dir / f"rotation-4deg-p-05-seed-{seed}"
        history = load_history(folder / "history.csv")
        histories[seed] = history
        final = history[300]
        model = MultilayerPerceptron.load(folder / "model-final.npz")
        predicted = np.argmax(model.predict(split.X_validation), axis=1)
        report = classification_metrics(
            split.y_validation, predicted, labels=tuple(range(10)))
        rotation_rows.append({
            "candidate": "rotation-4deg-p-05",
            "seed": seed,
            "epoch": 300,
            "validation_accuracy": float(final["validation_accuracy"]),
            "validation_macro_f1": float(final["validation_macro_f1_present"]),
            "validation_loss": float(final["validation_loss"]),
            "validation_f1_5": report.for_label(5).f1,
            "validation_f1_8": report.for_label(8).f1,
        })
        tail = history[251:301]
        tail_rows.append({
            "seed": seed,
            **{
                f"{metric}_mean_last_50": statistics.mean(
                    float(row[field]) for row in tail)
                for metric, field in (
                    ("accuracy", "validation_accuracy"),
                    ("macro_f1", "validation_macro_f1_present"),
                    ("loss", "validation_loss"),
                )
            },
            **{
                f"{metric}_sd_last_50": statistics.stdev(
                    float(row[field]) for row in tail)
                for metric, field in (
                    ("accuracy", "validation_accuracy"),
                    ("macro_f1", "validation_macro_f1_present"),
                    ("loss", "validation_loss"),
                )
            },
        })
    write_csv(output / "rotation-fixed-epoch-300.csv", rotation_rows)
    write_csv(output / "rotation-last-50-stability.csv", tail_rows)

    with control_csv.open(newline="") as file:
        control_rows = [row for row in csv.DictReader(file)
                        if row["candidate"] == "step-decay"]
    for seed, row in enumerate(control_rows):
        model = MultilayerPerceptron.load(
            control_dir / f"translation-step-decay-seed-{seed}" / "model-final.npz")
        predicted = np.argmax(model.predict(split.X_validation), axis=1)
        report = classification_metrics(
            split.y_validation, predicted, labels=tuple(range(10)))
        row["validation_f1_5"] = report.for_label(5).f1
        row["validation_f1_8"] = report.for_label(8).f1
    summary = []
    for candidate, rows in (("translation-step-decay", control_rows),
                            ("rotation-4deg-p-05", rotation_rows)):
        summary.append({
            "candidate": candidate,
            **{
                f"{metric}_mean": statistics.mean(float(row[field]) for row in rows)
                for metric, field in (
                    ("validation_accuracy", "validation_accuracy"),
                    ("validation_macro_f1", "validation_macro_f1_present"
                     if candidate == "translation-step-decay" else "validation_macro_f1"),
                    ("validation_loss", "validation_loss"),
                )
            },
            **{
                f"{metric}_sd": statistics.stdev(float(row[field]) for row in rows)
                for metric, field in (
                    ("validation_accuracy", "validation_accuracy"),
                    ("validation_macro_f1", "validation_macro_f1_present"
                     if candidate == "translation-step-decay" else "validation_macro_f1"),
                    ("validation_loss", "validation_loss"),
                )
            },
        })
    for item, rows in zip(summary, (control_rows, rotation_rows)):
        item.update({
            "validation_f1_5_mean": statistics.mean(
                float(row["validation_f1_5"]) for row in rows),
            "validation_f1_5_sd": statistics.stdev(
                float(row["validation_f1_5"]) for row in rows),
            "validation_f1_8_mean": statistics.mean(
                float(row["validation_f1_8"]) for row in rows),
            "validation_f1_8_sd": statistics.stdev(
                float(row["validation_f1_8"]) for row in rows),
        })
    summary[1].update({
        "accuracy_ge_098_count": sum(
            row["validation_accuracy"] >= 0.98 for row in rotation_rows),
        "last_50_accuracy_sd_mean": statistics.mean(
            row["accuracy_sd_last_50"] for row in tail_rows),
        "last_50_macro_f1_sd_mean": statistics.mean(
            row["macro_f1_sd_last_50"] for row in tail_rows),
        "last_50_loss_sd_mean": statistics.mean(
            row["loss_sd_last_50"] for row in tail_rows),
    })
    write_csv(output / "rotation-comparison-summary.csv", summary)
    (output / "rotation-comparison-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    epochs = np.arange(301)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, field, title in (
            (axes[0], "validation_accuracy", "Accuracy de validation"),
            (axes[1], "validation_macro_f1_present", "Macro-F1 de validation")):
        matrix = np.array([[float(row[field]) for row in histories[seed]]
                           for seed in range(5)])
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        axis.plot(epochs, mean, label="Traslación + rotación ±4°")
        axis.fill_between(epochs, mean - sd, mean + sd, alpha=0.18)
        axis.axvline(150, color="black", linestyle="--", alpha=0.45)
        axis.axvline(220, color="black", linestyle=":", alpha=0.55)
        axis.set(title=title, xlabel="Época", ylabel=title, xlim=(100, 300))
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylim(0.96, 0.99)
    axes[1].set_ylim(0.95, 0.985)
    fig.suptitle("Rotaciones pequeñas · media ± sd de cinco semillas")
    fig.savefig(output / "rotation-stability-curves.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True)
    parser.add_argument("--rotation", required=True)
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--control-csv", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--additional-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.search, args.rotation, args.control_dir, args.control_csv, args.data,
        args.additional_data, args.output)
