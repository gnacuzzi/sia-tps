"""Comparar augmentation por traslaciones contra el control en cinco semillas."""

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


def read_candidate(input_dir, candidate, prefix):
    rows = []
    for seed in range(5):
        metrics = json.loads(
            (input_dir / f"{prefix}{seed}" / "metrics.json").read_text())
        rows.append({
            "candidate": candidate,
            "seed": seed,
            "best_epoch": metrics["best_epoch"],
            "validation_accuracy": metrics["validation"]["accuracy"],
            "validation_macro_f1": metrics["validation"]["macro_f1"],
            "digit_5_f1": class_f1(metrics, 5),
            "digit_8_f1": class_f1(metrics, 8),
        })
    return rows


def run(control_dir, augmentation_dir, output):
    control_dir, augmentation_dir, output = map(
        Path, (control_dir, augmentation_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    rows = read_candidate(
        control_dir, "control", "softmax-ce-lr-001-seed-")
    rows += read_candidate(
        augmentation_dir, "translation-1-p-05",
        "translation-shift-1-p-05-seed-")
    write_csv(output / "translation-five-seeds.csv", rows)

    metrics_names = ("validation_accuracy", "validation_macro_f1",
                     "digit_5_f1", "digit_8_f1")
    summary_rows = []
    for candidate in ("control", "translation-1-p-05"):
        selected = [row for row in rows if row["candidate"] == candidate]
        summary = {"candidate": candidate}
        for metric in metrics_names:
            values = [row[metric] for row in selected]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
            summary[f"{metric}_minimum"] = min(values)
            summary[f"{metric}_maximum"] = max(values)
        summary_rows.append(summary)
    write_csv(output / "translation-five-seeds-summary.csv", summary_rows)
    (output / "translation-five-seeds-summary.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n")

    paired = []
    for seed in range(5):
        control = next(row for row in rows
                       if row["candidate"] == "control" and row["seed"] == seed)
        augmented = next(row for row in rows
                         if row["candidate"] == "translation-1-p-05"
                         and row["seed"] == seed)
        paired.append({"seed": seed, **{
            f"delta_{metric}": augmented[metric] - control[metric]
            for metric in metrics_names
        }})
    write_csv(output / "translation-paired-differences.csv", paired)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    plot_metrics = (("validation_accuracy", "Accuracy"),
                    ("validation_macro_f1", "Macro-F1"),
                    ("digit_8_f1", "F1 del 8"))
    for axis, (metric, title) in zip(axes, plot_metrics):
        for candidate in ("control", "translation-1-p-05"):
            selected = [row for row in rows if row["candidate"] == candidate]
            axis.plot([row["seed"] for row in selected],
                      [row[metric] for row in selected], marker="o",
                      label=candidate)
        axis.set(title=title, xlabel="Semilla", ylabel=title,
                 xticks=range(5))
        axis.legend()
    fig.suptitle("Augmentation por traslaciones frente al control")
    fig.savefig(output / "translation-five-seeds.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--augmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.control, args.augmentation, args.output)
