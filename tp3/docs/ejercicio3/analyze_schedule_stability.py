"""Comparar tasa constante y step decay sin seleccionar picos aislados."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_candidate(input_dir, candidate, prefix):
    histories, best_rows = {}, []
    for seed in range(5):
        folder = input_dir / f"{prefix}{seed}"
        with (folder / "history.csv").open(newline="") as file:
            histories[seed] = list(csv.DictReader(file))
        metrics = json.loads((folder / "metrics.json").read_text())
        best_rows.append({
            "candidate": candidate,
            "seed": seed,
            "best_epoch": metrics["best_epoch"],
            "best_validation_accuracy": metrics["validation"]["accuracy"],
            "best_validation_macro_f1": metrics["validation"]["macro_f1"],
        })
    return histories, best_rows


def run(constant_dir, schedule_dir, output):
    constant_dir, schedule_dir, output = map(
        Path, (constant_dir, schedule_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    constant, constant_best = load_candidate(
        constant_dir, "constant-0.01", "translation-shift-1-p-05-300-seed-")
    schedule, schedule_best = load_candidate(
        schedule_dir, "step-decay", "translation-step-decay-seed-")

    comparison_fields = (
        "loss", "validation_loss", "mse", "validation_mse",
        "training_macro_f1_present", "validation_accuracy",
        "validation_macro_f1_present",
    )
    for seed in range(5):
        for first, second in zip(constant[seed][:151], schedule[seed][:151]):
            assert first["epoch"] == second["epoch"]
            assert all(first[field] == second[field]
                       for field in comparison_fields)

    endpoint_rows, tail_rows = [], []
    metric_names = (
        "validation_accuracy", "validation_macro_f1_present", "validation_loss")
    for candidate, histories in (
            ("constant-0.01", constant), ("step-decay", schedule)):
        for seed in range(5):
            final = histories[seed][300]
            endpoint_rows.append({
                "candidate": candidate,
                "seed": seed,
                "epoch": 300,
                **{metric: float(final[metric]) for metric in metric_names},
            })
            tail = histories[seed][251:301]
            tail_row = {"candidate": candidate, "seed": seed}
            for metric in metric_names:
                values = [float(row[metric]) for row in tail]
                tail_row[f"{metric}_mean_last_50"] = statistics.mean(values)
                tail_row[f"{metric}_sd_last_50"] = statistics.stdev(values)
                tail_row[f"{metric}_minimum_last_50"] = min(values)
                tail_row[f"{metric}_maximum_last_50"] = max(values)
            tail_rows.append(tail_row)
    write_csv(output / "schedule-fixed-epoch-300.csv", endpoint_rows)
    write_csv(output / "schedule-last-50-stability.csv", tail_rows)

    summary_rows = []
    for candidate in ("constant-0.01", "step-decay"):
        endpoints = [row for row in endpoint_rows if row["candidate"] == candidate]
        tails = [row for row in tail_rows if row["candidate"] == candidate]
        summary = {"candidate": candidate}
        for metric in metric_names:
            values = [row[metric] for row in endpoints]
            summary[f"epoch_300_{metric}_mean"] = statistics.mean(values)
            summary[f"epoch_300_{metric}_sd"] = statistics.stdev(values)
            summary[f"last_50_{metric}_mean"] = statistics.mean(
                row[f"{metric}_mean_last_50"] for row in tails)
            summary[f"last_50_mean_within_seed_sd_{metric}"] = statistics.mean(
                row[f"{metric}_sd_last_50"] for row in tails)
        summary_rows.append(summary)
    write_csv(output / "schedule-stability-summary.csv", summary_rows)
    (output / "schedule-stability-summary.json").write_text(
        json.dumps(summary_rows, indent=2) + "\n")

    best_rows = constant_best + schedule_best
    write_csv(output / "schedule-best-checkpoints-diagnostic.csv", best_rows)

    epochs = np.arange(301)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for candidate, histories in (
            ("Tasa constante", constant), ("Step decay", schedule)):
        for axis, metric, title in (
                (axes[0], "validation_accuracy", "Accuracy de validation"),
                (axes[1], "validation_macro_f1_present", "Macro-F1 de validation")):
            matrix = np.array([
                [float(row[metric]) for row in histories[seed]]
                for seed in range(5)])
            mean = matrix.mean(axis=0)
            sd = matrix.std(axis=0, ddof=1)
            axis.plot(epochs, mean, label=candidate)
            axis.fill_between(epochs, mean - sd, mean + sd, alpha=0.16)
            axis.set(title=title, xlabel="Época", ylabel=title,
                     xlim=(100, 300))
    for axis in axes:
        axis.axvline(150, color="black", linestyle="--", alpha=0.45)
        axis.axvline(220, color="black", linestyle=":", alpha=0.55)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylim(0.95, 0.99)
    axes[1].set_ylim(0.94, 0.985)
    fig.suptitle("Learning rate constante frente a step decay · media ± sd")
    fig.savefig(output / "schedule-stability-curves.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.constant, args.schedule, args.output)
