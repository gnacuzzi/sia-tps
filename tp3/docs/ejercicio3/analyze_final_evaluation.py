"""Generar artefactos de la única evaluación final del ejercicio 3."""

import argparse
import csv
import json
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


def run(final_dir, cv_summary_path, exercise2_dir, output):
    final_dir, exercise2_dir, output = map(
        Path, (final_dir, exercise2_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((final_dir / "metrics.json").read_text())
    summary = json.loads((final_dir / "summary.json").read_text())
    source = json.loads((final_dir / "data-source.json").read_text())
    cv = json.loads(Path(cv_summary_path).read_text())
    exercise2 = json.loads((exercise2_dir / "summary.json").read_text())

    test = metrics["test"]
    per_class = [{
        "label": row["label"],
        "support": row["support"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "false_positive": row["false_positive"],
        "false_negative": row["false_negative"],
    } for row in test["per_class"]]
    write_csv(output / "final-test-per-class.csv", per_class)

    comparison = [
        {
            "evaluation": "exercise-2-test",
            "accuracy": exercise2["test_accuracy"],
            "macro_f1": exercise2["test_macro_f1"],
            "f1_5": exercise2["test_digit_5_f1"],
            "f1_8": None,
        },
        {
            "evaluation": "exercise-3-out-of-fold",
            "accuracy": cv["out_of_fold_accuracy"],
            "macro_f1": cv["out_of_fold_macro_f1"],
            "f1_5": cv["out_of_fold_f1_5"],
            "f1_8": cv["out_of_fold_f1_8"],
        },
        {
            "evaluation": "exercise-3-test",
            "accuracy": summary["test_accuracy"],
            "macro_f1": summary["test_macro_f1"],
            "f1_5": summary["test_digit_5_f1"],
            "f1_8": summary["test_digit_8_f1"],
        },
    ]
    write_csv(output / "final-comparison.csv", comparison)

    final_summary = {
        **summary,
        "test_errors": int(summary["test_samples"]
                           - np.trace(test["confusion_matrix"])),
        "accuracy_target": 0.98,
        "accuracy_target_reached": summary["test_accuracy"] >= 0.98,
        "distance_to_target_percentage_points": (
            summary["test_accuracy"] - 0.98) * 100,
        "out_of_fold_accuracy": cv["out_of_fold_accuracy"],
        "test_minus_out_of_fold_percentage_points": (
            summary["test_accuracy"] - cv["out_of_fold_accuracy"]) * 100,
        "test_evaluations": source["test_evaluations"],
        "test_sha256": source["test_sha256"],
    }
    (output / "final-summary.json").write_text(
        json.dumps(final_summary, indent=2) + "\n")

    with (final_dir / "training-history.csv").open(newline="") as file:
        history = list(csv.DictReader(file))
    epochs = np.array([int(row["epoch"]) for row in history])
    loss = np.array([float(row["loss"]) for row in history])
    macro_f1 = np.array([
        float(row["training_macro_f1_present"]) for row in history])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(epochs, loss)
    axes[0].set_yscale("log")
    axes[0].set(title="Entropía cruzada sobre development",
                xlabel="Época", ylabel="Cross-entropy")
    axes[1].plot(epochs, macro_f1)
    axes[1].set(title="Macro-F1 sobre development limpio",
                xlabel="Época", ylabel="Macro-F1", ylim=(0, 1.002))
    axes[2].plot(epochs, macro_f1)
    axes[2].set(title="Macro-F1 · detalle de la meseta",
                xlabel="Época", ylabel="Macro-F1", xlim=(150, 300),
                ylim=(0.985, 1.001))
    for axis in axes:
        axis.axvline(150, color="black", linestyle="--", alpha=0.45)
        axis.axvline(220, color="black", linestyle=":", alpha=0.55)
        axis.grid(alpha=0.2)
    fig.suptitle("Entrenamiento final con las 24.501 imágenes únicas")
    fig.savefig(output / "final-training-curves.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)

    matrix = np.asarray(test["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(10):
        for column in range(10):
            value = matrix[row, column]
            axis.text(column, row, str(value), ha="center", va="center",
                      color="white" if value > matrix.max() / 2 else "black",
                      fontsize=8)
    axis.set(xticks=range(10), yticks=range(10),
             xlabel="Predicción", ylabel="Clase real",
             title="Matriz de confusión · test final")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(output / "final-test-confusion-matrix.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", required=True)
    parser.add_argument("--cv-summary", required=True)
    parser.add_argument("--exercise2", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.final, args.cv_summary, args.exercise2, args.output)
