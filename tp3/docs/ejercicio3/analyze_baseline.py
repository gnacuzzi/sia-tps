"""Generar figuras consolidadas del baseline del ejercicio 3."""

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


def run(input_dir, output):
    input_dir, output = Path(input_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = input_dir / "baseline-exercise2-new-data"
    with (run_dir / "history.csv").open(newline="") as file:
        history = list(csv.DictReader(file))
    metrics = json.loads((run_dir / "metrics.json").read_text())

    epochs = np.asarray([int(row["epoch"]) for row in history])
    training_mse = np.asarray([float(row["mse"]) for row in history])
    validation_mse = np.asarray([float(row["validation_mse"]) for row in history])
    training_f1 = np.asarray([
        float(row["training_macro_f1_present"]) for row in history])
    validation_f1 = np.asarray([
        float(row["validation_macro_f1_present"]) for row in history])
    best_epoch = int(metrics["best_epoch"])
    validation = metrics["validation"]
    training = metrics["training"]
    final = history[-1]
    write_csv(output / "baseline-summary.csv", [{
        "checkpoint": "selected_best",
        "epoch": best_epoch,
        "training_accuracy": training["accuracy"],
        "validation_accuracy": validation["accuracy"],
        "training_macro_f1": training["macro_f1"],
        "validation_macro_f1": validation["macro_f1"],
        "validation_mse": float(history[best_epoch]["validation_mse"]),
    }, {
        "checkpoint": "fixed_final",
        "epoch": int(final["epoch"]),
        "training_accuracy": float(final["training_accuracy"]),
        "validation_accuracy": float(final["validation_accuracy"]),
        "training_macro_f1": float(final["training_macro_f1_present"]),
        "validation_macro_f1": float(final["validation_macro_f1_present"]),
        "validation_mse": float(final["validation_mse"]),
    }])
    write_csv(output / "baseline-per-class.csv", [{
        "class": row["label"],
        "support": row["support"],
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
    } for row in validation["per_class"]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(epochs, training_mse, label="Training")
    axes[0].plot(epochs, validation_mse, label="Validation")
    axes[0].axvline(best_epoch, color="#9d3b27", linestyle="--",
                    label=f"Mejor época: {best_epoch}")
    axes[0].set(title="MSE por época", xlabel="Época", ylabel="MSE")
    axes[0].legend()
    axes[1].plot(epochs, training_f1, label="Training")
    axes[1].plot(epochs, validation_f1, label="Validation")
    axes[1].axvline(best_epoch, color="#9d3b27", linestyle="--",
                    label=f"Mejor época: {best_epoch}")
    axes[1].set(title="Macro-F1 por época", xlabel="Época", ylabel="Macro-F1",
                ylim=(0, 1.02))
    axes[1].legend()
    fig.suptitle("Baseline del ejercicio 2 sobre development ampliado")
    fig.savefig(output / "baseline-curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    per_class = validation["per_class"]
    labels = [row["label"] for row in per_class]
    f1 = [row["f1"] for row in per_class]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    bars = ax.bar(labels, f1, color=[
        "#d97925" if label in (5, 8) else "#2878a0" for label in labels])
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in f1], padding=3,
                 fontsize=8)
    ax.set(title=f"F1 por clase en validation · mejor época {best_epoch}",
           xlabel="Dígito", ylabel="F1", xticks=labels, ylim=(0, 1.08))
    fig.savefig(output / "baseline-f1-per-class.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    matrix = np.asarray(validation["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    for expected in labels:
        for predicted in labels:
            value = int(matrix[expected, predicted])
            if value:
                ax.text(predicted, expected, value, ha="center", va="center",
                        fontsize=7,
                        color="white" if value > matrix.max() * 0.5 else "black")
    ax.set(title="Matriz de confusión de validation",
           xlabel="Predicción", ylabel="Clase real", xticks=labels, yticks=labels)
    fig.colorbar(image, ax=ax, label="Imágenes")
    fig.savefig(output / "baseline-confusion-matrix.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
