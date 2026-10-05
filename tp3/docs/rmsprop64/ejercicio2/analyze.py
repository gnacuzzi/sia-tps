"""Consolidar la elección de épocas y la evaluación final de RMSProp-64."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ADAM_SUMMARY = HERE.parents[1] / "ejercicio2" / "results" / "analysis-01-02" / "final-test-summary.csv"
ADAM_PER_CLASS = HERE.parents[1] / "ejercicio2" / "results" / "analysis-01-02" / "final-test-per-class.csv"


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_history(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def column(history, name):
    return np.array([float(row[name]) for row in history])


def epoch_selection(results, output):
    histories = [read_history(folder / "arch-64-rmsprop-lr-001" / "history.csv")
                 for folder in sorted(results.glob("01?-cross-validation-fold-*"))]
    validation = np.array([column(h, "validation_macro_f1_present") for h in histories])
    training = np.array([column(h, "training_macro_f1_present") for h in histories])
    mse = np.array([column(h, "validation_mse") for h in histories])
    mean = validation.mean(axis=0)
    smooth = np.convolve(mean, np.ones(11) / 11, mode="valid")
    chosen = int(smooth.argmax() + 5)
    rows = [{
        "epoch": epoch,
        "validation_macro_f1_mean": mean[epoch],
        "validation_macro_f1_sd": validation[:, epoch].std(ddof=1),
        "validation_mse_mean": mse[:, epoch].mean(),
        "gap_mean": (training[:, epoch] - validation[:, epoch]).mean(),
    } for epoch in range(len(mean))]
    write_csv(output / "epoch-selection.csv", rows)

    figure, axis = plt.subplots(figsize=(8, 4.2))
    epochs = np.arange(len(mean))
    for fold in validation:
        axis.plot(epochs, fold, color="#9FB3C8", lw=0.8, alpha=0.7)
    axis.plot(epochs, mean, color="#1F5D8F", lw=2, label="Media de los 5 folds")
    axis.fill_between(epochs, mean - validation.std(axis=0, ddof=1),
                      mean + validation.std(axis=0, ddof=1), color="#1F5D8F", alpha=0.12)
    axis.axvline(chosen, color="#E67E22", ls="--", lw=1.5,
                 label=f"Época elegida: {chosen}")
    axis.set_ylim(0.90, 0.97)
    axis.set_xlabel("Época")
    axis.set_ylabel("Macro-F1 de validation")
    axis.set_title("RMSProp-64 · 5-fold de digits.csv")
    axis.legend(loc="lower right", frameon=False)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "epoch-selection.png", dpi=180)
    plt.close(figure)
    return chosen, mean


def final_evaluation(results, output):
    folder = results / "02-final-rmsprop-64"
    summary = json.loads((folder / "summary.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    with ADAM_SUMMARY.open(newline="") as file:
        adam = next(csv.DictReader(file))
    rows = [
        {"model": "Adam-128 (ejercicio 2 original)", "epochs": 200,
         "parameters": 101770, "test_accuracy": float(adam["test_accuracy"]),
         "test_macro_f1": float(adam["test_macro_f1"]),
         "test_digit_5_f1": float(adam["test_digit_5_f1"]),
         "test_digit_8_f1": 0.0, "test_mse": float(adam["test_mse"]),
         "training_seconds": float(adam["training_seconds"]),
         "test_accuracy_without_8": float(adam["test_accuracy_without_8"]),
         "test_macro_f1_without_8": float(adam["test_macro_f1_without_8"])},
        {"model": "RMSProp-64", "epochs": metrics["epochs"],
         "parameters": metrics["parameter_count"],
         "test_accuracy": summary["test_accuracy"],
         "test_macro_f1": summary["test_macro_f1"],
         "test_digit_5_f1": summary["test_digit_5_f1"],
         "test_digit_8_f1": summary["test_digit_8_f1"],
         "test_mse": summary["test_mse"],
         "training_seconds": summary["training_seconds"]},
    ]
    per_class = metrics["test"]["per_class"]
    without_8 = [item for item in per_class if item["label"] != 8]
    confusion = np.array(metrics["test"]["confusion_matrix"])
    keep = [label for label in range(10) if label != 8]
    rows[1]["test_accuracy_without_8"] = (
        np.trace(confusion[np.ix_(keep, keep)]) / confusion[keep].sum())
    rows[1]["test_macro_f1_without_8"] = float(np.mean([item["f1"] for item in without_8]))
    write_csv(output / "final-test-summary.csv", rows)

    with ADAM_PER_CLASS.open(newline="") as file:
        adam_f1 = {int(row["label"]): float(row["f1"]) for row in csv.DictReader(file)}
    write_csv(output / "final-test-per-class.csv", [{
        "label": item["label"], "support": item["support"],
        "predicted": item["predicted"], "precision": item["precision"],
        "recall": item["recall"], "f1": item["f1"],
        "f1_adam_128": adam_f1[item["label"]],
    } for item in per_class])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    image = axes[0].imshow(confusion, cmap="Blues")
    for i in range(10):
        for j in range(10):
            value = confusion[i, j]
            axes[0].text(j, i, value, ha="center", va="center", fontsize=7,
                         color="white" if value > confusion.max() / 2 else "#102A43")
    axes[0].set_xticks(range(10))
    axes[0].set_yticks(range(10))
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Clase real")
    axes[0].set_title("Matriz de confusión en test")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    labels = np.arange(10)
    width = 0.38
    axes[1].bar(labels - width / 2, [adam_f1[label] for label in labels], width,
                color="#9FB3C8", label="Adam-128")
    axes[1].bar(labels + width / 2, [item["f1"] for item in per_class], width,
                color="#1F5D8F", label="RMSProp-64")
    axes[1].set_xticks(labels)
    axes[1].set_xlabel("Dígito")
    axes[1].set_ylabel("F1 en test")
    axes[1].set_title("F1 por dígito")
    axes[1].legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis="y", alpha=0.3)
    figure.suptitle("Evaluación única sobre digits_test.csv · RMSProp-64")
    figure.tight_layout()
    figure.savefig(output / "final-test-confusion-and-f1.png", dpi=180)
    plt.close(figure)

    history = read_history(folder / "training-history.csv")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = column(history, "epoch")
    axes[0].semilogy(epochs, column(history, "mse"), color="#1F5D8F")
    axes[0].set_title("MSE de training")
    axes[1].plot(epochs, column(history, "training_macro_f1_present"), color="#1F5D8F")
    axes[1].set_title("Macro-F1 de training")
    for axis in axes:
        axis.set_xlabel("Época")
        axis.grid(alpha=0.3)
    figure.suptitle("Entrenamiento final con las 12.449 imágenes de digits.csv")
    figure.tight_layout()
    figure.savefig(output / "final-training-curves.png", dpi=180)
    plt.close(figure)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "analysis")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    chosen, mean = epoch_selection(args.results, args.output)
    print(f"época elegida {chosen}: macro-F1 medio {mean[chosen]:.4f}; "
          f"época 200: {mean[200]:.4f}")
    for row in final_evaluation(args.results, args.output):
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
