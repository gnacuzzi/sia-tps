"""Auditar el candidato congelado con predicciones out-of-fold."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sia_tp3.experiments import load_digits_development_fold
from sia_tp3.metrics import classification_metrics
from sia_tp3.models import MultilayerPerceptron


LABELS = tuple(range(10))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(input_dir, data, additional_data, output):
    input_dir, output = Path(input_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fold_rows, histories = [], []
    all_indices, all_expected, all_predicted = [], [], []
    for fold_index in range(5):
        split = load_digits_development_fold(
            data, fold_count=5, fold_index=fold_index, fold_seed=0,
            additional_train_path=additional_data, deduplicate_inputs=True)
        folder = (input_dir / f"fold-{fold_index}"
                  / f"frozen-candidate-fold-{fold_index}")
        with (folder / "history.csv").open(newline="") as file:
            history = list(csv.DictReader(file))
        histories.append(history)
        final = history[300]
        model = MultilayerPerceptron.load(folder / "model-final.npz")
        predicted = np.argmax(model.predict(split.X_validation), axis=1)
        report = classification_metrics(
            split.y_validation, predicted, labels=LABELS)
        fold_rows.append({
            "fold": fold_index,
            "seed": fold_index,
            "validation_size": len(split.y_validation),
            "epoch": 300,
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "f1_5": report.for_label(5).f1,
            "f1_8": report.for_label(8).f1,
            "validation_loss": float(final["validation_loss"]),
        })
        all_indices.append(split.validation_indices)
        all_expected.append(split.y_validation)
        all_predicted.append(predicted)
    write_csv(output / "kfold-fixed-epoch-300.csv", fold_rows)

    indices = np.concatenate(all_indices)
    if len(indices) != 24501 or len(np.unique(indices)) != 24501:
        raise AssertionError("los folds no son una partición del development")
    expected = np.concatenate(all_expected)
    predicted = np.concatenate(all_predicted)
    pooled = classification_metrics(expected, predicted, labels=LABELS)
    summary = {
        "protocol": "5-fold paired with seeds 0-4",
        "fixed_epoch": 300,
        "development_predictions": len(expected),
        "unique_validation_indices": len(np.unique(indices)),
        "fold_accuracy_mean": statistics.mean(row["accuracy"] for row in fold_rows),
        "fold_accuracy_sd": statistics.stdev(row["accuracy"] for row in fold_rows),
        "fold_macro_f1_mean": statistics.mean(row["macro_f1"] for row in fold_rows),
        "fold_macro_f1_sd": statistics.stdev(row["macro_f1"] for row in fold_rows),
        "fold_f1_5_mean": statistics.mean(row["f1_5"] for row in fold_rows),
        "fold_f1_5_sd": statistics.stdev(row["f1_5"] for row in fold_rows),
        "fold_f1_8_mean": statistics.mean(row["f1_8"] for row in fold_rows),
        "fold_f1_8_sd": statistics.stdev(row["f1_8"] for row in fold_rows),
        "folds_accuracy_ge_098": sum(row["accuracy"] >= 0.98 for row in fold_rows),
        "out_of_fold_accuracy": pooled.accuracy,
        "out_of_fold_macro_f1": pooled.macro_f1,
        "out_of_fold_f1_5": pooled.for_label(5).f1,
        "out_of_fold_f1_8": pooled.for_label(8).f1,
        "out_of_fold_confusion_matrix": pooled.confusion_matrix.tolist(),
        "test_opened": False,
    }
    (output / "kfold-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    epochs = np.arange(301)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for axis, field, title in (
            (axes[0], "validation_accuracy", "Accuracy de validation"),
            (axes[1], "validation_macro_f1_present", "Macro-F1 de validation")):
        matrix = np.array([[float(row[field]) for row in history]
                           for history in histories])
        mean, sd = matrix.mean(axis=0), matrix.std(axis=0, ddof=1)
        axis.plot(epochs, mean, label="Media de los cinco folds")
        axis.fill_between(epochs, mean - sd, mean + sd, alpha=0.18)
        axis.axvline(150, color="black", linestyle="--", alpha=0.45)
        axis.axvline(220, color="black", linestyle=":", alpha=0.55)
        axis.set(title=title, xlabel="Época", ylabel=title, xlim=(100, 300))
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylim(0.96, 0.99)
    axes[1].set_ylim(0.95, 0.985)
    fig.suptitle("Confirmación 5-fold · candidato congelado, media ± sd")
    fig.savefig(output / "kfold-stability-curves.png", dpi=170,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--additional-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.input, args.data, args.additional_data, args.output)
