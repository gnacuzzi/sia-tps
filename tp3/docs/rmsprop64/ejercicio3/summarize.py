"""Resumir una etapa del ejercicio 3 con RMSProp-64.

Para cada corrida informa el mejor checkpoint (criterio del runner: mayor
macro-F1 de validation) y, si se pide, el estado de una época fija. El F1 del 5
y del 8 en época fija se recalcula con ``model-final.npz`` sobre la misma
validation, igual que en los análisis del ejercicio 3 original.
"""

import argparse
import csv
import json
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np

from sia_tp3.experiments import load_digits_development_split
from sia_tp3.metrics import classification_metrics
from sia_tp3.models import MultilayerPerceptron

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent / "results"


@lru_cache(maxsize=1)
def validation_split():
    return load_digits_development_split(
        ROOT / "data" / "digits.csv", validation_fraction=0.2, validation_seed=0,
        additional_train_path=ROOT / "data" / "more_digits.csv",
        deduplicate_inputs=True)


def history(folder):
    with (Path(folder) / "history.csv").open(newline="") as file:
        return list(csv.DictReader(file))


def per_class_f1(metrics, label):
    item = metrics["validation"]["per_class"][label]
    return item["f1"] if item["f1"] is not None else float("nan")


def best_row(stage, run):
    folder = RESULTS / stage / run
    metrics = json.loads((folder / "metrics.json").read_text())
    validation = metrics["validation"]
    return {
        "run": run,
        "best_epoch": metrics["best_epoch"],
        "accuracy": validation["accuracy"],
        "macro_f1": float(np.mean([item["f1"] for item in validation["per_class"]])),
        "f1_5": per_class_f1(metrics, 5),
        "f1_8": per_class_f1(metrics, 8),
    }


def fixed_row(stage, run, epoch):
    folder = RESULTS / stage / run
    rows = history(folder)
    row = rows[epoch]
    result = {
        "run": run,
        "epoch": epoch,
        "accuracy": float(row["validation_accuracy"]),
        "macro_f1": float(row["validation_macro_f1_present"]),
        "loss": float(row["validation_loss"]),
    }
    if epoch == len(rows) - 1:
        split = validation_split()
        model = MultilayerPerceptron.load(folder / "model-final.npz")
        predicted = np.argmax(model.predict(split.X_validation), axis=1)
        report = classification_metrics(
            split.y_validation, predicted, labels=tuple(range(10)))
        result["f1_5"] = report.for_label(5).f1
        result["f1_8"] = report.for_label(8).f1
    tail = rows[max(1, epoch - 49):epoch + 1]
    if len(tail) == 50:
        for name, field in (("accuracy", "validation_accuracy"),
                            ("macro_f1", "validation_macro_f1_present"),
                            ("loss", "validation_loss")):
            values = [float(item[field]) for item in tail]
            result[f"{name}_mean_last_50"] = statistics.mean(values)
            result[f"{name}_sd_last_50"] = statistics.stdev(values)
    return result


def aggregate(rows, fields):
    summary = {}
    for field in fields:
        values = [row[field] for row in rows if field in row]
        summary[f"{field}_mean"] = statistics.mean(values)
        summary[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[f"{field}_min"] = min(values)
    return summary


def runs(stage):
    return sorted(path.name for path in (RESULTS / stage).iterdir()
                  if (path / "history.csv").exists())


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("--epoch", type=int, help="época fija; por defecto mejor checkpoint")
    args = parser.parse_args()
    rows = [fixed_row(args.stage, run, args.epoch) if args.epoch is not None
            else best_row(args.stage, run) for run in runs(args.stage)]
    for row in rows:
        print({key: round(value, 4) if isinstance(value, float) else value
               for key, value in row.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
