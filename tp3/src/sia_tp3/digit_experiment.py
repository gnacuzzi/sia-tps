"""Ejecutar la búsqueda de hiperparámetros del ejercicio 2 sin abrir test."""

import argparse
import csv
import hashlib
import json
import math
import platform
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiments import load_digits_development_split
from .metrics import classification_metrics
from .models import MultilayerPerceptron
from .optimizers import (Adam, AdaptiveLearningRate, GradientDescent, Momentum,
                         RMSProp)
from .training import fit


LABELS = tuple(range(10))
CONFIG_KEYS = {
    "protocol", "validation_fraction", "validation_seed", "checkpoints", "runs"
}
REPEATED_CONFIG_KEYS = CONFIG_KEYS | {"repeat_seeds"}
RUN_KEYS = {
    "name", "stage", "architecture", "activations", "beta", "init_scale",
    "optimizer", "batch_size", "max_epochs", "model_seed", "shuffle_seed",
    "shuffle",
}
OPTIMIZER_KEYS = {
    "gradient_descent": {"name", "learning_rate"},
    "momentum": {"name", "learning_rate", "alpha"},
    "adaptive": {
        "name", "learning_rate", "increase_by", "decrease_fraction", "patience"
    },
    "rmsprop": {"name", "learning_rate", "gamma", "epsilon"},
    "adam": {"name", "learning_rate", "beta1", "beta2", "epsilon"},
}


def one_hot(labels, class_count=10):
    """Convertir etiquetas enteras en un objetivo con una posición activa."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("labels debe ser un vector no vacío")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels debe contener enteros")
    if type(class_count) is not int or class_count < 2:
        raise ValueError("class_count debe ser un entero mayor o igual a dos")
    if np.any(labels < 0) or np.any(labels >= class_count):
        raise ValueError("hay etiquetas fuera del rango de clases")
    encoded = np.zeros((len(labels), class_count), dtype=np.float64)
    encoded[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return encoded


def parameter_count(architecture):
    """Contar pesos y bias entrenables de una arquitectura densa."""
    return int(sum((source + 1) * destination
                   for source, destination in zip(architecture, architecture[1:])))


def _finite_number(value, name, *, positive=False, non_negative=False):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} debe ser numérico")
    if not np.isfinite(value):
        raise ValueError(f"{name} debe ser finito")
    if positive and value <= 0:
        raise ValueError(f"{name} debe ser positivo")
    if non_negative and value < 0:
        raise ValueError(f"{name} debe ser no negativo")


def _fraction(value, name, *, allow_one=False):
    _finite_number(value, name)
    upper_ok = value <= 1 if allow_one else value < 1
    if value < 0 or not upper_ok:
        ending = "[0,1]" if allow_one else "[0,1)"
        raise ValueError(f"{name} debe pertenecer a {ending}")


def _seed(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} debe ser un entero no negativo")


def _validate_optimizer(config):
    if not isinstance(config, dict) or config.get("name") not in OPTIMIZER_KEYS:
        raise ValueError("optimizador desconocido")
    name = config["name"]
    if set(config) != OPTIMIZER_KEYS[name]:
        raise ValueError(f"campos inválidos para el optimizador {name}")
    _finite_number(config["learning_rate"], "learning_rate", positive=True)
    if name == "momentum":
        _fraction(config["alpha"], "alpha")
    elif name == "adaptive":
        _finite_number(config["increase_by"], "increase_by", positive=True)
        _fraction(config["decrease_fraction"], "decrease_fraction")
        if config["decrease_fraction"] == 0:
            raise ValueError("decrease_fraction debe ser positivo")
        if type(config["patience"]) is not int or config["patience"] < 1:
            raise ValueError("patience debe ser un entero positivo")
    elif name == "rmsprop":
        _fraction(config["gamma"], "gamma")
        _finite_number(config["epsilon"], "epsilon", positive=True)
    elif name == "adam":
        _fraction(config["beta1"], "beta1")
        _fraction(config["beta2"], "beta2")
        _finite_number(config["epsilon"], "epsilon", positive=True)


def load_digit_search_config(path):
    """Validar el plan antes de crear salidas o cargar imágenes."""
    config = json.loads(Path(path).read_text())
    if (not isinstance(config, dict)
            or frozenset(config) not in {frozenset(CONFIG_KEYS),
                                         frozenset(REPEATED_CONFIG_KEYS)}):
        raise ValueError("campos globales inválidos en la configuración")
    if config["protocol"] != "search":
        raise ValueError("este runner sólo admite protocol=search")
    _finite_number(config["validation_fraction"], "validation_fraction")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction debe estar entre 0 y 1")
    _seed(config["validation_seed"], "validation_seed")
    if "repeat_seeds" in config:
        repeat_seeds = config["repeat_seeds"]
        if (not isinstance(repeat_seeds, list) or not repeat_seeds
                or len(set(repeat_seeds)) != len(repeat_seeds)):
            raise ValueError("repeat_seeds debe contener semillas únicas")
        for seed in repeat_seeds:
            _seed(seed, "repeat_seeds")

    checkpoints = config["checkpoints"]
    if (not isinstance(checkpoints, list) or not checkpoints
            or any(type(epoch) is not int or epoch < 0 for epoch in checkpoints)
            or checkpoints != sorted(set(checkpoints))):
        raise ValueError("checkpoints debe contener épocas únicas y ordenadas")

    runs = config["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs debe ser una lista no vacía")
    names = set()
    for run in runs:
        if not isinstance(run, dict) or set(run) != RUN_KEYS:
            raise ValueError("campos inválidos en una corrida")
        if (not isinstance(run["name"], str)
                or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", run["name"])
                or run["name"] in names):
            raise ValueError("cada corrida necesita un nombre único en minúsculas")
        names.add(run["name"])
        if not isinstance(run["stage"], str) or not run["stage"]:
            raise ValueError("stage debe ser texto no vacío")
        architecture = run["architecture"]
        if (not isinstance(architecture, list) or len(architecture) < 3
                or any(type(size) is not int or size < 1 for size in architecture)
                or architecture[0] != 784 or architecture[-1] != 10):
            raise ValueError("arquitectura de dígitos debe ser [784, ocultas..., 10]")
        expected_activations = ["tanh"] * (len(architecture) - 2) + ["logistic"]
        if run["activations"] != expected_activations:
            raise ValueError("las ocultas deben usar tanh y la salida logística")
        _finite_number(run["beta"], "beta", positive=True)
        _finite_number(run["init_scale"], "init_scale", non_negative=True)
        _validate_optimizer(run["optimizer"])
        batch_size = run["batch_size"]
        if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
            raise ValueError("batch_size debe ser un entero positivo o null")
        if type(run["max_epochs"]) is not int or run["max_epochs"] < 1:
            raise ValueError("max_epochs debe ser un entero positivo")
        if checkpoints[-1] > run["max_epochs"]:
            raise ValueError("ningún checkpoint puede superar max_epochs")
        _seed(run["model_seed"], "model_seed")
        _seed(run["shuffle_seed"], "shuffle_seed")
        if type(run["shuffle"]) is not bool:
            raise ValueError("shuffle debe ser booleano")
    return config


def _expanded_runs(config):
    """Crear una corrida por semilla sin duplicar bloques de configuración."""
    if "repeat_seeds" not in config:
        return config["runs"]
    expanded = []
    for run in config["runs"]:
        for seed in config["repeat_seeds"]:
            repeated = dict(run)
            repeated["name"] = f"{run['name']}-seed-{seed}"
            repeated["model_seed"] = seed
            repeated["shuffle_seed"] = seed
            expanded.append(repeated)
    return expanded


def _optimizer(config):
    name = config["name"]
    learning_rate = config["learning_rate"]
    if name == "gradient_descent":
        return GradientDescent(learning_rate)
    if name == "momentum":
        return Momentum(learning_rate, alpha=config["alpha"])
    if name == "adaptive":
        return AdaptiveLearningRate(
            learning_rate,
            increase_by=config["increase_by"],
            decrease_fraction=config["decrease_fraction"],
            patience=config["patience"],
        )
    if name == "rmsprop":
        return RMSProp(
            learning_rate, gamma=config["gamma"], epsilon=config["epsilon"])
    return Adam(
        learning_rate,
        beta1=config["beta1"], beta2=config["beta2"], epsilon=config["epsilon"])


def _macro_present(report, present_labels, attribute):
    values = [getattr(report.for_label(label), attribute) for label in present_labels]
    finite = np.asarray([value for value in values if np.isfinite(value)])
    return float(finite.mean()) if len(finite) else float("nan")


def _report_values(expected, outputs, present_labels):
    expected_labels = np.argmax(expected, axis=1)
    predicted_labels = np.argmax(outputs, axis=1)
    report = classification_metrics(expected_labels, predicted_labels, labels=LABELS)
    return report, {
        "accuracy": report.accuracy,
        "macro_precision_present": _macro_present(
            report, present_labels, "precision"),
        "macro_recall_present": _macro_present(report, present_labels, "recall"),
        "macro_f1_present": _macro_present(report, present_labels, "f1"),
    }


def _epoch_metrics(present_labels):
    def calculate(training_y, training_outputs, validation_y, validation_outputs):
        _, training = _report_values(training_y, training_outputs, present_labels)
        row = {f"training_{name}": value for name, value in training.items()}
        if validation_y is not None:
            _, validation = _report_values(
                validation_y, validation_outputs, present_labels)
            row.update({f"validation_{name}": value
                        for name, value in validation.items()})
        return row
    return calculate


def _write_csv(path, rows):
    if not rows:
        raise ValueError("no se pueden guardar filas vacías")
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _report_dict(report):
    return {
        "labels": list(report.labels),
        "confusion_matrix": report.confusion_matrix.tolist(),
        "accuracy": report.accuracy,
        "macro_precision": _json_value(report.macro_precision),
        "macro_recall": _json_value(report.macro_recall),
        "macro_f1": _json_value(report.macro_f1),
        "per_class": [
            {name: _json_value(value) for name, value in vars(item).items()}
            for item in report.per_class
        ],
    }


def _prediction_rows(indices, expected_labels, outputs):
    predicted = np.argmax(outputs, axis=1)
    rows = []
    for index, expected, prediction, scores in zip(
            indices, expected_labels, predicted, outputs):
        row = {
            "source_index": int(index),
            "expected": int(expected),
            "predicted": int(prediction),
        }
        row.update({f"output_{label}": float(scores[label]) for label in LABELS})
        rows.append(row)
    return rows


def _restore(model, state):
    for target, value in zip(model.weights + model.biases, state):
        target[...] = value


def _prepare_output(config, data_path, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output debe ser una carpeta nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_path)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(
        {"python": platform.python_version(), "numpy": np.__version__},
        indent=2) + "\n")
    (output / "data-source.json").write_text(json.dumps(
        {"development_path": str(data_path),
         "development_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
         "test_opened": False},
        indent=2) + "\n")
    return data_path, output


def run_digit_search(config, data_path, output, *, stages=None):
    """Ejecutar corridas sobre un holdout de development; test no es argumento."""
    if config["protocol"] != "search":
        raise ValueError("run_digit_search requiere protocol=search")
    data_path, output = _prepare_output(config, data_path, output)
    split = load_digits_development_split(
        data_path,
        validation_fraction=config["validation_fraction"],
        validation_seed=config["validation_seed"],
    )
    np.savez_compressed(
        output / "split-indices.npz",
        train=split.train_indices,
        validation=split.validation_indices,
    )
    present_labels = tuple(int(label) for label in np.unique(split.y_train))
    if set(present_labels) != set(int(label) for label in np.unique(split.y_validation)):
        raise ValueError("training y validation no conservaron las mismas clases")
    (output / "development-classes.json").write_text(json.dumps(
        {"present": list(present_labels),
         "absent": sorted(set(LABELS) - set(present_labels))},
        indent=2) + "\n")
    train_targets = one_hot(split.y_train)
    validation_targets = one_hot(split.y_validation)

    selected_stages = None if stages is None else set(stages)
    runs = [run for run in _expanded_runs(config)
            if selected_stages is None or run["stage"] in selected_stages]
    if not runs:
        raise ValueError("ninguna corrida coincide con los stages solicitados")

    summary = []
    for run in runs:
        folder = output / run["name"]
        folder.mkdir()
        model = MultilayerPerceptron(
            run["architecture"],
            activations=run["activations"],
            beta=run["beta"],
            seed=run["model_seed"],
            init_scale=run["init_scale"],
        )
        model.save(folder / "model-initial.npz")
        best = {"metric": -np.inf, "validation_mse": np.inf,
                "epoch": None, "state": None}

        def progress(row):
            metric = row["validation_macro_f1_present"]
            better = (metric > best["metric"] or (
                metric == best["metric"]
                and row["validation_mse"] < best["validation_mse"]))
            if better:
                best.update({
                    "metric": metric,
                    "validation_mse": row["validation_mse"],
                    "epoch": row["epoch"],
                    "state": [array.copy() for array in model.weights + model.biases],
                })
            if row["epoch"] in config["checkpoints"] or row["epoch"] == 0:
                print(
                    f"{run['name']} época={row['epoch']} "
                    f"train_mse={row['mse']:.6g} "
                    f"validation_mse={row['validation_mse']:.6g} "
                    f"validation_macro_f1={metric:.6g}",
                    flush=True,
                )

        start = perf_counter()
        try:
            history = fit(
                model,
                split.X_train,
                train_targets,
                optimizer=_optimizer(run["optimizer"]),
                batch_size=run["batch_size"],
                max_epochs=run["max_epochs"],
                target_mse=0.0,
                shuffle=run["shuffle"],
                seed=run["shuffle_seed"],
                validation_data=(split.X_validation, validation_targets),
                epoch_metrics=_epoch_metrics(present_labels),
                progress=progress,
            )
        except FloatingPointError as error:
            elapsed = perf_counter() - start
            failure = {"status": "divergent", "error": str(error), "seconds": elapsed}
            (folder / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            summary.append({
                "run": run["name"], "stage": run["stage"],
                "status": "divergent", "best_epoch": None,
                "validation_macro_f1_present": None,
                "validation_accuracy": None, "validation_mse": None,
                "parameter_count": parameter_count(run["architecture"]),
                "updates_at_best_epoch": None, "seconds": elapsed,
            })
            continue
        elapsed = perf_counter() - start
        model.save(folder / "model-final.npz")
        _write_csv(folder / "history.csv", history)
        checkpoints = [row for row in history
                       if row["epoch"] in config["checkpoints"]]
        _write_csv(folder / "checkpoints.csv", checkpoints)

        if best["state"] is None:
            raise RuntimeError("no se pudo seleccionar una época de validation")
        _restore(model, best["state"])
        model.save(folder / "model-best.npz")
        training_outputs = model.predict(split.X_train)
        validation_outputs = model.predict(split.X_validation)
        training_report, _ = _report_values(
            train_targets, training_outputs, present_labels)
        validation_report, validation_values = _report_values(
            validation_targets, validation_outputs, present_labels)
        (folder / "metrics.json").write_text(json.dumps({
            "best_epoch": best["epoch"],
            "present_labels": list(present_labels),
            "training": _report_dict(training_report),
            "validation": _report_dict(validation_report),
        }, indent=2) + "\n")
        _write_csv(
            folder / "training-predictions.csv",
            _prediction_rows(split.train_indices, split.y_train, training_outputs),
        )
        _write_csv(
            folder / "validation-predictions.csv",
            _prediction_rows(
                split.validation_indices, split.y_validation, validation_outputs),
        )
        effective_batch = len(split.X_train) if run["batch_size"] is None else run["batch_size"]
        updates_per_epoch = math.ceil(len(split.X_train) / effective_batch)
        summary.append({
            "run": run["name"],
            "stage": run["stage"],
            "status": "complete",
            "best_epoch": best["epoch"],
            "validation_macro_f1_present": validation_values["macro_f1_present"],
            "validation_accuracy": validation_values["accuracy"],
            "validation_mse": best["validation_mse"],
            "parameter_count": parameter_count(run["architecture"]),
            "updates_at_best_epoch": best["epoch"] * updates_per_epoch,
            "seconds": elapsed,
        })

    _write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="digits.csv; el modo search no acepta digits_test.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", action="append",
                        help="Ejecutar sólo este stage; puede repetirse")
    args = parser.parse_args()
    try:
        config = load_digit_search_config(args.config)
        run_digit_search(
            config, args.data, args.output, stages=args.stage)
    except (ValueError, OSError, FloatingPointError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
