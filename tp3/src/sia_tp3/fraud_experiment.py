"""Ejecutar las etapas de aprendizaje y generalización del ejercicio de fraude."""

import argparse
import csv
import hashlib
import json
import platform
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import FRAUD_FEATURES
from .experiments import (load_fraud_experiment_split,
                          load_fraud_learning_data)
from .models import Perceptron
from .training import fit


MODEL_KEYS = {"name", "activation", "beta", "init_scale"}
TRAINING_KEYS = {"optimizer", "learning_rate", "batch_size", "max_epochs",
                 "target_mse", "shuffle", "model_seed", "shuffle_seed"}
COMMON_CONFIG_KEYS = {"protocol", "models", "training"}
GENERALIZATION_CONFIG_KEYS = COMMON_CONFIG_KEYS | {
    "test_fraction", "validation_fraction", "test_seed", "validation_seed",
}


def _fraction(value, name):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} debe ser numérico")
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} debe estar entre 0 y 1")


def _seed(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} debe ser un entero no negativo")


def load_fraud_config(path):
    """Leer la configuración y rechazar errores antes de cargar o entrenar datos."""
    config = json.loads(Path(path).read_text())
    protocol = config.get("protocol")
    if protocol == "learning":
        expected_keys = COMMON_CONFIG_KEYS
    elif protocol == "generalization":
        expected_keys = GENERALIZATION_CONFIG_KEYS
    else:
        raise ValueError("protocol debe ser learning o generalization")
    if set(config) != expected_keys:
        raise ValueError("campos globales inválidos en la configuración de fraude")
    if protocol == "generalization":
        _fraction(config["test_fraction"], "test_fraction")
        _fraction(config["validation_fraction"], "validation_fraction")
        _seed(config["test_seed"], "test_seed")
        _seed(config["validation_seed"], "validation_seed")

    models = config["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("models debe ser una lista no vacía")
    names = set()
    for model in models:
        if not isinstance(model, dict) or set(model) != MODEL_KEYS:
            raise ValueError("campos de modelo inválidos")
        name = model["name"]
        if (not isinstance(name, str)
                or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
                or name in names):
            raise ValueError("cada modelo necesita un nombre único en minúsculas")
        names.add(name)
        if model["activation"] not in {"linear", "tanh", "logistic"}:
            raise ValueError("fraude requiere una activación continua conocida")
        if (not isinstance(model["beta"], (int, float))
                or isinstance(model["beta"], bool)
                or not np.isfinite(model["beta"]) or model["beta"] <= 0):
            raise ValueError("beta debe ser positivo y finito")
        if (not isinstance(model["init_scale"], (int, float))
                or isinstance(model["init_scale"], bool)
                or not np.isfinite(model["init_scale"]) or model["init_scale"] < 0):
            raise ValueError("init_scale debe ser no negativo y finito")
    activations = [model["activation"] for model in models]
    if protocol == "learning" and (
            activations.count("linear") != 1
            or sum(activation != "linear" for activation in activations) != 1):
        raise ValueError("learning requiere un modelo lineal y uno no lineal")
    if protocol == "generalization" and len(models) != 1:
        raise ValueError("generalization requiere exactamente el modelo seleccionado")

    training = config["training"]
    if not isinstance(training, dict) or set(training) != TRAINING_KEYS:
        raise ValueError("campos de training inválidos")
    if training["optimizer"] != "gradient_descent":
        raise ValueError("el baseline inicial sólo admite gradient_descent")
    if (not isinstance(training["learning_rate"], (int, float))
            or isinstance(training["learning_rate"], bool)
            or not np.isfinite(training["learning_rate"])
            or training["learning_rate"] <= 0):
        raise ValueError("learning_rate debe ser positivo y finito")
    batch_size = training["batch_size"]
    if (batch_size is not None
            and (type(batch_size) is not int or batch_size < 1)):
        raise ValueError("batch_size debe ser un entero positivo o null")
    if type(training["max_epochs"]) is not int or training["max_epochs"] < 1:
        raise ValueError("max_epochs debe ser un entero positivo")
    if (not isinstance(training["target_mse"], (int, float))
            or isinstance(training["target_mse"], bool)
            or not np.isfinite(training["target_mse"])
            or training["target_mse"] < 0):
        raise ValueError("target_mse debe ser no negativo y finito")
    if type(training["shuffle"]) is not bool:
        raise ValueError("shuffle debe ser booleano")
    _seed(training["model_seed"], "model_seed")
    _seed(training["shuffle_seed"], "shuffle_seed")
    return config


def _write_csv(path, rows):
    if not rows:
        raise ValueError("no se pueden guardar filas vacías")
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(indices, expected, predicted):
    return [
        {"source_index": int(index), "expected": float(target[0]),
         "predicted": float(output[0])}
        for index, target, output in zip(indices, expected, predicted)
    ]


def _prepare_output(config, data_path, output):
    """Crear una salida reproducible y registrar configuración, entorno y datos."""
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
        {"path": str(data_path),
         "sha256": hashlib.sha256(data_path.read_bytes()).hexdigest()},
        indent=2) + "\n")
    return data_path, output


def _train_models(config, output, X_train, y_train, train_indices, *,
                  validation_data=None, validation_indices=None):
    """Entrenar modelos comparables y guardar los artefactos de la etapa."""
    if (validation_data is None) != (validation_indices is None):
        raise ValueError("validation_data e índices deben indicarse juntos")
    summary = []
    training = config["training"]
    for model_config in config["models"]:
        name = model_config["name"]
        folder = output / name
        folder.mkdir()
        model = Perceptron(
            len(FRAUD_FEATURES),
            activation=model_config["activation"],
            beta=model_config["beta"],
            init_scale=model_config["init_scale"],
            seed=training["model_seed"],
        )
        model.save(folder / "initial.npz")

        def progress(row):
            if row["epoch"] % 100 == 0:
                message = (f"{name} época={row['epoch']} "
                           f"train={row['mse']:.6g}")
                if row["validation_mse"] is not None:
                    message += f" validation={row['validation_mse']:.6g}"
                print(message, flush=True)

        start = perf_counter()
        history = fit(
            model,
            X_train,
            y_train,
            learning_rate=training["learning_rate"],
            batch_size=training["batch_size"],
            max_epochs=training["max_epochs"],
            target_mse=training["target_mse"],
            shuffle=training["shuffle"],
            seed=training["shuffle_seed"],
            validation_data=validation_data,
            progress=progress,
        )
        elapsed = perf_counter() - start
        model.save(folder / "model.npz")
        _write_csv(folder / "history.csv", history)
        _write_csv(
            folder / "training-predictions.csv",
            _prediction_rows(train_indices, y_train, model.predict(X_train)),
        )
        if validation_data is not None:
            validation_X, validation_y = validation_data
            _write_csv(
                folder / "validation-predictions.csv",
                _prediction_rows(validation_indices, validation_y,
                                 model.predict(validation_X)),
            )
        last = history[-1]
        summary.append({
            "model": name,
            "epochs": last["epoch"],
            "training_mse": last["mse"],
            "validation_mse": last["validation_mse"],
            "reached_target_mse": last["converged"],
            "seconds": elapsed,
        })
    _write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run_fraud_learning(config, data_path, output):
    """Comparar lineal/no lineal entrenando ambos con todas las muestras."""
    if config["protocol"] != "learning":
        raise ValueError("run_fraud_learning requiere protocol=learning")
    data_path, output = _prepare_output(config, data_path, output)
    data = load_fraud_learning_data(data_path)
    np.savez_compressed(output / "sample-indices.npz", all=data.source_indices)
    data.standardizer.save(output / "standardizer.npz")
    return _train_models(
        config, output, data.X, data.y, data.source_indices)


def run_fraud_generalization(config, data_path, output):
    """Estudiar el modelo seleccionado con training, validation y test reservado."""
    if config["protocol"] != "generalization":
        raise ValueError("run_fraud_generalization requiere protocol=generalization")
    data_path, output = _prepare_output(config, data_path, output)
    split = load_fraud_experiment_split(
        data_path,
        test_fraction=config["test_fraction"],
        validation_fraction=config["validation_fraction"],
        test_seed=config["test_seed"],
        validation_seed=config["validation_seed"],
    )
    np.savez_compressed(
        output / "split-indices.npz",
        train=split.train_indices,
        validation=split.validation_indices,
        test=split.test_indices,
    )
    split.standardizer.save(output / "standardizer.npz")
    return _train_models(
        config, output, split.X_train, split.y_train, split.train_indices,
        validation_data=(split.X_validation, split.y_validation),
        validation_indices=split.validation_indices,
    )


def run_fraud_experiment(config, data_path, output):
    """Ejecutar únicamente la etapa declarada explícitamente en la configuración."""
    if config["protocol"] == "learning":
        return run_fraud_learning(config, data_path, output)
    return run_fraud_generalization(config, data_path, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_fraud_config(args.config)
        run_fraud_experiment(config, args.data, args.output)
    except (ValueError, OSError, FloatingPointError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0
