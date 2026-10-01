"""Entrenar el modelo de dígitos congelado y evaluar test una sola vez."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import _load_digit_file
from .digit_experiment import (LABELS, _epoch_metrics, _finite_number, _optimizer,
                               _prediction_rows, _report_dict, _report_values,
                               _seed, _validate_optimizer, _write_csv, one_hot,
                               parameter_count)
from .models import MultilayerPerceptron
from .training import fit


FINAL_CONFIG_KEYS = {
    "protocol", "architecture", "activations", "beta", "init_scale",
    "optimizer", "batch_size", "epochs", "model_seed", "shuffle_seed",
    "shuffle",
}


def load_final_digit_config(path):
    """Validar una configuración final sin admitir parámetros de búsqueda."""
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict) or set(config) != FINAL_CONFIG_KEYS:
        raise ValueError("campos inválidos en la configuración final")
    if config["protocol"] != "final":
        raise ValueError("la evaluación final requiere protocol=final")
    architecture = config["architecture"]
    if (not isinstance(architecture, list) or len(architecture) < 3
            or any(type(size) is not int or size < 1 for size in architecture)
            or architecture[0] != 784 or architecture[-1] != 10):
        raise ValueError("arquitectura de dígitos debe ser [784, ocultas..., 10]")
    expected_activations = ["tanh"] * (len(architecture) - 2) + ["logistic"]
    if config["activations"] != expected_activations:
        raise ValueError("las ocultas deben usar tanh y la salida logística")
    _finite_number(config["beta"], "beta", positive=True)
    _finite_number(config["init_scale"], "init_scale", non_negative=True)
    _validate_optimizer(config["optimizer"])
    if (type(config["batch_size"]) is not int
            or config["batch_size"] < 1):
        raise ValueError("batch_size debe ser un entero positivo")
    if type(config["epochs"]) is not int or config["epochs"] < 1:
        raise ValueError("epochs debe ser un entero positivo")
    _seed(config["model_seed"], "model_seed")
    _seed(config["shuffle_seed"], "shuffle_seed")
    if type(config["shuffle"]) is not bool:
        raise ValueError("shuffle debe ser booleano")
    return config


def run_final_digit_evaluation(config, development_path, test_path, output):
    """Entrenar con development completo y luego evaluar el test reservado."""
    if config["protocol"] != "final":
        raise ValueError("se requiere protocol=final")
    development_path, test_path = Path(development_path), Path(test_path)
    if development_path.resolve() == test_path.resolve():
        raise ValueError("development y test deben ser archivos diferentes")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output debe ser una carpeta nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(
        {"python": platform.python_version(), "numpy": np.__version__},
        indent=2) + "\n")

    # Test no se carga ni se consulta hasta que finaliza el entrenamiento.
    X_development, y_development = _load_digit_file(development_path)
    present_labels = tuple(int(label) for label in np.unique(y_development))
    if len(present_labels) < 2:
        raise ValueError("development debe contener al menos dos clases")
    development_targets = one_hot(y_development)
    model = MultilayerPerceptron(
        config["architecture"], activations=config["activations"],
        beta=config["beta"], seed=config["model_seed"],
        init_scale=config["init_scale"])
    model.save(output / "model-initial.npz")

    checkpoints = {0, 25, 50, 100, 150, config["epochs"]}

    def progress(row):
        if row["epoch"] in checkpoints:
            print(
                f"final época={row['epoch']} mse={row['mse']:.6g} "
                f"training_macro_f1={row['training_macro_f1_present']:.6g}",
                flush=True)

    start = perf_counter()
    history = fit(
        model, X_development, development_targets,
        optimizer=_optimizer(config["optimizer"]),
        batch_size=config["batch_size"], max_epochs=config["epochs"],
        target_mse=0.0, shuffle=config["shuffle"],
        seed=config["shuffle_seed"],
        epoch_metrics=_epoch_metrics(present_labels), progress=progress)
    training_seconds = perf_counter() - start
    model.save(output / "model-final.npz")
    _write_csv(output / "training-history.csv", history)
    development_outputs = model.predict(X_development)
    development_report, development_values = _report_values(
        development_targets, development_outputs, present_labels)

    # Única etapa de evaluación externa: no interviene en fit ni en selección.
    X_test, y_test = _load_digit_file(test_path)
    if not set(int(label) for label in np.unique(y_test)) <= set(LABELS):
        raise ValueError("test contiene etiquetas fuera del rango 0..9")
    test_targets = one_hot(y_test)
    test_outputs = model.predict(X_test)
    test_mse = float(np.mean((test_outputs - test_targets) ** 2))
    test_report, test_values = _report_values(
        test_targets, test_outputs, LABELS)

    metrics = {
        "training_seconds": training_seconds,
        "epochs": config["epochs"],
        "parameter_count": parameter_count(config["architecture"]),
        "development_mse": history[-1]["mse"],
        "test_mse": test_mse,
        "development": _report_dict(development_report),
        "test": _report_dict(test_report),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_csv(
        output / "test-predictions.csv",
        _prediction_rows(np.arange(len(y_test)), y_test, test_outputs))
    _write_csv(
        output / "test-confusion-matrix.csv",
        [{"expected": label, **{
            f"predicted_{prediction}": int(
                test_report.confusion_matrix[label, prediction])
            for prediction in range(10)}} for label in range(10)])
    summary = {
        "development_samples": len(y_development),
        "test_samples": len(y_test),
        "development_macro_f1": development_values["macro_f1_present"],
        "test_accuracy": test_values["accuracy"],
        "test_macro_f1": test_values["macro_f1_present"],
        "test_digit_5_f1": test_report.for_label(5).f1,
        "test_mse": test_mse,
        "training_seconds": training_seconds,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "data-source.json").write_text(json.dumps({
        "development_path": str(development_path),
        "development_sha256": hashlib.sha256(
            development_path.read_bytes()).hexdigest(),
        "test_path": str(test_path),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "test_opened": True,
        "test_evaluations": 1,
    }, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_final_digit_config(args.config)
        summary = run_final_digit_evaluation(
            config, args.development, args.test, args.output)
    except (ValueError, OSError, FloatingPointError,
            json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
