"""Entrenar el modelo de dígitos congelado y evaluar test una sola vez."""

import argparse
import hashlib
import json
import platform
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np

from .data import _load_digit_file
from .digit_augmentation import (random_translate_images,
                                 random_translate_rotate_images)
from .digit_development import load_unique_digit_development
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
OPTIONAL_FINAL_CONFIG_KEYS = {
    "loss", "l2_lambda", "augmentation", "learning_rate_schedule"
}


def load_final_digit_config(path):
    """Validar una configuración final sin admitir parámetros de búsqueda."""
    config = json.loads(Path(path).read_text())
    if (not isinstance(config, dict)
            or not FINAL_CONFIG_KEYS <= set(config)
            or set(config) - FINAL_CONFIG_KEYS - OPTIONAL_FINAL_CONFIG_KEYS):
        raise ValueError("campos inválidos en la configuración final")
    if config["protocol"] != "final":
        raise ValueError("la evaluación final requiere protocol=final")
    architecture = config["architecture"]
    if (not isinstance(architecture, list) or len(architecture) < 3
            or any(type(size) is not int or size < 1 for size in architecture)
            or architecture[0] != 784 or architecture[-1] != 10):
        raise ValueError("arquitectura de dígitos debe ser [784, ocultas..., 10]")
    loss = config.get("loss", "mse")
    if loss not in {"mse", "categorical_cross_entropy"}:
        raise ValueError("loss de clasificación desconocida")
    output_activation = (
        "softmax" if loss == "categorical_cross_entropy" else "logistic")
    expected_activations = ["tanh"] * (len(architecture) - 2) + [output_activation]
    if config["activations"] != expected_activations:
        raise ValueError("activaciones incompatibles con el loss final")
    _finite_number(config["beta"], "beta", positive=True)
    _finite_number(config["init_scale"], "init_scale", non_negative=True)
    _finite_number(config.get("l2_lambda", 0.0), "l2_lambda", non_negative=True)
    augmentation = config.get("augmentation")
    if augmentation is not None:
        if not isinstance(augmentation, dict):
            raise ValueError("augmentation final inválida")
        name = augmentation.get("name")
        expected = ({"name", "max_shift", "probability", "seed"}
                    if name == "translation" else {
                        "name", "max_shift", "translation_probability",
                        "max_angle_degrees", "rotation_probability", "seed"
                    } if name == "translation_rotation" else set())
        if not expected or set(augmentation) != expected:
            raise ValueError("augmentation final inválida")
        if (type(augmentation["max_shift"]) is not int
                or not 1 <= augmentation["max_shift"] <= 27):
            raise ValueError("max_shift final inválido")
        probabilities = ([augmentation["probability"]]
                         if name == "translation" else [
                             augmentation["translation_probability"],
                             augmentation["rotation_probability"]])
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               or not np.isfinite(value) or not 0 <= value <= 1
               for value in probabilities):
            raise ValueError("probabilidad de augmentation final inválida")
        if name == "translation_rotation":
            angle = augmentation["max_angle_degrees"]
            if (not isinstance(angle, (int, float)) or isinstance(angle, bool)
                    or not np.isfinite(angle) or not 0 < angle <= 45):
                raise ValueError("max_angle_degrees final inválido")
        _seed(augmentation["seed"], "augmentation seed")
    _validate_optimizer(config["optimizer"])
    if (type(config["batch_size"]) is not int
            or config["batch_size"] < 1):
        raise ValueError("batch_size debe ser un entero positivo")
    if type(config["epochs"]) is not int or config["epochs"] < 1:
        raise ValueError("epochs debe ser un entero positivo")
    schedule = config.get("learning_rate_schedule")
    if schedule is not None:
        if not isinstance(schedule, list) or not schedule:
            raise ValueError("learning_rate_schedule final inválido")
        starts = []
        for phase in schedule:
            if (not isinstance(phase, dict)
                    or set(phase) != {"start_epoch", "learning_rate"}):
                raise ValueError("fase final inválida")
            start = phase["start_epoch"]
            if type(start) is not int or not 1 <= start <= config["epochs"]:
                raise ValueError("start_epoch final inválido")
            _finite_number(
                phase["learning_rate"], "scheduled learning_rate", positive=True)
            starts.append(start)
        if starts != sorted(set(starts)) or starts[0] != 1:
            raise ValueError("schedule final debe empezar en 1 y estar ordenado")
        if schedule[0]["learning_rate"] != config["optimizer"]["learning_rate"]:
            raise ValueError("la primera tasa final debe coincidir con Adam")
        if config["optimizer"]["name"] == "adaptive":
            raise ValueError("schedule final incompatible con eta adaptativo")
    _seed(config["model_seed"], "model_seed")
    _seed(config["shuffle_seed"], "shuffle_seed")
    if type(config["shuffle"]) is not bool:
        raise ValueError("shuffle debe ser booleano")
    return config


def run_final_digit_evaluation(
        config, development_path, test_path, output, *,
        additional_development_path=None, deduplicate_inputs=False):
    """Entrenar con development completo y luego evaluar el test reservado."""
    if config["protocol"] != "final":
        raise ValueError("se requiere protocol=final")
    development_path, test_path = Path(development_path), Path(test_path)
    additional_development_path = (None if additional_development_path is None
                                   else Path(additional_development_path))
    if development_path.resolve() == test_path.resolve():
        raise ValueError("development y test deben ser archivos diferentes")
    if deduplicate_inputs and additional_development_path is None:
        raise ValueError("deduplicate_inputs requiere development adicional")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output debe ser una carpeta nueva o vacía")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(
        {"python": platform.python_version(), "numpy": np.__version__},
        indent=2) + "\n")

    # Test no se carga ni se consulta hasta que finaliza el entrenamiento.
    if deduplicate_inputs:
        development = load_unique_digit_development(
            development_path, additional_development_path)
        X_development, y_development = development.X, development.y
    else:
        X_development, y_development = _load_digit_file(development_path)
        if additional_development_path is not None:
            additional_X, additional_y = _load_digit_file(
                additional_development_path)
            X_development = np.concatenate([X_development, additional_X])
            y_development = np.concatenate([y_development, additional_y])
    present_labels = tuple(int(label) for label in np.unique(y_development))
    if len(present_labels) < 2:
        raise ValueError("development debe contener al menos dos clases")
    development_targets = one_hot(y_development)
    model = MultilayerPerceptron(
        config["architecture"], activations=config["activations"],
        beta=config["beta"], seed=config["model_seed"],
        init_scale=config["init_scale"], loss=config.get("loss", "mse"),
        l2_lambda=config.get("l2_lambda", 0.0))
    model.save(output / "model-initial.npz")

    checkpoints = {0, 25, 50, 100, 150, 200, 220, 250, config["epochs"]}

    def progress(row):
        if row["epoch"] in checkpoints:
            print(
                f"final época={row['epoch']} lr={row['learning_rate']:.6g} "
                f"loss={row['loss']:.6g} mse={row['mse']:.6g} "
                f"training_macro_f1={row['training_macro_f1_present']:.6g}",
                flush=True)

    augmentation = config.get("augmentation")
    if augmentation is None:
        batch_transform = None
    elif augmentation["name"] == "translation":
        batch_transform = partial(
            random_translate_images, max_shift=augmentation["max_shift"],
            probability=augmentation["probability"])
    else:
        batch_transform = partial(
            random_translate_rotate_images,
            max_shift=augmentation["max_shift"],
            translation_probability=augmentation["translation_probability"],
            max_angle_degrees=augmentation["max_angle_degrees"],
            rotation_probability=augmentation["rotation_probability"])
    schedule = {phase["start_epoch"]: phase["learning_rate"]
                for phase in config.get("learning_rate_schedule", [])}

    start = perf_counter()
    history = fit(
        model, X_development, development_targets,
        optimizer=_optimizer(config["optimizer"]),
        batch_size=config["batch_size"], max_epochs=config["epochs"],
        target_mse=0.0, shuffle=config["shuffle"],
        seed=config["shuffle_seed"],
        batch_transform=batch_transform,
        batch_transform_seed=(None if augmentation is None
                              else augmentation["seed"]),
        learning_rate_schedule=schedule,
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
    test_loss = model.data_loss(X_test, test_targets)
    test_report, test_values = _report_values(
        test_targets, test_outputs, LABELS)

    metrics = {
        "training_seconds": training_seconds,
        "epochs": config["epochs"],
        "parameter_count": parameter_count(config["architecture"]),
        "development_mse": history[-1]["mse"],
        "test_mse": test_mse,
        "test_loss": test_loss,
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
        "test_digit_8_f1": test_report.for_label(8).f1,
        "test_loss": test_loss,
        "test_mse": test_mse,
        "training_seconds": training_seconds,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    source = {
        "development_path": str(development_path),
        "development_sha256": hashlib.sha256(
            development_path.read_bytes()).hexdigest(),
        "test_path": str(test_path),
        "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "test_opened": True,
        "test_evaluations": 1,
        "deduplicate_inputs": deduplicate_inputs,
    }
    if additional_development_path is not None:
        source.update({
            "additional_development_path": str(additional_development_path),
            "additional_development_sha256": hashlib.sha256(
                additional_development_path.read_bytes()).hexdigest(),
        })
    (output / "data-source.json").write_text(json.dumps(source, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--additional-development", type=Path)
    parser.add_argument("--deduplicate-inputs", action="store_true")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_final_digit_config(args.config)
        summary = run_final_digit_evaluation(
            config, args.development, args.test, args.output,
            additional_development_path=args.additional_development,
            deduplicate_inputs=args.deduplicate_inputs)
    except (ValueError, OSError, FloatingPointError,
            json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
