"""Validar el motor reutilizable con los problemas pequeños de la consigna."""

import argparse
import csv
import json
import platform
import re
from pathlib import Path
from time import perf_counter

import numpy as np

from .models import MultilayerPerceptron, Perceptron
from .training import fit


CASE_KEYS = {"name", "dataset", "architecture", "activations", "beta", "init_scale",
             "learning_rate", "max_epochs", "target_mse", "shuffle"}


def dataset(name, *, sample_count, input_interval):
    """Construir los datos sintéticos respetando el orden bipolar de la consigna."""
    if name in {"and", "xor"}:
        X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]], dtype=float)
        y = [-1, -1, -1, 1] if name == "and" else [1, 1, -1, -1]
        return X, np.array(y, dtype=float).reshape(-1, 1)
    X = np.linspace(*input_interval, sample_count).reshape(-1, 1)
    if name == "linear":
        return X, X.copy()
    if name == "tanh":
        return X, np.tanh(X)
    raise ValueError(f"dataset desconocido: {name}")


def build_model(case, seed):
    """Crear el mismo modelo público que utilizarán los ejercicios posteriores."""
    options = {"seed": seed, "beta": case["beta"], "init_scale": case["init_scale"]}
    if len(case["architecture"]) == 2:
        if case["architecture"][-1] != 1:
            raise ValueError("la validación simple requiere una salida")
        return Perceptron(case["architecture"][0], activation=case["activations"][0], **options)
    return MultilayerPerceptron(case["architecture"], activations=case["activations"], **options)


def load_config(path):
    """Rechazar campos desconocidos y configuraciones inválidas antes de entrenar."""
    config = json.loads(Path(path).read_text())
    if set(config) != {"seeds", "sample_count", "input_interval", "cases"}:
        raise ValueError("campos de configuración inválidos")
    seeds = config["seeds"]
    if (not isinstance(seeds, list) or not seeds or
            any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds debe contener enteros no negativos distintos")
    n = config["sample_count"]
    if type(n) is not int or n < 2:
        raise ValueError("sample_count debe ser un entero mayor o igual a dos")
    interval = np.asarray(config["input_interval"], dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or interval[0] >= interval[1]:
        raise ValueError("input_interval debe contener dos límites finitos crecientes")
    if not isinstance(config["cases"], list) or not config["cases"]:
        raise ValueError("cases debe ser una lista no vacía")
    names = set()
    for case in config["cases"]:
        if set(case) != CASE_KEYS:
            raise ValueError("campos de caso inválidos")
        name = case["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or name in names:
            raise ValueError("cada nombre de caso debe ser único y contener letras minúsculas, números o guiones")
        names.add(name)
        model = MultilayerPerceptron(case["architecture"], activations=case["activations"],
                                    beta=case["beta"], init_scale=case["init_scale"])
        X, y = dataset(case["dataset"], sample_count=n, input_interval=interval)
        model.validate_data(X, y)
        if (not np.isfinite(case["learning_rate"]) or case["learning_rate"] <= 0 or
                type(case["max_epochs"]) is not int or case["max_epochs"] < 1 or
                not np.isfinite(case["target_mse"]) or case["target_mse"] < 0 or
                type(case["shuffle"]) is not bool):
            raise ValueError("hiperparámetros de entrenamiento inválidos")
    return config


def write_csv(path, rows):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_validation(config, output):
    """Entrenar todos los casos y guardar evidencia sin ocultar fallos de convergencia."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(
        {"python": platform.python_version(), "numpy": np.__version__}, indent=2) + "\n")
    summary = []
    for case in config["cases"]:
        X, y = dataset(case["dataset"], sample_count=config["sample_count"],
                       input_interval=config["input_interval"])
        for seed in config["seeds"]:
            folder = output / f"{case['name']}-seed-{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            model = build_model(case, seed)
            model.save(folder / "initial.npz")
            np.savez_compressed(folder / "data.npz", X=X, y=y)
            bipolar = case["dataset"] in {"and", "xor"}
            def progress(row):
                if row["epoch"] % 1000 == 0:
                    print(f"{case['name']} seed={seed} época={row['epoch']} MSE={row['mse']:.6g}", flush=True)
            start = perf_counter()
            history = fit(model, X, y, learning_rate=case["learning_rate"],
                          max_epochs=case["max_epochs"], target_mse=case["target_mse"],
                          shuffle=case["shuffle"], seed=seed,
                          require_bipolar_accuracy=bipolar, progress=progress)
            elapsed = perf_counter() - start
            model.save(folder / "model.npz")
            predictions = model.predict(X)
            write_csv(folder / "history.csv", history)
            rows = []
            for x, target, prediction in zip(X, y, predictions):
                row = {f"x{i + 1}": float(value) for i, value in enumerate(x)}
                row.update({"expected": float(target[0]), "predicted": float(prediction[0])})
                if bipolar:
                    row["class"] = int(1 if prediction[0] >= 0 else -1)
                rows.append(row)
            write_csv(folder / "predictions.csv", rows)
            last = history[-1]
            record = {"case": case["name"], "seed": seed, "epochs": last["epoch"],
                      "mse": last["mse"], "accuracy": last["accuracy"],
                      "passed": last["converged"], "seconds": elapsed}
            summary.append(record)
            print(f"  {'PASS' if record['passed'] else 'FAIL'}: {record['epochs']} épocas, MSE={record['mse']:.6g}", flush=True)
    write_csv(output / "summary.csv", summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validación de perceptrones del TP3")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("output debe ser una carpeta nueva o vacía para conservar corridas anteriores")
        summary = run_validation(config, args.output)
    except (ValueError, OSError, FloatingPointError) as error:
        parser.exit(2, f"Error: {error}\n")
    return 0 if all(row["passed"] for row in summary) else 1
