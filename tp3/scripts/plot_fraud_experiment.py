"""Graficar una corrida de fraude guardada sin volver a entrenar modelos."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(path):
    """Leer y validar las curvas de training y validation de un modelo."""
    with Path(path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"historial vacío: {path}")
    required = {"epoch", "mse", "validation_mse"}
    if not required.issubset(rows[0]):
        raise ValueError(f"faltan columnas en el historial: {path}")
    validation_values = [row["validation_mse"].strip() for row in rows]
    if any(validation_values) and not all(validation_values):
        raise ValueError(f"validation incompleta en el historial: {path}")
    try:
        epochs = np.asarray([int(row["epoch"]) for row in rows])
        training = np.asarray([float(row["mse"]) for row in rows])
        validation = (np.asarray([float(value) for value in validation_values])
                      if all(validation_values) else None)
    except (TypeError, ValueError) as error:
        raise ValueError(f"historial inválido: {path}") from error
    if not np.isfinite(training).all() or np.any(training < 0):
        raise ValueError(f"los MSE deben ser no negativos y finitos: {path}")
    if validation is not None and (
            not np.isfinite(validation).all() or np.any(validation < 0)):
        raise ValueError(f"los MSE deben ser no negativos y finitos: {path}")
    if epochs[0] != 0 or np.any(np.diff(epochs) != 1):
        raise ValueError(f"las épocas deben empezar en cero y ser consecutivas: {path}")
    return epochs, training, validation


def discover_models(run):
    """Obtener el orden de modelos registrado por el runner."""
    summary_path = run / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"no existe {summary_path}")
    summary = json.loads(summary_path.read_text())
    models = [row.get("model") for row in summary]
    if not models or any(not isinstance(name, str) for name in models):
        raise ValueError("summary.json no contiene modelos válidos")
    if len(set(models)) != len(models):
        raise ValueError("summary.json contiene modelos repetidos")
    return models


def _format_axis(axis, model, *, logarithmic=False):
    axis.set(title=model.capitalize(), xlabel="Época", ylabel="MSE")
    if logarithmic:
        axis.set_yscale("log")
    axis.grid(alpha=0.2)
    axis.legend()


def plot_run(run, output):
    """Crear vistas completas, normal y logarítmica, de una corrida."""
    run, output = Path(run), Path(output)
    models = discover_models(run)
    histories = {
        model: load_history(run / model / "history.csv")
        for model in models
    }
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    for filename, logarithmic in [("learning-curves.png", False),
                                  ("learning-curves-log.png", True)]:
        figure, axes = plt.subplots(
            len(models), 1, figsize=(9, 3.6 * len(models)),
            squeeze=False, layout="constrained")
        for axis, model in zip(axes[:, 0], models):
            epochs, training, validation = histories[model]
            axis.plot(epochs, training, label="Training", color="#2864a5")
            if validation is None:
                best = int(np.argmin(training))
                axis.scatter(epochs[best], training[best], color="#174778", zorder=3,
                             label=f"Menor training: época {epochs[best]}")
            else:
                best = int(np.argmin(validation))
                axis.plot(epochs, validation, label="Validation", color="#d46b27")
                axis.scatter(epochs[best], validation[best], color="#a82f1e", zorder=3,
                             label=f"Mejor validation: época {epochs[best]}")
            _format_axis(axis, model, logarithmic=logarithmic)
        figure.suptitle("Ejercicio 1 · Evolución del error por época")
        figure.savefig(output / filename, dpi=170)
        plt.close(figure)

    diagnostics = []
    for model, (epochs, training, validation) in histories.items():
        tail_count = min(len(epochs) - 1, max(1, int(np.ceil((len(epochs) - 1) * 0.1))))
        best_training = int(np.argmin(training))
        best_validation = int(np.argmin(validation)) if validation is not None else None
        diagnostics.append({
            "model": model,
            "last_epoch": int(epochs[-1]),
            "final_training_mse": float(training[-1]),
            "best_training_epoch": int(epochs[best_training]),
            "best_training_mse": float(training[best_training]),
            "training_change_last_10_percent": float(
                training[-1] - training[-1 - tail_count]),
            "final_validation_mse": (float(validation[-1])
                                     if validation is not None else None),
            "best_validation_epoch": (int(epochs[best_validation])
                                      if validation is not None else None),
            "best_validation_mse": (float(validation[best_validation])
                                    if validation is not None else None),
            "validation_change_last_10_percent": (
                float(validation[-1] - validation[-1 - tail_count])
                if validation is not None else None),
        })
    with (output / "learning-diagnostics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="carpeta producida por el runner de fraude")
    parser.add_argument("--output", type=Path,
                        help="carpeta de gráficos; por defecto RUN/plots")
    args = parser.parse_args()
    output = args.output if args.output is not None else args.run / "plots"
    try:
        diagnostics = plot_run(args.run, output)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")
    for row in diagnostics:
        if row["best_validation_epoch"] is None:
            print(f"{row['model']}: menor training en época "
                  f"{row['best_training_epoch']} "
                  f"(MSE={row['best_training_mse']:.6g})")
        else:
            print(f"{row['model']}: mejor validation en época "
                  f"{row['best_validation_epoch']} "
                  f"(MSE={row['best_validation_mse']:.6g})")
    print(f"Gráficos guardados en {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
