"""Ampliar las oscilaciones del entrenamiento final sin reabrir test."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def run(history_path, output_path, window=10):
    with Path(history_path).open(newline="") as file:
        rows = list(csv.DictReader(file))
    epochs = np.array([int(row["epoch"]) for row in rows])
    mse = np.array([float(row["mse"]) for row in rows])
    macro_f1 = np.array([
        float(row["training_macro_f1_present"]) for row in rows])
    smooth_epochs = epochs[window - 1:]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].plot(epochs, mse, alpha=0.55, label="MSE crudo")
    axes[0].plot(smooth_epochs, moving_average(mse, window), linewidth=2,
                 label=f"Media móvil ({window} épocas)")
    axes[0].set(title="MSE de training: detalle desde época 40",
                xlabel="Época", ylabel="MSE", yscale="log", xlim=(40, 200))
    axes[1].plot(epochs, macro_f1, alpha=0.55, label="Macro-F1 crudo")
    axes[1].plot(smooth_epochs, moving_average(macro_f1, window), linewidth=2,
                 label=f"Media móvil ({window} épocas)")
    axes[1].set(title="Macro-F1 de training: detalle desde época 40",
                xlabel="Época", ylabel="Macro-F1", xlim=(40, 200),
                ylim=(0.975, 1.001))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Estabilidad del entrenamiento final del ejercicio 2")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()
    run(args.history, args.output, args.window)
