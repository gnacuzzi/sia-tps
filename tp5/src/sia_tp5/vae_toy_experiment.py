"""Entrenar y documentar el VAE de juguete con perdida KL."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .optimizers import optimizer_from_config
from .toy_data import toy_bars
from .training import fit_variational_autoencoder
from .variational_autoencoder import VariationalAutoencoder


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(path, history):
    epochs = [row["epoch"] for row in history]
    total = [row["total_loss"] for row in history]
    reconstruction = [row["reconstruction_loss"] for row in history]
    weighted_kl = [row["kl_weight"] * row["kl_loss"] for row in history]
    raw_kl = [row["kl_loss"] for row in history]
    errors = [row["mean_pixel_errors"] for row in history]

    figure, (loss_axis, diagnostic_axis) = plt.subplots(
        1, 2, figsize=(11, 4.5)
    )
    loss_axis.plot(epochs, total, label="Total")
    loss_axis.plot(epochs, reconstruction, label="Reconstrucción")
    loss_axis.plot(epochs, weighted_kl, label="beta × KL")
    loss_axis.set_xlabel("Época")
    loss_axis.set_ylabel("Contribución a la loss")
    loss_axis.set_title("Componentes de la pérdida")
    loss_axis.grid(alpha=0.2)
    loss_axis.legend()

    diagnostic_axis.plot(epochs, raw_kl, color="tab:orange", label="KL")
    diagnostic_axis.set_xlabel("Época")
    diagnostic_axis.set_ylabel("KL sin ponderar", color="tab:orange")
    diagnostic_axis.tick_params(axis="y", labelcolor="tab:orange")
    diagnostic_axis.grid(alpha=0.2)
    error_axis = diagnostic_axis.twinx()
    error_axis.plot(
        epochs, errors, color="tab:blue", alpha=0.65,
        label="Errores de píxel",
    )
    error_axis.set_ylabel("Errores medios de píxel", color="tab:blue")
    error_axis.tick_params(axis="y", labelcolor="tab:blue")
    diagnostic_axis.set_title("Regularización y reconstrucción binaria")

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(configuration):
    inputs = toy_bars()
    model = VariationalAutoencoder(
        configuration["architecture"],
        latent_layer=configuration["latent_layer"],
        activations=configuration["activations"],
        loss=configuration.get("loss", "bce"),
        seed=configuration["model_seed"],
        sampling_seed=configuration["sampling_seed"],
        kl_weight=configuration["kl_weight"],
    )
    optimizer = optimizer_from_config(configuration["optimizer"])
    history = fit_variational_autoencoder(
        model,
        inputs,
        optimizer=optimizer,
        max_epochs=configuration["epochs"],
        generator=np.random.default_rng(configuration["training_seed"]),
    )
    tail_size = min(100, len(history))
    tail = history[-tail_size:]
    mu, log_var = model.encode_distribution(inputs)
    summary = {
        "epochs": configuration["epochs"],
        "kl_weight": model.kl_weight,
        "initial_total_loss": history[0]["total_loss"],
        "final_total_loss": history[-1]["total_loss"],
        "tail_mean_total_loss": float(np.mean([
            row["total_loss"] for row in tail
        ])),
        "tail_mean_reconstruction_loss": float(np.mean([
            row["reconstruction_loss"] for row in tail
        ])),
        "tail_mean_kl_loss": float(np.mean([
            row["kl_loss"] for row in tail
        ])),
        "tail_mean_pixel_errors": float(np.mean([
            row["mean_pixel_errors"] for row in tail
        ])),
        "tail_mean_exact_patterns": float(np.mean([
            row["exact_patterns"] for row in tail
        ])),
        "mu_min": float(mu.min()),
        "mu_max": float(mu.max()),
        "variance_min": float(np.exp(log_var).min()),
        "variance_max": float(np.exp(log_var).max()),
    }

    history_path = Path(configuration["history_output"])
    summary_path = Path(configuration["summary_output"])
    figure_path = Path(configuration["figure_output"])
    _write_csv(history_path, history)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _plot_history(figure_path, history)
    if "model_output" in configuration:
        model.save(configuration["model_output"])
    return model, history, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    _, _, summary = run(configuration)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
