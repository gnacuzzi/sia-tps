"""Ejecutar y conservar una corrida reproducible del autoencoder."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .data import load_font
from .optimizers import optimizer_from_config
from .training import fit_autoencoder, reconstruction_metrics


def _write_history(path, history):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def _plot_reconstructions(path, images, predictions, labels):
    binary = (predictions >= 0.5).reshape(-1, 7, 5)
    figure, axes = plt.subplots(8, 8, figsize=(10, 13))
    for index, label in enumerate(labels):
        original_axis = axes[2 * (index // 8), index % 8]
        predicted_axis = axes[2 * (index // 8) + 1, index % 8]
        original_axis.imshow(images[index], cmap="binary", vmin=0, vmax=1,
                             interpolation="nearest")
        predicted_axis.imshow(binary[index], cmap="binary", vmin=0, vmax=1,
                              interpolation="nearest")
        original_axis.set_title(f"{label} original", fontsize=7)
        predicted_axis.set_title(f"{label} reconstruida", fontsize=7)
        original_axis.set_xticks([])
        original_axis.set_yticks([])
        predicted_axis.set_xticks([])
        predicted_axis.set_yticks([])
    figure.suptitle("Originales y reconstrucciones binarizadas", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_latent(path, latent, labels):
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(latent[:, 0], latent[:, 1], color="tab:blue", alpha=0.7)
    for coordinates, label in zip(latent, labels):
        axis.annotate(label, coordinates, xytext=(4, 4),
                      textcoords="offset points", fontsize=9)
    axis.axhline(0, color="0.8", linewidth=0.8)
    axis.axvline(0, color="0.8", linewidth=0.8)
    axis.set_xlabel("z1")
    axis.set_ylabel("z2")
    axis.set_title("Representación de los 32 patrones en el espacio latente")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_experiment(configuration, output_root):
    if configuration.get("loss") != "mse":
        raise ValueError("la implementación inicial sólo admite loss MSE")
    dataset = load_font()
    model = Autoencoder(
        configuration["architecture"],
        latent_layer=configuration["latent_layer"],
        activations=configuration["activations"],
        seed=configuration["seed"],
    )
    optimizer = optimizer_from_config(configuration["optimizer"])
    output = Path(output_root) / configuration["name"]
    if output.exists():
        raise FileExistsError(
            f"la carpeta de corrida ya existe y no se sobrescribirá: {output}"
        )
    output.mkdir(parents=True)

    start = perf_counter()
    history = fit_autoencoder(
        model,
        dataset.patterns,
        optimizer=optimizer,
        max_epochs=configuration["max_epochs"],
        target_max_pixel_errors=configuration["target_max_pixel_errors"],
    )
    elapsed = perf_counter() - start
    predictions = model.predict(dataset.patterns)
    metrics = reconstruction_metrics(dataset.patterns, predictions)
    per_pattern = {
        label: int(errors)
        for label, errors in zip(dataset.labels, metrics.pop("pixel_errors"))
    }
    summary = {
        **metrics,
        "final_loss": history[-1]["loss"],
        "epochs": history[-1]["epoch"],
        "elapsed_seconds": elapsed,
        "converged": (
            metrics["max_pixel_errors"]
            <= configuration["target_max_pixel_errors"]
        ),
        "pixel_errors_by_pattern": per_pattern,
    }

    (output / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_history(output / "history.csv", history)
    model.save(output / "model.npz")
    _plot_reconstructions(
        output / "reconstructions.png", dataset.images, predictions, dataset.labels
    )
    _plot_latent(output / "latent.png", model.encode(dataset.patterns), dataset.labels)
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    output, summary = run_experiment(configuration, arguments.output_root)
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
