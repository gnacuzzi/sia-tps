"""Visualizar dónde caen en el latente las versiones ruidosas de cada letra.

Compara el autoencoder del 1A con una variante del DAE que usa latente de
dimensión 2. Esa variante existe solo para poder dibujar el latente: el DAE
final del 1B usa latente 5. Cada punto es el código de una letra con ruido
bit-flip; se colorea según si el decoder devuelve exactamente la letra limpia.

Uso: uv run python -m sia_tp5.latent_regions
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .autoencoder import Autoencoder  # noqa: E402
from .data import load_font  # noqa: E402
from .noise import bit_flip  # noqa: E402
from .optimizers import optimizer_from_config  # noqa: E402
from .training import (  # noqa: E402
    fit_autoencoder,
    fit_multilevel_denoising_autoencoder,
)

OUTPUT = Path(__file__).resolve().parents[2] / "docs" / "ejs-figs" / "1b-latente-ae-vs-dae.png"
NOISE_PROBABILITY = 0.10
REPETITIONS = 100
EVALUATION_SEED = 300

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
RECOVERED = "#2a78d6"
FAILED = "#e34948"


def train_basic_autoencoder(patterns):
    """Configuración final del 1A: 35-45-2-45-35, BCE, GD 0,1, 2.000 épocas."""
    model = Autoencoder([35, 45, 2, 45, 35], latent_layer=2,
                        activations=["tanh", "linear", "tanh", "logistic"],
                        loss="bce", seed=0)
    fit_autoencoder(model, patterns,
                    optimizer=optimizer_from_config(
                        {"name": "gradient_descent", "learning_rate": 0.1}),
                    max_epochs=2000, target_max_pixel_errors=None)
    return model


def train_denoising_latent2(patterns):
    """Receta final del DAE (ver ejercicio_1b.md) con latente 2 en lugar de 5."""
    model = Autoencoder([35, 45, 15, 2, 15, 45, 35], latent_layer=3,
                        activations=["leaky_relu", "leaky_relu", "linear",
                                     "leaky_relu", "leaky_relu", "logistic"],
                        loss="bce", seed=0, leaky_slope=0.01)
    fit_multilevel_denoising_autoencoder(
        model, patterns,
        optimizer=optimizer_from_config(
            {"name": "adam", "learning_rate": 0.005, "beta1": 0.9, "beta2": 0.999}),
        max_epochs=10000, noise_probabilities=[0.05, 0.1, 0.2],
        generator=np.random.default_rng(100))
    return model


def noisy_codes(model, patterns):
    """Codificar REPETITIONS corrupciones de cada letra y medir recuperación."""
    generator = np.random.default_rng(EVALUATION_SEED)
    repeated = np.repeat(patterns, REPETITIONS, axis=0)
    noisy = bit_flip(repeated, NOISE_PROBABILITY, generator=generator)
    codes = model.encode(noisy)
    recovered = np.all((model.predict(noisy) >= 0.5) == (repeated >= 0.5), axis=1)
    return codes, recovered


def plot_panel(axis, model, dataset, title):
    codes, recovered = noisy_codes(model, dataset.patterns)
    clean = model.encode(dataset.patterns)
    axis.scatter(codes[recovered, 0], codes[recovered, 1], s=6, color=RECOVERED,
                 alpha=0.35, linewidth=0, label="con ruido, recuperada exacta")
    axis.scatter(codes[~recovered, 0], codes[~recovered, 1], s=6, color=FAILED,
                 alpha=0.55, linewidth=0, label="con ruido, NO recuperada")
    axis.scatter(clean[:, 0], clean[:, 1], s=14, color=TEXT_PRIMARY, zorder=3,
                 label="letra limpia")
    for label, (z1, z2) in zip(dataset.labels, clean):
        axis.annotate(label, (z1, z2), xytext=(3, 3), textcoords="offset points",
                      fontsize=8, color=TEXT_PRIMARY, zorder=4)
    rate = recovered.mean() * 100
    axis.set_title(f"{title}\nrecuperación exacta con p = {NOISE_PROBABILITY:.2f}: "
                   f"{rate:.1f} %".replace(".", ","), fontsize=10, color=TEXT_PRIMARY)
    axis.set_xlabel("z1", color=TEXT_SECONDARY)
    axis.set_ylabel("z2", color=TEXT_SECONDARY)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    return rate


def main() -> None:
    dataset = load_font()
    basic = train_basic_autoencoder(dataset.patterns)
    denoising = train_denoising_latent2(dataset.patterns)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    rates = [
        plot_panel(axes[0], basic, dataset,
                   "Autoencoder del 1A (entrenado solo con letras limpias)"),
        plot_panel(axes[1], denoising, dataset,
                   "DAE con latente 2 (entrenado con ruido)"),
    ]
    axes[1].legend(loc="lower right", fontsize=8, frameon=False, markerscale=2.5)
    figure.suptitle(f"Dónde caen {REPETITIONS} versiones con ruido de cada letra",
                    color=TEXT_PRIMARY, fontsize=12)
    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(f"recuperación exacta: AE {rates[0]:.1f} %, DAE latente 2 {rates[1]:.1f} %")
    print(f"figura: {OUTPUT}")


if __name__ == "__main__":
    main()
