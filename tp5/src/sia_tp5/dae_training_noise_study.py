"""¿Mejora el DAE en p = 0,30 si se entrena también con ese nivel de ruido?

Compara el DAE final (lote limpio + 0,05 + 0,10 + 0,20) contra la misma receta
con un bloque adicional de corrupciones con p = 0,30. Ambas variantes se
entrenan con las semillas 0-9 de configs/dae_adam_lr005_seed*.json y se evalúan
exactamente igual que la curva final (100 corrupciones por letra y nivel,
generador de evaluación reiniciado en cada nivel con la semilla 200 + i).

Uso: uv run python -m sia_tp5.dae_training_noise_study
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .autoencoder import Autoencoder
from .data import load_font
from .denoising_experiment import evaluate_denoising
from .optimizers import optimizer_from_config
from .training import fit_multilevel_denoising_autoencoder

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "ejs-figs" / "1b-entrenar-con-p030.csv"
SEEDS = range(10)
EVALUATION_LEVELS = [0.0, 0.05, 0.1, 0.2, 0.3]
VARIANTS = {
    "final (limpio + 0,05 + 0,10 + 0,20)": [0.05, 0.1, 0.2],
    "con 0,30 (limpio + 0,05 + 0,10 + 0,20 + 0,30)": [0.05, 0.1, 0.2, 0.3],
}


def train_and_evaluate(configuration: dict, training_levels: list[float],
                       dataset) -> list[dict]:
    model = Autoencoder(
        configuration["architecture"],
        latent_layer=configuration["latent_layer"],
        activations=configuration["activations"],
        loss=configuration["loss"],
        seed=configuration["model_seed"],
        leaky_slope=configuration.get("leaky_slope", 0.01),
    )
    fit_multilevel_denoising_autoencoder(
        model, dataset.patterns,
        optimizer=optimizer_from_config(configuration["optimizer"]),
        max_epochs=configuration["max_epochs"],
        noise_probabilities=training_levels,
        generator=np.random.default_rng(configuration["training_noise_seed"]),
    )
    summaries = []
    for probability in EVALUATION_LEVELS:
        _, summary = evaluate_denoising(
            model, dataset.patterns, dataset.labels,
            probability=probability, repetitions=100,
            generator=np.random.default_rng(configuration["evaluation_noise_seed"]),
        )
        summaries.append(summary)
    return summaries


def main() -> None:
    dataset = load_font()
    rows = []
    for seed in SEEDS:
        configuration = json.loads(
            (ROOT / "configs" / f"dae_adam_lr005_seed{seed}.json").read_text())
        for variant, levels in VARIANTS.items():
            for summary in train_and_evaluate(configuration, levels, dataset):
                rows.append({
                    "variant": variant,
                    "seed": seed,
                    "noise_probability": summary["noise_probability"],
                    "mean_input_pixel_errors": summary["mean_input_pixel_errors"],
                    "mean_output_pixel_errors": summary["mean_output_pixel_errors"],
                    "exact_recovery_rate": summary["exact_recovery_rate"],
                    "worsened_rate": summary["worsened_rate"],
                })
            print(f"semilla {seed}: {variant} listo", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"resultados: {OUTPUT}")


if __name__ == "__main__":
    main()
