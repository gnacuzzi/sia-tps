"""Entrenar y evaluar una corrida reproducible de denoising."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .autoencoder import Autoencoder
from .data import load_font
from .noise import bit_flip
from .optimizers import optimizer_from_config
from .training import (fit_denoising_autoencoder,
                       fit_mixed_denoising_autoencoder,
                       fit_multilevel_denoising_autoencoder)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def evaluate_denoising(model, clean_inputs, labels, *, probability,
                       repetitions, generator):
    """Medir errores antes y después sobre corrupciones independientes."""
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) \
            or repetitions < 1:
        raise ValueError("repetitions debe ser un entero positivo")

    rows = []
    for repetition in range(repetitions):
        noisy = bit_flip(clean_inputs, probability, generator=generator)
        predictions = model.predict(noisy)
        binary = (predictions >= 0.5).astype(np.float64)
        input_errors = np.sum(noisy != clean_inputs, axis=1)
        output_errors = np.sum(binary != clean_inputs, axis=1)
        for label, before, after in zip(labels, input_errors, output_errors):
            rows.append({
                "repetition": repetition,
                "label": label,
                "input_pixel_errors": int(before),
                "output_pixel_errors": int(after),
                "improvement": int(before - after),
            })

    before = np.array([row["input_pixel_errors"] for row in rows])
    after = np.array([row["output_pixel_errors"] for row in rows])
    summary = {
        "noise_probability": float(probability),
        "repetitions": repetitions,
        "evaluated_samples": len(rows),
        "mean_input_pixel_errors": float(before.mean()),
        "std_input_pixel_errors": float(before.std()),
        "mean_output_pixel_errors": float(after.mean()),
        "std_output_pixel_errors": float(after.std()),
        "mean_improvement": float((before - after).mean()),
        "exact_recovery_rate": float(np.mean(after == 0)),
        "improved_rate": float(np.mean(after < before)),
        "unchanged_rate": float(np.mean(after == before)),
        "worsened_rate": float(np.mean(after > before)),
    }
    return rows, summary


def run_denoising_experiment(configuration, output_root):
    """Ejecutar entrenamiento DAE y evaluación externa definida por JSON."""
    dataset = load_font()
    model = Autoencoder(
        configuration["architecture"],
        latent_layer=configuration["latent_layer"],
        activations=configuration["activations"],
        loss=configuration["loss"],
        seed=configuration["model_seed"],
        leaky_slope=configuration.get("leaky_slope", 0.01),
    )
    output = Path(output_root) / configuration["name"]
    if output.exists():
        raise FileExistsError(
            f"la carpeta de corrida ya existe y no se sobrescribirá: {output}"
        )
    output.mkdir(parents=True)

    start = perf_counter()
    include_clean_inputs = configuration.get("include_clean_inputs", False)
    training_levels = configuration.get("training_noise_probabilities")
    if training_levels is not None:
        if not include_clean_inputs:
            raise ValueError(
                "training_noise_probabilities requiere include_clean_inputs=true"
            )
        fit = fit_multilevel_denoising_autoencoder
    else:
        fit = (fit_mixed_denoising_autoencoder if include_clean_inputs
               else fit_denoising_autoencoder)
    fit_arguments = {
        "optimizer": optimizer_from_config(configuration["optimizer"]),
        "max_epochs": configuration["max_epochs"],
        "generator": np.random.default_rng(configuration["training_noise_seed"]),
    }
    if training_levels is not None:
        fit_arguments["noise_probabilities"] = training_levels
    else:
        fit_arguments["noise_probability"] = configuration[
            "training_noise_probability"
        ]
    if not include_clean_inputs and training_levels is None:
        fit_arguments["corruptions_per_sample"] = configuration.get(
            "corruptions_per_sample", 1
        )
    history = fit(
        model,
        dataset.patterns,
        **fit_arguments,
    )
    elapsed = perf_counter() - start
    rows, evaluation = evaluate_denoising(
        model,
        dataset.patterns,
        dataset.labels,
        probability=configuration["evaluation_noise_probability"],
        repetitions=configuration["evaluation_repetitions"],
        generator=np.random.default_rng(configuration["evaluation_noise_seed"]),
    )
    summary = {
        "training_noise_probability": configuration.get(
            "training_noise_probability"
        ),
        "training_noise_probabilities": training_levels,
        "corruptions_per_sample": configuration.get(
            "corruptions_per_sample", 1
        ),
        "include_clean_inputs": include_clean_inputs,
        "epochs": history[-1]["epoch"],
        "initial_training_loss": history[0]["loss"],
        "final_training_loss": history[-1]["loss"],
        "elapsed_seconds": elapsed,
        "evaluation": evaluation,
    }

    (output / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "history.csv", history)
    _write_csv(output / "evaluation.csv", rows)
    model.save(output / "model.npz")
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    output, summary = run_denoising_experiment(
        configuration, arguments.output_root
    )
    print(json.dumps({"output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
