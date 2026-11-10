"""Evaluar un DAE congelado sobre una curva de niveles de ruido."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .data import load_font
from .denoising_experiment import evaluate_denoising


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot_curve(path, summaries):
    probabilities = [row["noise_probability"] for row in summaries]
    input_errors = [row["mean_input_pixel_errors"] for row in summaries]
    output_errors = [row["mean_output_pixel_errors"] for row in summaries]
    exact_rates = [100.0 * row["exact_recovery_rate"] for row in summaries]

    figure, (error_axis, recovery_axis) = plt.subplots(
        1, 2, figsize=(11, 4.5)
    )
    error_axis.plot(probabilities, input_errors, marker="o", label="Entrada")
    error_axis.plot(probabilities, output_errors, marker="o", label="Salida DAE")
    error_axis.set_xlabel("Probabilidad de bit-flip")
    error_axis.set_ylabel("Errores de píxel promedio")
    error_axis.set_title("Error antes y después del denoising")
    error_axis.grid(alpha=0.2)
    error_axis.legend()

    recovery_axis.plot(
        probabilities, exact_rates, marker="o", color="tab:green"
    )
    recovery_axis.set_xlabel("Probabilidad de bit-flip")
    recovery_axis.set_ylabel("Recuperaciones exactas (%)")
    recovery_axis.set_ylim(0, 100)
    recovery_axis.set_title("Recuperación exacta")
    recovery_axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_noise_curve(configuration, output_root):
    """Evaluar sin actualizar pesos y guardar resultados reproducibles."""
    probabilities = configuration["noise_probabilities"]
    if (not isinstance(probabilities, list) or not probabilities
            or len(set(probabilities)) != len(probabilities)):
        raise ValueError("noise_probabilities debe ser una lista no vacía y única")

    output = Path(output_root) / configuration["name"]
    if output.exists():
        raise FileExistsError(
            f"la carpeta de evaluación ya existe y no se sobrescribirá: {output}"
        )
    output.mkdir(parents=True)

    dataset = load_font()
    model = Autoencoder.load(configuration["model_path"])
    summaries = []
    all_rows = []
    for probability in probabilities:
        rows, summary = evaluate_denoising(
            model,
            dataset.patterns,
            dataset.labels,
            probability=probability,
            repetitions=configuration["evaluation_repetitions"],
            generator=np.random.default_rng(configuration["evaluation_seed"]),
        )
        summaries.append(summary)
        all_rows.extend({"noise_probability": probability, **row} for row in rows)

    (output / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output / "summary.csv", summaries)
    _write_csv(output / "evaluation.csv", all_rows)
    _plot_curve(output / "noise-curve.png", summaries)
    return output, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    output, summaries = run_noise_curve(configuration, arguments.output_root)
    print(json.dumps({"output": str(output), "levels": summaries}, indent=2))


if __name__ == "__main__":
    main()
