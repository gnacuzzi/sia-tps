"""Agregar corridas DAE finales y elegir ejemplos reproducibles."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .data import FONT_HEIGHT, FONT_WIDTH, load_font
from .noise import bit_flip


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def aggregate_curves(curves):
    """Agregar por nivel las métricas calculadas independientemente por semilla."""
    grouped = defaultdict(list)
    for curve in curves:
        for row in curve:
            grouped[row["noise_probability"]].append(row)

    rows = []
    for probability in sorted(grouped):
        seeds = grouped[probability]
        row = {"noise_probability": probability, "seeds": len(seeds)}
        for metric in ("mean_input_pixel_errors", "mean_output_pixel_errors",
                       "exact_recovery_rate", "worsened_rate"):
            values = np.array([seed[metric] for seed in seeds], dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return rows


def _evaluate_cases(run, probabilities, repetitions, dataset):
    model = Autoencoder.load(run["model_path"])
    records = []
    for probability in probabilities:
        generator = np.random.default_rng(run["evaluation_seed"])
        for repetition in range(repetitions):
            noisy = bit_flip(dataset.patterns, probability, generator=generator)
            output = (model.predict(noisy) >= 0.5).astype(np.float64)
            before = np.sum(noisy != dataset.patterns, axis=1)
            after = np.sum(output != dataset.patterns, axis=1)
            for index, label in enumerate(dataset.labels):
                records.append({
                    "seed": run["seed"],
                    "noise_probability": probability,
                    "repetition": repetition,
                    "label": label,
                    "index": index,
                    "input_pixel_errors": int(before[index]),
                    "output_pixel_errors": int(after[index]),
                    "improvement": int(before[index] - after[index]),
                    "clean": dataset.patterns[index],
                    "noisy": noisy[index],
                    "output": output[index],
                })
    return records


def aggregate_labels(records):
    """Resumir la dificultad de cada carácter en cada nivel y semilla."""
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["noise_probability"], record["label"])].append(record)

    rows = []
    for (probability, label), group in sorted(grouped.items()):
        before = np.array([row["input_pixel_errors"] for row in group])
        after = np.array([row["output_pixel_errors"] for row in group])
        rows.append({
            "noise_probability": probability,
            "label": label,
            "evaluated_samples": len(group),
            "mean_input_pixel_errors": float(before.mean()),
            "mean_output_pixel_errors": float(after.mean()),
            "exact_recovery_rate": float(np.mean(after == 0)),
            "improved_rate": float(np.mean(after < before)),
            "worsened_rate": float(np.mean(after > before)),
        })
    return rows


def _representative(candidates):
    before = np.array([row["input_pixel_errors"] for row in candidates])
    after = np.array([row["output_pixel_errors"] for row in candidates])
    improvement = before - after
    targets = np.median(before), np.median(after), np.median(improvement)
    return min(
        candidates,
        key=lambda row: (
            abs(row["input_pixel_errors"] - targets[0])
            + abs(row["output_pixel_errors"] - targets[1])
            + abs(row["improvement"] - targets[2]),
            row["seed"], row["repetition"], row["index"],
        ),
    )


def select_representative_cases(records):
    """Elegir ejemplos con reglas fijadas, no mediante inspección visual."""
    easy = [row for row in records
            if row["noise_probability"] == 0.1
            and row["input_pixel_errors"] > 0
            and row["output_pixel_errors"] == 0]
    difficult = [row for row in records
                 if row["noise_probability"] == 0.2
                 and 0 < row["output_pixel_errors"] < row["input_pixel_errors"]]
    failed = [row for row in records
              if row["noise_probability"] == 0.3
              and row["output_pixel_errors"] > row["input_pixel_errors"]]
    if not easy or not difficult or not failed:
        raise ValueError("no hay candidatos para las tres categorías")

    easy_case = min(
        easy,
        key=lambda row: (-row["input_pixel_errors"], row["seed"],
                         row["repetition"], row["index"]),
    )
    return {"easy": easy_case,
            "difficult": _representative(difficult),
            "failed": _representative(failed)}


def _plot_curve(path, rows):
    probabilities = np.array([row["noise_probability"] for row in rows])
    input_mean = np.array([row["mean_input_pixel_errors_mean"] for row in rows])
    input_std = np.array([row["mean_input_pixel_errors_std"] for row in rows])
    output_mean = np.array([row["mean_output_pixel_errors_mean"] for row in rows])
    output_std = np.array([row["mean_output_pixel_errors_std"] for row in rows])
    exact_mean = 100 * np.array([row["exact_recovery_rate_mean"] for row in rows])
    exact_std = 100 * np.array([row["exact_recovery_rate_std"] for row in rows])

    figure, (error_axis, exact_axis) = plt.subplots(1, 2, figsize=(11, 4.5))
    error_axis.plot(probabilities, input_mean, "o--", label="Entrada")
    error_axis.fill_between(probabilities, input_mean - input_std,
                            input_mean + input_std, alpha=0.15)
    error_axis.plot(probabilities, output_mean, "o-", label="Salida DAE")
    error_axis.fill_between(probabilities, output_mean - output_std,
                            output_mean + output_std, alpha=0.2)
    error_axis.set(xlabel="Probabilidad de bit-flip",
                   ylabel="Errores de píxel promedio",
                   title="Error antes y después del denoising")
    error_axis.grid(alpha=0.2)
    error_axis.legend()

    exact_axis.plot(probabilities, exact_mean, "o-", color="tab:green")
    exact_axis.fill_between(probabilities, exact_mean - exact_std,
                            exact_mean + exact_std, color="tab:green", alpha=0.2)
    exact_axis.set(xlabel="Probabilidad de bit-flip",
                   ylabel="Recuperaciones exactas (%)",
                   title="Recuperación exacta (10 semillas)", ylim=(0, 102))
    exact_axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_cases(path, selected):
    categories = [("easy", "Fácil"), ("difficult", "Difícil"),
                  ("failed", "Fallido")]
    columns = [("clean", "Objetivo limpio"), ("noisy", "Entrada corrupta"),
               ("output", "Reconstrucción")]
    figure, axes = plt.subplots(3, 3, figsize=(7.5, 8.5))
    for row_index, (key, category) in enumerate(categories):
        case = selected[key]
        for column_index, (field, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            image = case[field].reshape(FONT_HEIGHT, FONT_WIDTH)
            axis.imshow(1 - image, cmap="gray", vmin=0, vmax=1,
                        interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title)
        axes[row_index, 0].set_ylabel(
            f"{category}\np={case['noise_probability']:.2f}, {case['label']}\n"
            f"{case['input_pixel_errors']} → {case['output_pixel_errors']} errores",
            rotation=0, labelpad=48, va="center",
        )
    figure.suptitle("Casos representativos del DAE congelado", y=0.99)
    figure.tight_layout(rect=(0.08, 0, 1, 0.97))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_final_analysis(configuration):
    output = Path(configuration["output_dir"])
    if output.exists():
        raise FileExistsError(f"la carpeta ya existe y no se sobrescribirá: {output}")
    output.mkdir(parents=True)

    curves = []
    for run in configuration["runs"]:
        summary_path = Path(run["curve_summary"])
        curves.append(json.loads(summary_path.read_text(encoding="utf-8")))
    aggregate = aggregate_curves(curves)

    dataset = load_font()
    records = []
    for run in configuration["runs"]:
        records.extend(_evaluate_cases(
            run, configuration["noise_probabilities"],
            configuration["evaluation_repetitions"], dataset,
        ))
    label_rows = aggregate_labels(records)
    selected = select_representative_cases(records)

    _write_csv(output / "noise-curve-aggregate.csv", aggregate)
    _write_csv(output / "label-difficulty.csv", label_rows)
    _plot_curve(output / "noise-curve-aggregate.png", aggregate)
    _plot_cases(output / "representative-cases.png", selected)

    serializable_cases = {
        category: {key: value for key, value in case.items()
                   if key not in {"clean", "noisy", "output"}}
        for category, case in selected.items()
    }
    (output / "representative-cases.json").write_text(
        json.dumps(serializable_cases, indent=2) + "\n", encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    return output, aggregate, label_rows, serializable_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    output, aggregate, _, cases = run_final_analysis(configuration)
    print(json.dumps({"output": str(output), "curve": aggregate,
                      "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
