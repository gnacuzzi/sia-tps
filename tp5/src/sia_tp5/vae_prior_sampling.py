"""Muestrear el prior normal del VAE de juguete y evaluar sus salidas."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .toy_data import TOY_BAR_LABELS, toy_bars
from .variational_autoencoder import VariationalAutoencoder


def evaluate_prior(model, *, sample_count, seed, threshold=0.5):
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int)
            or sample_count < 1):
        raise ValueError("sample_count debe ser un entero positivo")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold debe pertenecer a (0, 1)")
    latent_size = model.architecture[model.latent_layer]
    generator = np.random.default_rng(seed)
    latent = generator.standard_normal((sample_count, latent_size))
    probabilities = model.decode(latent)
    binary = (probabilities >= threshold).astype(np.int64)
    patterns = toy_bars().astype(np.int64)
    distances = np.sum(
        binary[:, None, :] != patterns[None, :, :], axis=2
    )
    nearest_indices = np.argmin(distances, axis=1)
    minimum_distances = distances[
        np.arange(sample_count), nearest_indices
    ]
    valid = minimum_distances == 0

    rows = []
    for index in range(sample_count):
        nearest = int(nearest_indices[index])
        rows.append({
            "sample": index,
            "z": json.dumps(latent[index].tolist()),
            "binary": "".join(str(value) for value in binary[index]),
            "valid_bar": bool(valid[index]),
            "nearest_label": TOY_BAR_LABELS[nearest],
            "minimum_hamming_distance": int(minimum_distances[index]),
        })

    valid_counts = {
        label: int(np.sum(valid & (nearest_indices == index)))
        for index, label in enumerate(TOY_BAR_LABELS)
    }
    unique_patterns = len({tuple(pattern) for pattern in binary.tolist()})
    summary = {
        "sample_count": sample_count,
        "seed": seed,
        "threshold": threshold,
        "valid_samples": int(np.sum(valid)),
        "valid_rate": float(np.mean(valid)),
        "mean_minimum_hamming_distance": float(np.mean(minimum_distances)),
        "maximum_minimum_hamming_distance": int(np.max(minimum_distances)),
        "unique_binary_patterns": unique_patterns,
        "valid_counts": valid_counts,
    }
    return rows, summary, probabilities, binary


def plot_samples(path, rows, probabilities, binary, *, display_count):
    count = min(display_count, len(rows))
    columns = 6
    rows_count = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(1.8 * columns, 1.9 * rows_count),
        squeeze=False,
    )
    for index, axis in enumerate(axes.ravel()):
        if index >= count:
            axis.axis("off")
            continue
        axis.imshow(
            probabilities[index].reshape(3, 3), cmap="binary",
            vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        record = rows[index]
        if record["valid_bar"]:
            title = record["nearest_label"].replace("horizontal_", "H").replace(
                "vertical_", "V"
            )
            color = "#2b8a3e"
        else:
            title = f"d={record['minimum_hamming_distance']}"
            color = "#c2410c"
        axis.set_title(title, color=color, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.8)
        decided = binary[index].reshape(3, 3)
        for row in range(3):
            for column in range(3):
                axis.text(
                    column, row, str(decided[row, column]),
                    ha="center", va="center", fontsize=8,
                    color="tab:blue" if decided[row, column] else "#999999",
                )
    figure.suptitle(
        "Muestras del prior N(0,I): probabilidades y decisión binaria",
        fontsize=14,
    )
    figure.text(
        0.5, 0.01,
        "Verde = barra válida; naranja = distancia de Hamming a la barra más cercana",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(configuration):
    model = VariationalAutoencoder.load(configuration["model_path"])
    rows, summary, probabilities, binary = evaluate_prior(
        model,
        sample_count=configuration["sample_count"],
        seed=configuration["seed"],
        threshold=configuration.get("threshold", 0.5),
    )
    records_path = Path(configuration["records_output"])
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary_path = Path(configuration["summary_output"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    plot_samples(
        configuration["figure_output"], rows, probabilities, binary,
        display_count=configuration["display_count"],
    )
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    _, summary = run(configuration)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
