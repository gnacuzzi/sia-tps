"""Decodificar y documentar un recorrido lineal en el espacio latente."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .autoencoder import Autoencoder
from .data import load_font


def interpolate(model, dataset, left_label, right_label, t_values,
                *, threshold=0.5):
    """Evaluar puntos ``(1-t) z_left + t z_right`` y clasificar sus salidas."""
    if left_label == right_label:
        raise ValueError("los extremos de la interpolación deben ser distintos")
    if left_label not in dataset.labels or right_label not in dataset.labels:
        raise ValueError("ambos extremos deben pertenecer al dataset")
    values = np.asarray(t_values, dtype=np.float64)
    if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
            or np.any((values < 0.0) | (values > 1.0))):
        raise ValueError("t_values debe contener al menos dos valores entre 0 y 1")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold debe estar entre 0 y 1")

    codes = model.encode(dataset.patterns)
    left_code = codes[dataset.labels.index(left_label)]
    right_code = codes[dataset.labels.index(right_label)]
    latent = ((1.0 - values[:, None]) * left_code
              + values[:, None] * right_code)
    probabilities = model.decode(latent)
    binary = (probabilities >= threshold).astype(np.int64)

    records = []
    for t, code, output, decided in zip(
            values, latent, probabilities, binary):
        distances = np.sum(dataset.patterns != decided, axis=1).astype(int)
        minimum = int(distances.min())
        matches = [
            dataset.labels[index]
            for index in np.flatnonzero(distances == 0)
        ]
        nearest = [
            dataset.labels[index]
            for index in np.flatnonzero(distances == minimum)
        ]
        records.append({
            "t": float(t),
            "latent": code.tolist(),
            "probabilities": output.tolist(),
            "binary": decided.tolist(),
            "is_new": not matches,
            "matches": matches,
            "nearest_labels": nearest,
            "minimum_hamming_distance": minimum,
        })
    return records


def plot_interpolation(path, records, left_label, right_label, *, threshold):
    """Mostrar en paralelo las probabilidades y su decisión binaria."""
    columns = len(records)
    figure, axes = plt.subplots(
        2, columns, figsize=(2.05 * columns, 5.6), squeeze=False,
        gridspec_kw={"hspace": 0.38, "wspace": 0.22},
    )
    for column, record in enumerate(records):
        continuous = np.asarray(record["probabilities"]).reshape(7, 5)
        binary = np.asarray(record["binary"]).reshape(7, 5)
        axes[0, column].imshow(
            continuous, cmap="binary", vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        axes[1, column].imshow(
            binary, cmap="binary", vmin=0.0, vmax=1.0,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"t = {record['t']:.2f}", fontsize=11)
        if record["is_new"]:
            status = (
                "NUEVO\n"
                f"más cercano: {','.join(record['nearest_labels'])}\n"
                f"difiere en {record['minimum_hamming_distance']} píxel(es)"
            )
            color = "#0b7285"
        else:
            status = f"EXISTENTE: {','.join(record['matches'])}"
            color = "#495057"
        axes[1, column].set_title(status, color=color, fontsize=9)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            for spine in axes[row, column].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.8 if record["is_new"] else 1.0)

    axes[0, 0].set_ylabel(
        "Salida continua\n(negro = probabilidad 1)", fontsize=10
    )
    axes[1, 0].set_ylabel(
        f"Salida binaria\n(negro si p ≥ {threshold:g})", fontsize=10
    )
    figure.suptitle(
        f"Qué genera el decoder al recorrer de '{left_label}' a '{right_label}'",
        fontsize=16,
    )
    figure.text(
        0.5, 0.025,
        "Marco turquesa = patrón que no aparece en el conjunto de entrenamiento",
        ha="center", color="#0b7285", fontsize=10,
    )
    figure.subplots_adjust(top=0.84, bottom=0.12, left=0.075, right=0.99)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run(configuration):
    """Ejecutar una interpolación configurada y guardar figura y datos."""
    model = Autoencoder.load(configuration["model"])
    dataset = load_font()
    threshold = float(configuration.get("threshold", 0.5))
    records = interpolate(
        model,
        dataset,
        configuration["left_label"],
        configuration["right_label"],
        configuration["t_values"],
        threshold=threshold,
    )
    data_path = Path(configuration["data_output"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**configuration, "records": records}
    data_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_interpolation(
        configuration["figure_output"], records,
        configuration["left_label"], configuration["right_label"],
        threshold=threshold,
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
    records = run(configuration)
    print(json.dumps({
        "figure": configuration["figure_output"],
        "data": configuration["data_output"],
        "points": len(records),
        "new_points": sum(record["is_new"] for record in records),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
