#!/usr/bin/env python3
"""Generar el grafico de la diapositiva de epocas sin unir corridas independientes."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "docs"
    / "kohonen-experimentos"
    / "05-epochs"
    / "generated"
    / "resumen_epocas.csv"
)
OUTPUT = SOURCE.with_name("curvas_epocas_puntos")

PROFILES = (
    "k4-constant-local",
    "k4-decay-middle",
    "k5-constant-global",
    "k5-decay-wide",
    "k6-decay-global",
)
COLORS = {
    "k4-constant-local": "#53636c",
    "k4-decay-middle": "#8ba0aa",
    "k5-constant-global": "#1f5f8b",
    "k5-decay-wide": "#15999c",
    "k6-decay-global": "#e67e22",
}
EPOCHS = (25, 50, 100, 150, 250, 400)


def main() -> None:
    cache = Path(tempfile.gettempdir()) / "sia-tp4-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with SOURCE.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.edgecolor": "#cbd6df",
            "axes.labelcolor": "#53636c",
            "xtick.color": "#53636c",
            "ytick.color": "#53636c",
            "text.color": "#53636c",
        }
    )

    definitions = (
        ("quantization_error", "Error de cuantización", None),
        ("dead_units_pct", "% unidades muertas", None),
        ("training_seconds", "Tiempo de entrenamiento (s)", None),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))

    for axis, (field, title, ylabel) in zip(axes, definitions):
        for profile in PROFILES:
            profile_rows = sorted(
                (row for row in rows if row["profile"] == profile),
                key=lambda row: int(row["epochs"]),
            )
            axis.errorbar(
                [int(row["epochs"]) for row in profile_rows],
                [float(row[f"{field}_mean"]) for row in profile_rows],
                yerr=[float(row[f"{field}_std"]) for row in profile_rows],
                color=COLORS[profile],
                linestyle="none",
                marker="o",
                markersize=5.5,
                markeredgewidth=0,
                elinewidth=1.5,
                capsize=3,
                label=profile,
            )
        axis.axvline(150, color="#e67e22", linestyle=":", linewidth=1.4)
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", color="#102c46")
        axis.set_xlabel("épocas", fontsize=11)
        if ylabel:
            axis.set_ylabel(ylabel)
        axis.set_xticks(EPOCHS)
        axis.grid(color="#dbe4ea", alpha=0.8, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    axes[2].legend(loc="upper left", frameon=False, fontsize=9, handletextpad=0.8)
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.18, top=0.9, wspace=0.28)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=160, facecolor="white")
    figure.savefig(
        OUTPUT.with_suffix(".svg"),
        format="svg",
        facecolor="white",
        metadata={"Date": None},
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
