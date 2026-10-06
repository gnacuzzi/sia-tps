"""Graficar la comparación rápida de profundidad para Adam y RMSProp."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "output" / "depth-adam-rmsprop-quick-20261006"
SUMMARY = RESULTS / "summary.csv"
OUTPUT = RESULTS / "depth-adam-rmsprop-validation.png"

DEPTH_LABELS = ["64", "64–64", "64–64–64"]
COLORS = {"RMSProp": "#f47c20", "Adam": "#2ca02c"}


def identify(run_name):
    optimizer = "RMSProp" if "rmsprop" in run_name else "Adam"
    base = run_name.rsplit("-seed-", 1)[0]
    hidden = base.removeprefix("depth-").removesuffix("-rmsprop").removesuffix("-adam")
    return optimizer, hidden.replace("-", "–")


values = defaultdict(list)
with SUMMARY.open(newline="") as source:
    for row in csv.DictReader(source):
        optimizer, architecture = identify(row["run"])
        values[(optimizer, architecture)].append(
            float(row["validation_macro_f1_present"])
        )

fig, ax = plt.subplots(figsize=(11.8, 6.4), dpi=180)
x = np.arange(len(DEPTH_LABELS), dtype=float)
offsets = {"RMSProp": -0.08, "Adam": 0.08}

for optimizer in ("RMSProp", "Adam"):
    groups = [np.asarray(values[(optimizer, label)]) for label in DEPTH_LABELS]
    means = np.asarray([group.mean() for group in groups])
    deviations = np.asarray([group.std(ddof=1) for group in groups])
    positions = x + offsets[optimizer]
    ax.errorbar(
        positions,
        means,
        yerr=deviations,
        color=COLORS[optimizer],
        marker="o",
        markersize=8,
        linewidth=2.7,
        capsize=6,
        label=f"{optimizer} · media ± desvío",
        zorder=3,
    )
    for position, group in zip(positions, groups):
        jitter = np.linspace(-0.025, 0.025, len(group))
        ax.scatter(
            position + jitter,
            group,
            s=30,
            color=COLORS[optimizer],
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85,
            zorder=4,
        )

ax.annotate(
    "1 de 3 corridas cayó a 0,765",
    xy=(x[1] + offsets["RMSProp"], min(values[("RMSProp", "64–64")])),
    xytext=(0.25, 0.80),
    textcoords="data",
    color="#9d4f0f",
    fontsize=11,
    arrowprops={"arrowstyle": "->", "color": "#9d4f0f", "lw": 1.4},
)

ax.set_xticks(x, DEPTH_LABELS)
ax.set_xlabel("Capas ocultas", fontsize=13)
ax.set_ylabel("Mejor macro-F1 de validación", fontsize=13)
ax.set_ylim(0.74, 0.965)
ax.grid(axis="y", alpha=0.23)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right", frameon=True, fontsize=11)
ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
print(OUTPUT)
