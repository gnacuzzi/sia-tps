"""Figuras comparativas del ejercicio 1B a partir de los CSV de docs/dae/.

No entrena modelos: grafica los agregados ya publicados.

Uso: uv run python -m sia_tp5.report_figures_1b
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOCS = Path(__file__).resolve().parents[2] / "docs"
DAE = DOCS / "dae"
OUTPUT_DIR = DOCS / "ejs-figs"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
LEVEL_COLORS = {"0.1": "#2a78d6", "0.2": "#eb6834", "0.3": "#1baf7a"}
LEVELS = ["0.1", "0.2", "0.3"]


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _level(value: str) -> str:
    return str(float(value))


def _pick(path: Path, key: str | None, value: str | None, mean: str, sd: str) -> dict:
    """Devuelve {nivel: (media, desvío)} para la fila que coincide con key=value."""
    result = {}
    for row in _rows(path):
        if key is not None and row[key] != value:
            continue
        result[_level(row["noise_probability"])] = (float(row[mean]), float(row[sd]))
    return result


def evolution_steps() -> list[tuple[str, dict]]:
    agg = ("exact_recovery_rate_mean", "exact_recovery_rate_sd")
    return [
        ("Red del 1A\ntasa 0,03", _pick(DAE / "experiments/confirmation-aggregate.csv",
                                         None, None, *agg)),
        ("+ red profunda\n45-15-2", _pick(DAE / "experiments/architecture-comparison.csv",
                                          "architecture", "35-45-15-2-15-45-35", *agg)),
        ("+ lote limpio\n+ ruidoso", _pick(DAE / "experiments/clean-noisy-comparison.csv",
                                            "protocol", "clean_plus_noisy", *agg)),
        ("+ tasa\ndecreciente", _pick(DAE / "experiments/schedule-comparison.csv",
                                       "schedule", "exponential_0.03_to_0.003", *agg)),
        ("+ latente 5", _pick(DAE / "experiments/latent-comparison.csv",
                              "latent_dimension", "5", *agg)),
        ("+ ruido\nmultinivel", _pick(DAE / "experiments/multilevel-comparison.csv",
                                       "training_noise", "multilevel_0.05_0.10_0.20", *agg)),
        ("+ Leaky ReLU", _pick(DAE / "experiments/activation-comparison.csv",
                               "activation", "leaky_relu", *agg)),
        ("+ Adam 0,005", _pick(DAE / "experiments/adam-rate-confirmation.csv",
                               "learning_rate", "0.005", "exact_mean", "exact_std")),
        ("Final\n10 semillas", _pick(DAE / "final/noise-curve-aggregate.csv", None, None,
                                      "exact_recovery_rate_mean", "exact_recovery_rate_std")),
    ]


def plot_evolution(path: Path) -> None:
    steps = evolution_steps()
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for level in LEVELS:
        means = np.array([step[1][level][0] for step in steps]) * 100
        sds = np.array([step[1][level][1] for step in steps]) * 100
        color = LEVEL_COLORS[level]
        ax.errorbar(x, means, yerr=sds, color=color, linewidth=2, marker="o",
                    markersize=6, capsize=3, label=f"p = {level.replace('.', ',')}")
        ax.text(x[-1] + 0.15, means[-1], f"{means[-1]:.1f} %".replace(".", ","),
                va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xticks(x, [label for label, _ in steps], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.4, len(steps) - 0.3)
    ax.set_ylabel("recuperación exacta (%)", color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(title="ruido de evaluación", frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Recuperación exacta después de cada decisión",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_label_difficulty(path: Path, level: str = "0.2") -> None:
    rows = [row for row in _rows(DAE / "final/label-difficulty.csv")
            if _level(row["noise_probability"]) == level]
    rows.sort(key=lambda row: float(row["exact_recovery_rate"]))
    labels = [row["label"] for row in rows]
    values = np.array([float(row["exact_recovery_rate"]) for row in rows]) * 100
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.set_xlim(-0.9, len(rows) - 0.4)
    ax.bar(np.arange(len(rows)), values, color=LEVEL_COLORS[level], width=0.75)
    mean = values.mean()
    ax.axhline(mean, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    ax.text(-0.6, mean + 1.5, f"promedio {mean:.1f} %".replace(".", ","),
            ha="left", fontsize=9, color=TEXT_SECONDARY)
    ax.set_xticks(np.arange(len(rows)), labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("recuperación exacta (%)", color=TEXT_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"Recuperación exacta de cada carácter con p = "
                 f"{level.replace('.', ',')}\n"
                 "10 semillas x 100 corrupciones por carácter",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- Una figura por decisión del recorrido -----------------------------------

CHOSEN = "#2a78d6"
ALTERNATIVES = ["#eb6834", "#1baf7a"]
ALL_LEVELS = ["0.0", "0.05", "0.1", "0.2", "0.3"]


def _series_agg(path, key, value, error=("mean_output_pixel_errors_mean",
                                         "mean_output_pixel_errors_sd"),
                exact=("exact_recovery_rate_mean", "exact_recovery_rate_sd")):
    """{nivel: (error, desvío error, exacta, desvío exacta)} de un CSV agregado."""
    result = {}
    for row in _rows(path):
        if key is not None and row[key] != value:
            continue
        result[_level(row["noise_probability"])] = (
            float(row[error[0]]), float(row[error[1]]),
            float(row[exact[0]]), float(row[exact[1]]))
    return result


def _series_single(path, filters):
    """Igual que _series_agg para CSV de una sola semilla (desvío cero)."""
    result = {}
    for row in _rows(path):
        if any(row[key] != value for key, value in filters.items()):
            continue
        result[_level(row["noise_probability"])] = (
            float(row["mean_output_pixel_errors"]), 0.0,
            float(row["exact_recovery_rate"]), 0.0)
    return result


def _series_per_seed(path, variant):
    values: dict = {}
    for row in _rows(path):
        if row["variant"] != variant:
            continue
        level = _level(row["noise_probability"])
        values.setdefault(level, ([], []))
        values[level][0].append(float(row["mean_output_pixel_errors"]))
        values[level][1].append(float(row["exact_recovery_rate"]))
    return {level: (np.mean(err), np.std(err, ddof=1), np.mean(ex), np.std(ex, ddof=1))
            for level, (err, ex) in values.items()}


def plot_decision(path: Path, title: str, series: list[tuple[str, bool, dict]]) -> None:
    """Error de salida y recuperación exacta por nivel; la elegida en azul."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.array([float(level) for level in ALL_LEVELS])
    left.plot(x, 35 * x, color=TEXT_SECONDARY, linestyle=":", linewidth=1.2,
              label="errores de entrada antes de la red")
    alternative = iter(ALTERNATIVES)
    for label, chosen, data in series:
        color = CHOSEN if chosen else next(alternative)
        style = dict(color=color, marker="o", markersize=5, capsize=3,
                     linewidth=2.4 if chosen else 1.6,
                     label=(f"{label} ({chosen})" if isinstance(chosen, str)
                            else f"{label} (elegida)" if chosen else label))
        levels = [level for level in ALL_LEVELS if level in data]
        xs = np.array([float(level) for level in levels])
        left.errorbar(xs, [data[l][0] for l in levels], yerr=[data[l][1] for l in levels],
                      **style)
        right.errorbar(xs, [data[l][2] * 100 for l in levels],
                       yerr=[data[l][3] * 100 for l in levels], **style)
    left.set_ylabel("píxeles incorrectos por letra", color=TEXT_SECONDARY)
    left.set_title("Error de salida (menor es mejor)", fontsize=10, color=TEXT_PRIMARY)
    right.set_ylabel("recuperación exacta (%)", color=TEXT_SECONDARY)
    right.set_title("Recuperación exacta (mayor es mejor)", fontsize=10,
                    color=TEXT_PRIMARY)
    right.set_ylim(0, 102)
    for axis in (left, right):
        axis.set_xticks(x, [level.replace(".", ",") for level in
                            ["0", "0.05", "0.10", "0.20", "0.30"]])
        axis.set_xlabel("ruido de evaluación p", color=TEXT_SECONDARY)
        axis.grid(color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles, labels = right.get_legend_handles_labels()
    ref_handle, ref_label = left.get_legend_handles_labels()
    fig.legend(handles + ref_handle[:1], labels + ref_label[:1],
               loc="lower center", ncol=len(labels) + 1,
               frameon=False, fontsize=9)
    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def decision_figures() -> list[tuple[str, str, list]]:
    ex = DAE / "experiments"
    agg = _series_agg
    return [
        ("1b-d1-tasa.png", "Tasa de descenso de gradiente (red del 1A, 1 semilla)", [
            ("tasa 0,01", False, _series_single(ex / "rate-comparison-seed0.csv",
                                                {"learning_rate": "0.01"})),
            ("tasa 0,03", "punto de partida",
             _series_single(ex / "rate-comparison-seed0.csv",
                                               {"learning_rate": "0.03"})),
            ("tasa 0,05", False, _series_single(ex / "rate-comparison-seed0.csv",
                                                {"learning_rate": "0.05"})),
        ]),
        ("1b-d2-corrupciones.png", "Corrupciones por letra en cada época (3 semillas)", [
            ("K = 1", True, agg(ex / "corruptions-comparison.csv",
                                "corruptions_per_sample", "1")),
            ("K = 4", False, agg(ex / "corruptions-comparison.csv",
                                 "corruptions_per_sample", "4")),
        ]),
        ("1b-d3-arquitectura.png",
         "Arquitectura (base y profunda: 3 semillas; ancha: semilla 0)", [
            ("base 35-45-2-45-35", False, agg(ex / "architecture-comparison.csv",
                                              "architecture", "35-45-2-45-35")),
            ("profunda 35-45-15-2-15-45-35", True,
             agg(ex / "architecture-comparison.csv", "architecture",
                 "35-45-15-2-15-45-35")),
            ("ancha 35-60-2-60-35", False,
             _series_single(ex / "architecture-screen-seed0.csv",
                            {"architecture": "35-60-2-60-35"})),
        ]),
        ("1b-d4-lote.png", "Composición del lote de entrenamiento (3 semillas)", [
            ("solo con ruido", False, agg(ex / "clean-noisy-comparison.csv",
                                          "protocol", "noisy_only")),
            ("limpias + con ruido", True, agg(ex / "clean-noisy-comparison.csv",
                                               "protocol", "clean_plus_noisy")),
        ]),
        ("1b-d5-tasa-decreciente.png", "Tasa constante o decreciente (3 semillas)", [
            ("constante 0,03", False, agg(ex / "schedule-comparison.csv",
                                          "schedule", "constant_0.03")),
            ("decreciente 0,03 → 0,003", "elegida para GD",
             agg(ex / "schedule-comparison.csv", "schedule",
                 "exponential_0.03_to_0.003")),
        ]),
        ("1b-d6-latente.png", "Dimensión del espacio latente (3 semillas)", [
            ("latente de 2 valores", False, agg(ex / "latent-comparison.csv",
                                     "latent_dimension", "2")),
            ("latente de 5 valores", True, agg(ex / "latent-comparison.csv",
                                    "latent_dimension", "5")),
        ]),
        ("1b-d7-ruido-entrenamiento.png", "Niveles de ruido al entrenar (3 semillas)", [
            ("entrenamiento solo con p = 0,10", False,
             agg(ex / "multilevel-comparison.csv",
                                         "training_noise", "single_0.10")),
            ("entrenamiento con p = 0,05 + 0,10 + 0,20", True,
             agg(ex / "multilevel-comparison.csv",
                                              "training_noise",
                                              "multilevel_0.05_0.10_0.20")),
        ]),
        ("1b-d8-activacion.png", "Activación de las capas ocultas (3 semillas)", [
            ("tanh", False, agg(ex / "activation-comparison.csv", "activation", "tanh")),
            ("Leaky ReLU", True, agg(ex / "activation-comparison.csv",
                                     "activation", "leaky_relu")),
        ]),
        ("1b-d9-optimizador.png", "Optimizador (1 semilla)", [
            ("GD con tasa decreciente", False,
             _series_single(ex / "optimizer-screen-seed0.csv",
                            {"optimizer": "gradient_descent_schedule"})),
            ("Adam 0,001", False, _series_single(ex / "optimizer-screen-seed0.csv",
                                                 {"optimizer": "adam",
                                                  "learning_rate": "0.001"})),
            ("Adam 0,003", "pasa a confirmación",
             _series_single(ex / "optimizer-screen-seed0.csv",
                                                {"optimizer": "adam",
                                                 "learning_rate": "0.003"})),
        ]),
        ("1b-d10-tasa-adam.png", "Tasa de Adam (3 semillas)", [
            ("Adam 0,003", False, agg(ex / "adam-rate-confirmation.csv", "learning_rate",
                                      "0.003", error=("error_mean", "error_std"),
                                      exact=("exact_mean", "exact_std"))),
            ("Adam 0,005", True, agg(ex / "adam-rate-confirmation.csv", "learning_rate",
                                     "0.005", error=("error_mean", "error_std"),
                                     exact=("exact_mean", "exact_std"))),
        ]),
    ]


def plot_training_with_p030(path: Path) -> None:
    """Barras agrupadas: modelo final (sin 0,30) contra entrenado con 0,30."""
    source = OUTPUT_DIR / "1b-entrenar-con-p030.csv"
    if not source.exists():
        return
    rows = _rows(source)
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    names = ["sin 0,30 (modelo final)", "con 0,30"]
    colors = [CHOSEN, ALTERNATIVES[0]]
    levels = ["0.1", "0.2", "0.3"]
    metrics = [
        ("mean_output_pixel_errors", "Error de salida\n(píxeles mal por letra, menor es mejor)",
         1, "{:.2f}"),
        ("exact_recovery_rate", "Recuperación exacta\n(% de letras perfectas, mayor es mejor)",
         100, "{:.1f}"),
        ("worsened_rate", "Casos empeorados\n(% que sale peor de lo que entró, menor es mejor)",
         100, "{:.1f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    width = 0.38
    for axis, (metric, title, scale, fmt) in zip(axes, metrics):
        for offset, variant, name, color in zip((-width / 2, width / 2), variants,
                                                names, colors):
            means, sds = [], []
            for level in levels:
                values = [float(row[metric]) * scale for row in rows
                          if row["variant"] == variant
                          and _level(row["noise_probability"]) == level]
                means.append(np.mean(values))
                sds.append(np.std(values, ddof=1))
            xs = np.arange(len(levels)) + offset
            axis.bar(xs, means, width=width * 0.95, color=color, yerr=sds, capsize=3,
                     error_kw={"elinewidth": 1, "ecolor": TEXT_SECONDARY}, label=name)
            for x, mean, sd in zip(xs, means, sds):
                axis.text(x, mean + sd, fmt.format(mean).replace(".", ","), ha="center",
                          va="bottom", fontsize=8, color=TEXT_PRIMARY)
        axis.set_xticks(range(len(levels)), ["p = 0,10", "p = 0,20", "p = 0,30"])
        axis.set_title(title, fontsize=10, color=TEXT_PRIMARY)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.margins(y=0.15)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("¿Ayuda entrenar también con letras con p = 0,30?\n"
                 "mismo DAE y mismas 10 semillas; media ± desvío entre semillas",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_evolution(OUTPUT_DIR / "1b-evolucion.png")
    plot_label_difficulty(OUTPUT_DIR / "1b-dificultad-por-letra.png")
    for name, title, series in decision_figures():
        plot_decision(OUTPUT_DIR / name, title, series)
    plot_training_with_p030(OUTPUT_DIR / "1b-entrenar-con-p030.png")
    print(f"Figuras guardadas en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
