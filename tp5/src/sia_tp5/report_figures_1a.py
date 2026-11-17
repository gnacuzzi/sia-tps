"""Figuras comparativas del ejercicio 1A para docs/ejs.md.

Los valores provienen de las corridas documentadas en docs/ejercicio_1a.md
(los resultados crudos viven en results/, que no se versiona). Este script
solo los grafica; no entrena modelos.

Uso: uv run python -m sia_tp5.report_figures_1a
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "ejs-figs"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_MSE = "#2a78d6"
SERIES_BCE = "#eb6834"

# Tamiz grueso, semilla 0, MSE, 2.000 épocas: máximo de píxeles incorrectos.
SCREEN_ARCHS = [
    "A0  35-2-35\n(177 parám.)",
    "A1  35-10-2-10-35\n(797)",
    "35-15-2-15-35\n(1.177)",
    "A2  35-20-2-20-35\n(1.557)",
    "A3  35-35-2-35-35\n(2.697)",
    "A4  35-20-10-2-10-20-35\n(1.937)",
]
SCREEN_OPTIMIZERS = ["GD 0,3", "Momentum 0,01", "RMSProp 0,01", "Adam 0,01"]
SCREEN_MAX_ERRORS = np.array(
    [
        [11, 11, 10, 10],
        [6, 6, 3, 4],
        [5, 5, 3, 3],
        [2, 4, 2, 3],
        [2, 3, 3, 3],
        [2, 4, 2, 6],
    ]
)

# Máximo de píxeles incorrectos por semilla en cada etapa del embudo.
FINALISTS = [
    ("A3  35-35-2-35-35, GD 0,3, MSE", "MSE", [2, 2, 1]),
    ("A4  35-20-10-2-10-20-35, GD 0,3, MSE", "MSE", [2, 3, 11]),
    ("35-45-2-45-35, GD 0,3, MSE", "MSE", [1, 2, 1]),
    ("35-50-2-50-35, GD 0,3, MSE", "MSE", [1, 2, 2]),
    ("35-60-2-60-35, GD 0,3, MSE", "MSE", [1, 2, 3]),
    ("D1  35-35-15-2-15-35-35, GD 0,3, MSE", "MSE", [7, 1, 2]),
    ("D2  35-45-20-2-20-45-35, GD 0,3, MSE", "MSE", [1, 1, 2]),
    ("35-45-2-45-35, GD 0,03, BCE", "BCE", [0, 1, 0]),
    ("35-45-2-45-35, GD 0,1, BCE", "BCE", [0] * 10),
]


def _style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)


def plot_screen_heatmap(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    image = ax.imshow(SCREEN_MAX_ERRORS, cmap="Blues", vmin=0, vmax=12, aspect="auto")
    ax.set_xticks(range(len(SCREEN_OPTIMIZERS)), SCREEN_OPTIMIZERS)
    ax.set_yticks(range(len(SCREEN_ARCHS)), SCREEN_ARCHS, fontsize=9)
    ax.tick_params(length=0, colors=TEXT_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(SCREEN_MAX_ERRORS.shape[0]):
        for col in range(SCREEN_MAX_ERRORS.shape[1]):
            value = SCREEN_MAX_ERRORS[row, col]
            color = "white" if value >= 7 else TEXT_PRIMARY
            weight = "bold" if value <= 2 else "normal"
            ax.text(col, row, str(value), ha="center", va="center", color=color,
                    fontsize=12, fontweight=weight)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("máximo de píxeles incorrectos (menor es mejor)",
                       color=TEXT_SECONDARY)
    colorbar.outline.set_visible(False)
    ax.set_title("Tamiz grueso: arquitectura x optimizador\n"
                 "MSE, 2.000 épocas, semilla 0 — ninguna combinación llega a 1",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_funnel_seeds(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    labels = [name for name, _, _ in FINALISTS]
    for index, (_, loss, values) in enumerate(FINALISTS):
        color = SERIES_MSE if loss == "MSE" else SERIES_BCE
        values = np.asarray(values, dtype=float)
        # Semillas con el mismo valor se apilan verticalmente para que se vean todas.
        for value in np.unique(values):
            count = int(np.sum(values == value))
            offsets = np.linspace(-0.28, 0.28, count) if count > 1 else np.zeros(1)
            ax.scatter(np.full(count, value), index + offsets, s=30, color=color,
                       edgecolor="white", linewidth=0.6, zorder=3)
        compliant = int(np.sum(values <= 1))
        ax.text(1.01, index, f"{compliant}/{values.size} semillas cumplen",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9, color=TEXT_SECONDARY)
    ax.axvspan(-0.5, 1.5, color="#1baf7a", alpha=0.10, zorder=0)
    ax.axvline(1.5, color="#1baf7a", linewidth=1.2, linestyle="--", zorder=1)
    ax.text(1.65, -0.75, "<- zona que cumple el requisito (máx. <= 1)", ha="left", va="center",
            fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlim(-0.5, 11.7)
    ax.set_ylim(len(FINALISTS) - 0.5, -1.1)
    ax.set_xticks(range(0, 12))
    ax.set_xlabel("máximo de píxeles incorrectos al final, un punto por semilla",
                  color=TEXT_SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    _style_axes(ax)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=SERIES_MSE, label="loss MSE"),
        plt.Line2D([], [], marker="o", linestyle="", color=SERIES_BCE, label="loss BCE"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    ax.set_title("Embudo de selección: robustez frente a la semilla",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.subplots_adjust(left=0.36, right=0.80, top=0.92, bottom=0.11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_funnel_seeds_mse(path: Path) -> None:
    """Resumir la robustez de los finalistas MSE sin desplazar puntos entre filas."""
    mse_finalists = [row for row in FINALISTS if row[1] == "MSE"]
    short_labels = [
        "A3  35-35-2-35-35",
        "A4  35-20-10-2-10-20-35",
        "Ancho 45  35-45-2-45-35",
        "Ancho 50  35-50-2-50-35",
        "Ancho 60  35-60-2-60-35",
        "D1  35-35-15-2-15-35-35",
        "D2  35-45-20-2-20-45-35",
    ]

    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    for index, (_, _, raw_values) in enumerate(mse_finalists):
        values = np.asarray(raw_values, dtype=int)
        for value in np.unique(values):
            count = int(np.sum(values == value))
            ax.scatter(value, index, s=52 + 34 * (count - 1), color=SERIES_MSE,
                       edgecolor="white", linewidth=0.8, zorder=3)
            if count > 1:
                ax.text(value, index, str(count), ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold", zorder=4)
        compliant = int(np.sum(values <= 1))
        ax.text(1.01, index, f"{compliant}/3 cumplen",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=9, color=TEXT_SECONDARY)

    ax.axvspan(-0.5, 1.5, color="#1baf7a", alpha=0.10, zorder=0)
    ax.axvline(1.5, color="#1baf7a", linewidth=1.2, linestyle="--", zorder=1)
    ax.text(1.65, -0.72, "zona que cumple: máximo <= 1", ha="left", va="center",
            fontsize=9, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(short_labels)), short_labels, fontsize=9)
    ax.set_xlim(-0.5, 11.7)
    ax.set_ylim(len(short_labels) - 0.5, -1.05)
    ax.set_xticks(range(0, 12))
    ax.set_xlabel("máximo de píxeles incorrectos al final (menor es mejor)",
                  color=TEXT_SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    _style_axes(ax)
    ax.set_title("Refinamiento de arquitectura con MSE\n"
                 "GD 0,3 y tres semillas por configuración",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.text(0.34, 0.025,
             "Un número dentro de un punto indica semillas con el mismo resultado",
             ha="left", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.34, right=0.83, top=0.86, bottom=0.13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bce_rate_speed(path: Path) -> None:
    """Comparar cuándo las tasas BCE viables alcanzan el requisito."""
    rates = ["0,03", "0,1"]
    first_requirement_epochs = {
        "0,03": [1599, 1501, 1577],
        "0,1": [582, 583, 770],
    }
    colors = ["#6a9bd4", SERIES_BCE]
    x = np.arange(len(rates))
    offsets = np.array([-0.035, 0.0, 0.035])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for index, rate in enumerate(rates):
        values = np.asarray(first_requirement_epochs[rate])
        ax.scatter(np.full(values.size, index) + offsets, values, s=58,
                   color=colors[index],
                   edgecolor="white", linewidth=0.8, zorder=3)
        median = int(np.median(values))
        ax.plot([index - 0.12, index + 0.12], [median, median],
                color=TEXT_PRIMARY, linewidth=2.2, zorder=4)
        ax.text(index, median - 70, f"mediana: {median}", ha="center", va="top",
                fontsize=9, color=TEXT_PRIMARY)

    ax.set_xticks(x, ["GD 0,03", "GD 0,1"])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 1850)
    ax.set_ylabel("primera época con máximo de errores <= 1",
                  color=TEXT_SECONDARY)
    ax.set_title("BCE con tasa 0,1 alcanza el requisito antes\n"
                 "Misma arquitectura y mismas tres inicializaciones",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style_axes(ax)
    fig.text(0.16, 0.025,
             "Cada punto representa una semilla. Menos épocas indica mayor rapidez.",
             ha="left", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.16, right=0.84, top=0.84, bottom=0.16)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bce_rate_curves_seed0(path: Path) -> None:
    """Mostrar la trayectoria completa de las cuatro tasas del tamiz BCE."""
    import csv

    histories = (Path(__file__).resolve().parents[2] / "results"
                 / "bce-gd-rate-calibration-seed0-v1" / "histories")
    series = [
        ("0,01", "0p01", "#7b8794", "lenta"),
        ("0,03", "0p03", "#2a78d6", "viable"),
        ("0,1", "0p1", "#1baf7a", "más rápida"),
        ("0,3", "0p3", "#d43c2f", "inestable"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for rate, tag, color, reading in series:
        source = histories / (
            f"width-45__gradient_descent__lr-{tag}__seed-0.csv"
        )
        with source.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        epochs = np.array([int(row["epoch"]) for row in rows])
        errors = np.array([int(row["max_pixel_errors"]) for row in rows])
        ax.plot(epochs, errors, color=color, linewidth=1.25,
                label=f"tasa {rate}: {reading}", alpha=0.92)

    ax.axhspan(-0.5, 1.5, color="#1baf7a", alpha=0.09, zorder=0)
    ax.axhline(1, color="#1baf7a", linewidth=1.1, linestyle="--", zorder=1)
    ax.text(1980, 1.6, "requisito: máximo <= 1", ha="right", va="bottom",
            fontsize=8, color="#16825d")
    ax.set_xlim(0, 2000)
    ax.set_ylim(-0.5, 33)
    ax.set_xlabel("época", color=TEXT_SECONDARY)
    ax.set_ylabel("máximo de píxeles incorrectos", color=TEXT_SECONDARY)
    ax.set_title("Calibración de la tasa con BCE\n"
                 "Red 35-45-2-45-35, descenso de gradiente y semilla 0",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.grid(color=GRID, linewidth=0.8)
    _style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bce_rate_requirement_seed0(path: Path) -> None:
    """Comparar el primer cumplimiento de las cuatro tasas BCE en semilla 0."""
    rates = ["0,01", "0,03", "0,1", "0,3"]
    first_epoch = [None, 1599, 582, None]
    final_errors = [6, 0, 0, 22]
    colors = ["#7b8794", "#2a78d6", "#1baf7a", "#d43c2f"]
    x = np.arange(len(rates))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(2000, color="#9aa5b1", linewidth=1.1, linestyle="--")
    ax.text(3.45, 2000, "presupuesto: 2.000 épocas", ha="right", va="bottom",
            fontsize=8, color=TEXT_SECONDARY)

    for index, (epoch, errors, color) in enumerate(
            zip(first_epoch, final_errors, colors)):
        if epoch is None:
            ax.scatter(index, 2075, marker="x", s=90, linewidth=2.4,
                       color=color, zorder=3)
            ax.text(index, 1980, f"no alcanzó\n{errors} errores al final",
                    ha="center", va="top", fontsize=9, color=TEXT_SECONDARY)
        else:
            ax.scatter(index, epoch, s=95, color=color, edgecolor="white",
                       linewidth=0.9, zorder=3)
            ax.text(index, epoch + 70, f"época {epoch}\n0 errores al final",
                    ha="center", va="bottom", fontsize=9, color=TEXT_SECONDARY)

    ax.set_xticks(x, [f"tasa {rate}" for rate in rates])
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0, 2180)
    ax.set_ylabel("primera época con máximo de errores <= 1",
                  color=TEXT_SECONDARY)
    ax.set_title("Velocidad de las cuatro tasas con BCE\n"
                 "Red 35-45-2-45-35, descenso de gradiente y semilla 0",
                 color=TEXT_PRIMARY, fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style_axes(ax)
    fig.text(0.16, 0.025,
             "Las cruces indican que la tasa no cumplió dentro del presupuesto.",
             ha="left", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    fig.subplots_adjust(left=0.16, right=0.96, top=0.84, bottom=0.17)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# Resumen de configs/bce_architecture_comparison_seeds0_9.json, copiado a docs/
# porque results/ no se versiona.
BCE_SUMMARY = OUTPUT_DIR / "1a-bce-arquitecturas.csv"
BCE_ROWS = [
    ("A0-direct", "A0  35-2-35"),
    ("A1-width-10", "A1  35-10-2-10-35"),
    ("width-15", "35-15-2-15-35"),
    ("A2-width-20", "A2  35-20-2-20-35"),
    ("A3-width-35", "A3  35-35-2-35-35"),
    ("width-45", "35-45-2-45-35"),
    ("A4-depth-20-10", "A4  35-20-10-2-10-20-35"),
    ("D1-deep-35-15", "D1  35-35-15-2-15-35-35"),
    ("D2-deep-45-20", "D2  35-45-20-2-20-45-35"),
]
BCE_RATES = ["0.1", "0.03"]
CLIP = 12


def _load_bce_results(path: Path) -> tuple[dict, dict]:
    import csv
    errors: dict = {}
    parameters: dict = {}
    with path.open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = (row["architecture"], row["learning_rate"])
            errors.setdefault(key, []).append(int(row["max_pixel_errors"]))
            parameters[row["architecture"]] = int(row["parameters"])
    return errors, parameters


def plot_bce_architectures(path: Path, summary: Path = BCE_SUMMARY) -> None:
    errors, parameters = _load_bce_results(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), sharey=True)
    labels = [f"{name}\n({parameters[key]:,} parám.)".replace(",", ".")
              for key, name in BCE_ROWS]
    for axis, rate in zip(axes, BCE_RATES):
        axis.axvspan(-0.5, 1.5, color="#1baf7a", alpha=0.10, zorder=0)
        axis.axvline(1.5, color="#1baf7a", linewidth=1.2, linestyle="--", zorder=1)
        for index, (key, _) in enumerate(BCE_ROWS):
            values = np.asarray(errors[(key, rate)])
            # Valores mayores que CLIP se agrupan en una columna "> CLIP".
            plotted = np.where(values > CLIP, CLIP + 1.6, values)
            for value in np.unique(plotted):
                count = int(np.sum(plotted == value))
                offsets = (np.linspace(-0.3, 0.3, count) if count > 1
                           else np.zeros(1))
                axis.scatter(np.full(count, value), index + offsets, s=22,
                             marker=">" if value > CLIP else "o",
                             color=SERIES_BCE, edgecolor="white", linewidth=0.5,
                             zorder=3)
            compliant = int(np.sum(values <= 1))
            weight = "bold" if compliant == values.size else "normal"
            axis.text(1.0, index, f" {compliant}/10", transform=axis.get_yaxis_transform(),
                      va="center", ha="left", fontsize=9, color=TEXT_PRIMARY,
                      fontweight=weight)
        axis.set_xlim(-0.5, CLIP + 2.4)
        ticks = list(range(0, CLIP + 1, 2))
        axis.set_xticks(ticks + [CLIP + 1.6], [str(t) for t in ticks] + [f">{CLIP}"])
        axis.set_title(f"GD, tasa {rate.replace('.', ',')}", color=TEXT_PRIMARY,
                       fontsize=11)
        axis.set_xlabel("máximo de píxeles incorrectos (un punto por semilla)",
                        color=TEXT_SECONDARY, fontsize=9)
        axis.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        _style_axes(axis)
    axes[0].set_yticks(range(len(BCE_ROWS)), labels, fontsize=9)
    axes[0].set_ylim(len(BCE_ROWS) - 0.5, -0.6)
    for boundary in (5.5,):
        for axis in axes:
            axis.axhline(boundary, color=TEXT_SECONDARY, linewidth=0.6, alpha=0.5)
    fig.suptitle("Arquitecturas con BCE: 10 semillas, 2.000 épocas\n"
                 "a la derecha de cada panel: semillas que cumplen máx. <= 1",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.subplots_adjust(left=0.20, right=0.95, wspace=0.18, top=0.86, bottom=0.11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_screen_heatmap(OUTPUT_DIR / "1a-tamiz-arquitectura-optimizador.png")
    plot_funnel_seeds(OUTPUT_DIR / "1a-embudo-semillas.png")
    if BCE_SUMMARY.exists():
        plot_bce_architectures(OUTPUT_DIR / "1a-bce-arquitecturas.png")
    print(f"Figuras guardadas en {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


# --- Curvas por época de la comparación con BCE ---------------------------------

BCE_HISTORIES = (Path(__file__).resolve().parents[2] / "results"
                 / "bce-architecture-comparison-seeds0-9" / "histories")
CURVES_CSV = OUTPUT_DIR / "1a-bce-curvas.csv"


def aggregate_bce_curves() -> None:
    """Mediana y cuartiles del máximo de errores por época (10 semillas)."""
    import csv
    if not BCE_HISTORIES.exists():
        return
    rows = []
    for key, _ in BCE_ROWS:
        for rate in BCE_RATES:
            tag = rate.replace(".", "p")
            series = []
            for seed in range(10):
                path = BCE_HISTORIES / f"{key}__gradient_descent__lr-{tag}__seed-{seed}.csv"
                with path.open(encoding="utf-8") as handle:
                    series.append([int(r["max_pixel_errors"]) for r in csv.DictReader(handle)])
            values = np.array(series)
            for epoch in range(0, values.shape[1], 10):
                column = values[:, epoch]
                rows.append({"architecture": key, "learning_rate": rate, "epoch": epoch,
                             "median": float(np.median(column)),
                             "q25": float(np.percentile(column, 25)),
                             "q75": float(np.percentile(column, 75))})
    with CURVES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_bce_curves(path: Path) -> None:
    import csv
    if not CURVES_CSV.exists():
        return
    with CURVES_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    panels = [
        ("Una capa oculta · GD 0,1", "0.1",
         [("A1-width-10", "10"), ("width-15", "15"), ("A2-width-20", "20"),
          ("A3-width-35", "35"), ("width-45", "45")]),
        ("Profundas · GD 0,03", "0.03",
         [("A4-depth-20-10", "A4 20-10"), ("D1-deep-35-15", "D1 35-15"),
          ("D2-deep-45-20", "D2 45-20")]),
    ]
    palette = ["#9aa5b1", "#eda100", "#1baf7a", "#eb6834", "#2a78d6"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for axis, (title, rate, series) in zip(axes, panels):
        colors = palette[-len(series):]
        for (key, label), color in zip(series, colors):
            data = [r for r in rows if r["architecture"] == key and r["learning_rate"] == rate]
            epochs = np.array([int(r["epoch"]) for r in data])
            median = np.array([float(r["median"]) for r in data])
            axis.fill_between(epochs, [float(r["q25"]) for r in data],
                              [float(r["q75"]) for r in data], color=color, alpha=0.15,
                              linewidth=0)
            axis.plot(epochs, median, color=color, linewidth=2, label=label)
        axis.axhline(1, color="#1baf7a", linestyle="--", linewidth=1)
        axis.set_yscale("symlog", linthresh=2)
        axis.set_yticks([0, 1, 2, 5, 10, 20])
        axis.set_yticklabels(["0", "1", "2", "5", "10", "20"])
        axis.set_xlabel("época", color=TEXT_SECONDARY)
        axis.set_title(title, fontsize=10, color=TEXT_PRIMARY)
        axis.legend(frameon=False, fontsize=9, title="arquitectura", title_fontsize=9)
        axis.grid(color=GRID, linewidth=0.8)
        _style_axes(axis)
    axes[0].set_ylabel("máximo de píxeles incorrectos", color=TEXT_SECONDARY)
    fig.suptitle("Curvas de entrenamiento con BCE: mediana y cuartiles de 10 semillas",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    aggregate_bce_curves()
    plot_bce_curves(OUTPUT_DIR / "1a-bce-curvas.png")
