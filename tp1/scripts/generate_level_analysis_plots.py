"""Generate narrative plots for the three main Sokoban experiment levels."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = PROJECT_ROOT / "results" / "comparacion_principal_summary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "plots" / "level_analysis"

LEVEL_ORDER = ("level_03", "level_02", "level_04")
LEVEL_LABELS = {
    "level_03": "Base compacta",
    "level_02": "Intermedio guiado",
    "level_04": "Difícil sensible",
}
SHORT_LEVEL_LABELS = {
    "level_03": "Base",
    "level_02": "Intermedio",
    "level_04": "Difícil",
}

MMM = "minimum_matching_manhattan"
SPA = "shortest_push_access"
METHOD_ORDER = (
    ("bfs", "null"),
    ("dfs", "null"),
    ("greedy", MMM),
    ("greedy", SPA),
    ("astar", MMM),
    ("astar", SPA),
)
METHOD_LABELS = {
    ("bfs", "null"): "BFS",
    ("dfs", "null"): "DFS",
    ("greedy", MMM): "Greedy · MMM",
    ("greedy", SPA): "Greedy · SPA",
    ("astar", MMM): "A* · MMM",
    ("astar", SPA): "A* · SPA",
}
METHOD_SHORT_LABELS = {
    ("bfs", "null"): "BFS",
    ("dfs", "null"): "DFS",
    ("greedy", MMM): "G·MMM",
    ("greedy", SPA): "G·SPA",
    ("astar", MMM): "A*·MMM",
    ("astar", SPA): "A*·SPA",
}
COLORS = {
    "bfs": "#4C78A8",
    "dfs": "#F58518",
    "greedy": "#54A24B",
    "astar": "#E45756",
}

BACKGROUND = "#FFFFFF"
FOREGROUND = "#202020"
MUTED = "#666666"
GRID = "#D9D9D9"

Method = Tuple[str, str]
Row = Mapping[str, str]
Index = Dict[Tuple[str, Method], Row]


def read_summary(path: Path) -> Index:
    with path.open(encoding="utf-8", newline="") as summary_file:
        rows = list(csv.DictReader(summary_file))
    if not rows:
        raise ValueError(f"CSV without data: {path}")

    index: Index = {}
    for row in rows:
        level = Path(row["level"]).stem
        method = (row["algorithm"], row["heuristic"])
        index[(level, method)] = row

    missing = [
        (level, method)
        for level in LEVEL_ORDER
        for method in METHOD_ORDER
        if (level, method) not in index
    ]
    if missing:
        raise ValueError(f"Missing summary cases: {missing}")
    return index


def value(index: Index, level: str, method: Method, field: str) -> float:
    return float(index[(level, method)][field])


def configure_style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": FOREGROUND,
            "axes.titlecolor": FOREGROUND,
            "xtick.color": FOREGROUND,
            "ytick.color": FOREGROUND,
            "text.color": FOREGROUND,
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
        }
    )


def style_axis(axis, *, grid_axis: str = "x") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.grid(axis=grid_axis, color=GRID, alpha=0.5, linewidth=0.8)
    axis.set_axisbelow(True)


def decimal(number: float, digits: int = 2) -> str:
    return f"{number:.{digits}f}".replace(".", ",")


def ratio_label(number: float) -> str:
    if number >= 10:
        digits = 1
    elif number >= 0.1:
        digits = 2
    else:
        digits = 3
    return f"{decimal(number, digits)}×"


def plot_difficulty_dimensions(plt, index: Index):
    bfs = ("bfs", "null")
    optimal_costs = [
        value(index, level, bfs, "solution_cost_median")
        for level in LEVEL_ORDER
    ]
    bfs_expanded = [
        value(index, level, bfs, "expanded_nodes_median")
        for level in LEVEL_ORDER
    ]
    sensitivity = []
    for level, optimal in zip(LEVEL_ORDER, optimal_costs):
        worst = max(
            value(index, level, method, "solution_cost_median")
            for method in METHOD_ORDER
        )
        sensitivity.append(worst / optimal)

    figure, axes = plt.subplots(1, 3, figsize=(12, 6.75))
    labels = [SHORT_LEVEL_LABELS[level] for level in LEVEL_ORDER]
    colors = ["#7A9E65", "#D19A4A", "#B85C5C"]
    panels = (
        (
            optimal_costs,
            "Longitud del plan",
            "Costo óptimo (movimientos)",
            False,
            lambda number: f"{number:.0f}",
        ),
        (
            bfs_expanded,
            "Trabajo de búsqueda",
            "Nodos expandidos por BFS",
            False,
            lambda number: f"{decimal(number / 1000, 1)} mil",
        ),
        (
            sensitivity,
            "Sensibilidad al método",
            "Peor costo / costo BFS",
            True,
            lambda number: f"{decimal(number, 2)}×",
        ),
    )

    for axis, (values, title, x_label, logarithmic, formatter) in zip(
        axes, panels
    ):
        positions = range(len(values))
        bars = axis.barh(positions, values, color=colors, alpha=0.9)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(title, weight="bold", pad=14)
        axis.set_xlabel(x_label)
        if logarithmic:
            axis.set_xscale("log")
            axis.set_xlim(1, 60)
            axis.axvline(1, color=MUTED, linestyle="--", linewidth=1)
        axis.bar_label(
            bars,
            labels=[formatter(number) for number in values],
            padding=5,
            fontsize=12,
            weight="bold",
        )
        axis.margins(x=0.24)
        style_axis(axis)

    figure.suptitle(
        "Los niveles son difíciles por razones distintas",
        fontsize=21,
        weight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        "La longitud del camino, el trabajo de búsqueda y la sensibilidad no crecen juntas",
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.87), w_pad=2.5)
    return figure


def plot_path_length_vs_search(plt, index: Index):
    from matplotlib.ticker import FuncFormatter

    bfs = ("bfs", "null")
    figure, axis = plt.subplots(figsize=(12, 6.75))

    offsets = {
        "level_03": (-18, 16),
        "level_02": (-8, -34),
        "level_04": (14, 12),
    }
    for level, color in zip(
        LEVEL_ORDER,
        ("#7A9E65", "#D19A4A", "#B85C5C"),
    ):
        cost = value(index, level, bfs, "solution_cost_median")
        expanded = value(index, level, bfs, "expanded_nodes_median")
        axis.scatter(cost, expanded, s=230, color=color, zorder=3)
        dx, dy = offsets[level]
        axis.annotate(
            SHORT_LEVEL_LABELS[level],
            (cost, expanded),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=14,
            weight="bold",
            ha="left",
        )

    base_cost = value(index, "level_03", bfs, "solution_cost_median")
    base_expanded = value(index, "level_03", bfs, "expanded_nodes_median")
    intermediate_cost = value(index, "level_02", bfs, "solution_cost_median")
    intermediate_expanded = value(
        index, "level_02", bfs, "expanded_nodes_median"
    )
    axis.annotate(
        "Más movimientos, pero menos estados explorados",
        xy=(intermediate_cost, intermediate_expanded),
        xytext=(base_cost + 7, base_expanded + 7500),
        arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 1.5},
        fontsize=12,
        color=MUTED,
    )

    axis.set_xlabel("Costo óptimo de BFS (movimientos)")
    axis.set_ylabel("Nodos expandidos por BFS")
    axis.set_xlim(35, 86)
    axis.set_ylim(25000, 70000)
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda number, _: f"{number:,.0f}".replace(",", "."))
    )
    axis.set_title(
        "Un camino más largo no implica una búsqueda mayor",
        fontsize=21,
        weight="bold",
        pad=24,
    )
    axis.text(
        0.5,
        1.01,
        "El nivel intermedio exige más acciones, pero sus corredores restringen las alternativas",
        transform=axis.transAxes,
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    style_axis(axis, grid_axis="both")
    figure.tight_layout(pad=2.2)
    return figure


def plot_hard_level_sensitivity(plt, index: Index):
    level = "level_04"
    selected = (
        ("bfs", "null"),
        ("dfs", "null"),
        ("greedy", MMM),
        ("astar", MMM),
    )
    bfs_cost = value(index, level, ("bfs", "null"), "solution_cost_median")
    bfs_expanded = value(
        index, level, ("bfs", "null"), "expanded_nodes_median"
    )
    cost_ratios = [
        value(index, level, method, "solution_cost_median") / bfs_cost
        for method in selected
    ]
    work_ratios = [
        value(index, level, method, "expanded_nodes_median") / bfs_expanded
        for method in selected
    ]
    labels = [METHOD_LABELS[method] for method in selected]
    colors = [COLORS[method[0]] for method in selected]

    figure, axes = plt.subplots(1, 2, figsize=(12, 6.75))
    panels = (
        (axes[0], cost_ratios, "Calidad de la solución", "Costo / costo BFS"),
        (axes[1], work_ratios, "Trabajo de búsqueda", "Expandidos / expandidos BFS"),
    )
    for axis, values, title, x_label in panels:
        positions = range(len(values))
        bars = axis.barh(positions, values, color=colors, alpha=0.9)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xscale("log")
        axis.axvline(1, color=MUTED, linestyle="--", linewidth=1.2)
        axis.set_title(title, weight="bold", pad=14)
        axis.set_xlabel(x_label + " (escala log)")
        axis.bar_label(
            bars,
            labels=[ratio_label(number) for number in values],
            padding=5,
            fontsize=12,
            weight="bold",
        )
        axis.margins(x=0.22)
        style_axis(axis)

    axes[0].set_xlim(0.8, 65)
    axes[1].set_xlim(0.008, 1.8)
    figure.suptitle(
        "El nivel difícil amplifica la elección del método",
        fontsize=21,
        weight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        "DFS multiplica el costo 37,9×; Greedy·MMM reduce el trabajo 74,6× por 18,5% más movimientos",
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.87), w_pad=3)
    return figure


def plot_quality_work_tradeoff(plt, index: Index):
    figure, axes = plt.subplots(1, 3, figsize=(12, 6.75), sharex=True, sharey=True)
    label_offsets = {
        "level_03": {
            ("bfs", "null"): (8, 14),
            ("dfs", "null"): (8, 28),
            ("greedy", MMM): (8, 12),
            ("greedy", SPA): (8, -18),
            ("astar", MMM): (-82, 28),
            ("astar", SPA): (-82, -28),
        },
        "level_02": {
            ("bfs", "null"): (8, -28),
            ("dfs", "null"): (8, 12),
            ("greedy", MMM): (8, 12),
            ("greedy", SPA): (8, 10),
            ("astar", MMM): (-82, 28),
            ("astar", SPA): (-82, -28),
        },
        "level_04": {
            ("bfs", "null"): (8, 14),
            ("dfs", "null"): (8, -18),
            ("greedy", MMM): (8, 12),
            ("greedy", SPA): (8, -18),
            ("astar", MMM): (-82, 28),
            ("astar", SPA): (-82, -28),
        },
    }

    for axis, level in zip(axes, LEVEL_ORDER):
        bfs_cost = value(index, level, ("bfs", "null"), "solution_cost_median")
        bfs_expanded = value(
            index, level, ("bfs", "null"), "expanded_nodes_median"
        )
        for method in METHOD_ORDER:
            quality = value(index, level, method, "solution_cost_median") / bfs_cost
            work = value(index, level, method, "expanded_nodes_median") / bfs_expanded
            axis.scatter(
                work,
                quality,
                s=85,
                color=COLORS[method[0]],
                zorder=3,
            )
            axis.annotate(
                METHOD_SHORT_LABELS[method],
                (work, quality),
                xytext=label_offsets[level][method],
                textcoords="offset points",
                fontsize=9.5,
                color=FOREGROUND,
            )

        axis.axvline(1, color=GRID, linestyle="--", linewidth=1)
        axis.axhline(1, color=GRID, linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(0.008, 1.8)
        axis.set_ylim(0.93, 55)
        axis.set_title(LEVEL_LABELS[level], weight="bold", pad=12)
        style_axis(axis, grid_axis="both")

    axes[0].set_ylabel("Calidad relativa: costo / costo BFS")
    figure.supxlabel("Trabajo relativo: expandidos / expandidos BFS (escala logarítmica)")
    figure.suptitle(
        "Calidad y trabajo muestran compromisos distintos",
        fontsize=21,
        weight="bold",
        y=0.97,
    )
    figure.text(
        0.5,
        0.91,
        "Más cerca del extremo inferior izquierdo = menor costo y menos expansiones",
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    figure.tight_layout(rect=(0.03, 0.08, 0.99, 0.87), w_pad=2)
    return figure


def plot_astar_spa_cost(plt, index: Index):
    metrics = (
        ("expanded_nodes_median", "Expandidos"),
        ("max_frontier_size_median", "Frontera máxima"),
        ("elapsed_seconds_median", "Tiempo"),
    )
    positions = range(len(LEVEL_ORDER))
    width = 0.22
    colors = ("#4C78A8", "#54A24B", "#F58518")

    figure, axis = plt.subplots(figsize=(12, 6.75))
    for metric_index, ((field, label), color) in enumerate(zip(metrics, colors)):
        ratios = [
            value(index, level, ("astar", SPA), field)
            / value(index, level, ("astar", MMM), field)
            for level in LEVEL_ORDER
        ]
        offsets = [
            position + (metric_index - 1) * width for position in positions
        ]
        bars = axis.bar(offsets, ratios, width, color=color, label=label)
        axis.bar_label(
            bars,
            labels=[f"{decimal(ratio, 2)}×" for ratio in ratios],
            padding=4,
            fontsize=10,
        )

    axis.axhline(1, color=MUTED, linestyle="--", linewidth=1.2)
    axis.set_ylim(0, 5.35)
    axis.set_xticks(positions, [SHORT_LEVEL_LABELS[level] for level in LEVEL_ORDER])
    axis.set_ylabel("Razón SPA / MMM en A*")
    axis.set_title(
        "SPA no mejora el costo y suele pagar más memoria y tiempo",
        fontsize=21,
        weight="bold",
        pad=24,
    )
    axis.text(
        0.5,
        1.01,
        "Debajo de 1 mejora a MMM; por encima de 1 consume más",
        transform=axis.transAxes,
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    axis.legend(frameon=False, ncols=3, loc="upper left")
    style_axis(axis, grid_axis="y")
    figure.tight_layout(pad=2.2)
    return figure


def generate_plots(index: Index, output_dir: Path, dpi: int = 200) -> Sequence[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configure_style(plt)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = (
        ("difficulty_dimensions", plot_difficulty_dimensions(plt, index)),
        ("path_length_vs_search", plot_path_length_vs_search(plt, index)),
        ("hard_level_sensitivity", plot_hard_level_sensitivity(plt, index)),
        ("quality_work_tradeoff", plot_quality_work_tradeoff(plt, index)),
        ("astar_spa_vs_mmm", plot_astar_spa_cost(plt, index)),
    )
    generated = []
    for filename, figure in figures:
        path = output_dir / f"{filename}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        generated.append(path)
    return generated


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate narrative plots for the three experiment levels."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Summary CSV produced by run_experiments.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination directory",
    )
    parser.add_argument("--dpi", type=int, default=200)
    arguments = parser.parse_args(argv)

    try:
        generated = generate_plots(
            read_summary(arguments.summary),
            arguments.output_dir,
            arguments.dpi,
        )
    except (OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
