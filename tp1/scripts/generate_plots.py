"""Generate comparison charts from raw Sokoban experiment repetitions."""

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {
    "level",
    "algorithm",
    "heuristic",
    "repetition",
    "status",
    "solution_cost",
    "expanded_nodes",
    "max_frontier_size",
    "elapsed_seconds",
}

METRICS = (
    ("solution_cost", "Costo de la solución", "Movimientos", "solution_cost", True),
    ("expanded_nodes", "Nodos expandidos", "Nodos", "expanded_nodes", False),
    (
        "max_frontier_size",
        "Máximo tamaño de la frontera",
        "Nodos en la frontera",
        "max_frontier",
        False,
    ),
)

METHOD_ORDER = {
    ("bfs", "null"): 0,
    ("dfs", "null"): 1,
    ("greedy", "minimum_matching_manhattan"): 2,
    ("greedy", "shortest_push_access"): 3,
    ("astar", "minimum_matching_manhattan"): 4,
    ("astar", "shortest_push_access"): 5,
}

HEURISTIC_LABELS = {
    "minimum_matching_manhattan": "MMM",
    "shortest_push_access": "SPA",
    "deadlock_aware_reverse_push_matching": "Reverse Push",
    "pair_pattern_database_matching": "Pair PDB",
}

COLORS = {
    "bfs": "#4C78A8",
    "dfs": "#F58518",
    "greedy": "#54A24B",
    "astar": "#E45756",
}

LEVEL_LABELS = {
    "level_03": "Bajo",
    "level_02": "Intermedio",
    "level_04": "Difícil",
}

Method = Tuple[str, str]
Run = Mapping[str, str]
GroupedRuns = Dict[str, Dict[Method, List[Run]]]


def read_raw_results(path: Path) -> List[Dict[str, str]]:
    """Read and validate the raw CSV produced by run_experiments.py."""

    with path.open(encoding="utf-8", newline="") as results_file:
        reader = csv.DictReader(results_file)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(
                "The CSV is not a raw experiment file; missing columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)

    if not rows:
        raise ValueError("The experiment CSV has no data rows")
    return rows


def method_label(method: Method) -> str:
    algorithm, heuristic = method
    if algorithm == "astar":
        algorithm_label = "A*"
    else:
        algorithm_label = algorithm.upper().replace("GREEDY", "Greedy")
    if heuristic in {"", "null"}:
        return algorithm_label
    return f"{algorithm_label} · {HEURISTIC_LABELS.get(heuristic, heuristic)}"


def level_label(level: str) -> str:
    stem = Path(level).stem
    return LEVEL_LABELS.get(stem, stem.replace("_", " ").title())


def _comparison_axes(plt, *, compact: bool = False):
    figure, mosaic = plt.subplot_mosaic(
        [
            ["low", "."],
            ["low", "hard"],
            ["intermediate", "hard"],
            ["intermediate", "."],
        ],
        figsize=(16, 9),
        sharex=True,
        empty_sentinel=".",
        gridspec_kw={
            "width_ratios": [1, 1],
            "hspace": 0.30 if compact else 0.28,
            "wspace": 0.42 if compact else 0.22,
        },
    )
    return figure, [mosaic["low"], mosaic["intermediate"], mosaic["hard"]]


def group_runs(
    rows: Sequence[Run], level_order: Sequence[str]
) -> Tuple[List[str], List[Method], GroupedRuns]:
    grouped: GroupedRuns = {}
    methods = set()
    for row in rows:
        method = (row["algorithm"].lower(), row["heuristic"])
        methods.add(method)
        grouped.setdefault(row["level"], {}).setdefault(method, []).append(row)

    requested = [name.removesuffix(".txt") for name in level_order]
    rank = {name: index for index, name in enumerate(requested)}
    levels = sorted(
        grouped,
        key=lambda level: (rank.get(Path(level).stem, len(rank)), Path(level).stem),
    )
    ordered_methods = sorted(
        methods,
        key=lambda method: (
            METHOD_ORDER.get(method, len(METHOD_ORDER)),
            method_label(method),
        ),
    )
    return levels, ordered_methods, grouped


def generate_charts(
    rows: Sequence[Run],
    output_dir: Path,
    image_format: str = "png",
    dpi: int = 180,
    level_order: Sequence[str] = (),
) -> List[Path]:
    """Create five requested charts and return their output paths."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 14,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    levels, methods, grouped = group_runs(rows, level_order)
    generated: List[Path] = []

    for field, title, x_label, filename, successes_only in METRICS:
        figure = _metric_figure(
            plt,
            levels,
            methods,
            grouped,
            field,
            title,
            x_label,
            successes_only,
        )
        generated.append(_save(plt, figure, output_dir, filename, image_format, dpi))

    time_figure = _time_repetitions(plt, levels, methods, grouped)
    generated.append(
        _save(
            plt,
            time_figure,
            output_dir,
            "elapsed_time_repetitions",
            image_format,
            dpi,
        )
    )

    outcome_figure = _outcome_matrix(plt, levels, methods, grouped)
    generated.append(
        _save(plt, outcome_figure, output_dir, "outcome_matrix", image_format, dpi)
    )
    return generated


def _metric_figure(
    plt,
    levels: Sequence[str],
    methods: Sequence[Method],
    grouped: GroupedRuns,
    field: str,
    title: str,
    x_label: str,
    successes_only: bool,
):
    summaries: Dict[Tuple[str, Method], Optional[float]] = {}
    positive_values = []
    for level in levels:
        for method in methods:
            runs = grouped[level].get(method, [])
            if successes_only:
                runs = [run for run in runs if run["status"] == "success"]
            values = [value for run in runs if (value := _number(run[field])) is not None]
            summary = statistics.mean(values) if values else None
            summaries[(level, method)] = summary
            if summary is not None and summary > 0:
                positive_values.append(summary)

    logarithmic = (
        bool(positive_values)
        and max(positive_values) / min(positive_values) >= 20
    )
    is_solution_cost = field == "solution_cost"
    figure, axes = _comparison_axes(plt, compact=is_solution_cost)
    panel_title_size = 19 if is_solution_cost else 17
    tick_size = 15 if is_solution_cost else 14
    value_size = 15 if is_solution_cost else 13

    for axis, level in zip(axes, levels):
        for position, method in enumerate(methods):
            value = summaries[(level, method)]
            color = COLORS.get(method[0], "#8E8E8E")
            if value is None:
                axis.text(
                    0.01,
                    position,
                    "sin solución",
                    transform=axis.get_yaxis_transform(),
                    va="center",
                    fontsize=12,
                )
                continue
            axis.barh(
                position,
                value,
                color=color,
                alpha=0.9,
            )
            axis.annotate(
                _format_value(value),
                (value, position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=value_size,
            )

        axis.set_yticks(range(len(methods)), [method_label(method) for method in methods])
        axis.invert_yaxis()
        axis.set_title(
            level_label(level),
            loc="left",
            fontsize=panel_title_size,
            weight="bold",
            pad=5,
        )
        axis.tick_params(axis="both", labelsize=tick_size)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        if logarithmic:
            axis.set_xscale("log")
        axis.margins(x=0.08 if is_solution_cost else 0.12)

    figure.supxlabel(
        x_label + (" (escala logarítmica)" if logarithmic else ""),
        fontsize=17 if is_solution_cost else 15,
    )
    figure.suptitle(
        title,
        fontsize=25 if is_solution_cost else 22,
        weight="bold",
        y=0.975 if is_solution_cost else 0.985,
    )
    if is_solution_cost:
        figure.subplots_adjust(left=0.11, right=0.985, bottom=0.08, top=0.91)
    else:
        figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.94), w_pad=3, h_pad=2)
    return figure


def _time_repetitions(plt, levels, methods, grouped):
    all_times = [
        float(run["elapsed_seconds"])
        for level_groups in grouped.values()
        for runs in level_groups.values()
        for run in runs
    ]
    logarithmic = max(all_times) / min(all_times) >= 20
    figure, axes = _comparison_axes(plt)
    labels = [method_label(method) for method in methods]

    for axis, level in zip(axes, levels):
        for position, method in enumerate(methods):
            values = [
                float(run["elapsed_seconds"])
                for run in grouped[level].get(method, [])
            ]
            if not values:
                continue
            mean = statistics.mean(values)
            deviation = statistics.stdev(values) if len(values) > 1 else 0
            lower_deviation = min(deviation, mean * 0.95)
            axis.barh(
                position,
                mean,
                xerr=[[lower_deviation], [deviation]],
                color=COLORS.get(method[0], "#8E8E8E"),
                alpha=0.88,
                error_kw={
                    "ecolor": "#222222",
                    "capsize": 4,
                    "elinewidth": 1.5,
                    "capthick": 1.5,
                },
            )
            axis.annotate(
                f"{mean:.3f} s" if mean < 1 else f"{mean:.2f} s",
                (mean + deviation, position),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=12,
            )
        axis.set_yticks(range(len(methods)), labels)
        axis.invert_yaxis()
        axis.set_title(level_label(level), loc="left", fontsize=17, weight="bold")
        axis.tick_params(axis="both", labelsize=14)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        if logarithmic:
            axis.set_xscale("log")
        axis.margins(x=0.16)

    figure.supxlabel(
        "Segundos" + (" (escala logarítmica)" if logarithmic else "")
    )
    figure.suptitle(
        "Tiempo promedio de búsqueda",
        fontsize=22,
        weight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.935,
        "Barra = promedio de 10 ejecuciones · error = una desviación estándar",
        ha="center",
        fontsize=13,
        color="#555555",
    )
    figure.tight_layout(rect=(0.02, 0.04, 0.99, 0.88), w_pad=3, h_pad=2)
    return figure


def _outcome_matrix(plt, levels, methods, grouped):
    rates = []
    annotations = []
    for method in methods:
        rate_row = []
        annotation_row = []
        for level in levels:
            runs = grouped[level].get(method, [])
            successes = sum(run["status"] == "success" for run in runs)
            cutoffs = sum(run["status"] == "cutoff" for run in runs)
            failures = sum(run["status"] == "failure" for run in runs)
            rate_row.append(successes / len(runs) if runs else 0)
            annotation_row.append(f"{successes} S · {cutoffs} C · {failures} F")
        rates.append(rate_row)
        annotations.append(annotation_row)

    figure, axis = plt.subplots(
        figsize=(max(7, 2.4 * len(levels)), max(4.5, 0.8 * len(methods)))
    )
    image = axis.imshow(rates, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(levels)), [level_label(level) for level in levels])
    axis.set_yticks(range(len(methods)), [method_label(method) for method in methods])
    for row_index, row in enumerate(annotations):
        for column_index, annotation in enumerate(row):
            rate = rates[row_index][column_index]
            color = "white" if rate < 0.35 or rate > 0.75 else "#202020"
            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=color,
                fontsize=12,
                weight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Proporción de éxitos", fontsize=15)
    colorbar.ax.tick_params(labelsize=12)
    axis.set_title("Resultados por método y nivel", fontsize=22, weight="bold", pad=18)
    axis.set_xlabel("S = success · C = cutoff · F = failure", fontsize=15)
    axis.tick_params(axis="both", labelsize=14)
    figure.tight_layout()
    return figure


def _number(value: str) -> Optional[float]:
    if value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _format_value(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def _save(plt, figure, output_dir, filename, image_format, dpi):
    path = output_dir / f"{filename}.{image_format}"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate charts from raw run_experiments repetitions."
    )
    parser.add_argument("results_csv", type=Path, help="Raw experiment CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "plots",
        help="Chart destination (default: results/plots)",
    )
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--level-order",
        nargs="*",
        default=(),
        metavar="LEVEL",
        help="Optional order using file stems, e.g. level_03 level_02 level_04",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        rows = read_raw_results(arguments.results_csv)
        paths = generate_charts(
            rows,
            arguments.output_dir,
            image_format=arguments.format,
            dpi=arguments.dpi,
            level_order=arguments.level_order,
        )
    except (ImportError, OSError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print("Generated charts:")
    for path in paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
