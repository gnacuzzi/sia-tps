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
    return Path(level).stem.replace("_", " ").title()


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

    time_figure = _time_boxplot(plt, levels, methods, grouped)
    generated.append(
        _save(plt, time_figure, output_dir, "elapsed_time_boxplot", image_format, dpi)
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
    summaries: Dict[Tuple[str, Method], Tuple[Optional[float], float, float]] = {}
    positive_medians = []
    for level in levels:
        for method in methods:
            runs = grouped[level].get(method, [])
            if successes_only:
                runs = [run for run in runs if run["status"] == "success"]
            values = [value for run in runs if (value := _number(run[field])) is not None]
            summary = _median_and_iqr(values)
            summaries[(level, method)] = summary
            if summary[0] is not None and summary[0] > 0:
                positive_medians.append(summary[0])

    logarithmic = (
        bool(positive_medians)
        and max(positive_medians) / min(positive_medians) >= 20
    )
    figure, axes = plt.subplots(
        len(levels),
        1,
        figsize=(11, max(3.2 * len(levels), 4.5)),
        squeeze=False,
        sharex=True,
    )

    for axis, level in zip(axes[:, 0], levels):
        for position, method in enumerate(methods):
            median, lower_error, upper_error = summaries[(level, method)]
            color = COLORS.get(method[0], "#8E8E8E")
            if median is None:
                axis.text(
                    0.01,
                    position,
                    "sin solución",
                    transform=axis.get_yaxis_transform(),
                    va="center",
                    fontsize=8,
                )
                continue
            axis.barh(
                position,
                median,
                xerr=[[lower_error], [upper_error]],
                color=color,
                alpha=0.9,
                error_kw={"ecolor": "#333333", "capsize": 3, "elinewidth": 1},
            )
            axis.annotate(
                _format_value(median),
                (median, position),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )

        axis.set_yticks(range(len(methods)), [method_label(method) for method in methods])
        axis.invert_yaxis()
        axis.set_title(level_label(level), loc="left", fontsize=11, weight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        if logarithmic:
            axis.set_xscale("log")
        axis.margins(x=0.12)

    axes[-1, 0].set_xlabel(x_label + (" (escala logarítmica)" if logarithmic else ""))
    figure.suptitle(title + " (mediana e IQR)", fontsize=15, weight="bold")
    figure.tight_layout()
    return figure


def _time_boxplot(plt, levels, methods, grouped):
    all_times = [
        float(run["elapsed_seconds"])
        for level_groups in grouped.values()
        for runs in level_groups.values()
        for run in runs
    ]
    logarithmic = max(all_times) / min(all_times) >= 20
    figure, axes = plt.subplots(
        len(levels),
        1,
        figsize=(11, max(3.2 * len(levels), 4.5)),
        squeeze=False,
        sharex=True,
    )
    labels = [method_label(method) for method in methods]

    for axis, level in zip(axes[:, 0], levels):
        values = [
            [float(run["elapsed_seconds"]) for run in grouped[level].get(method, [])]
            for method in methods
        ]
        boxes = axis.boxplot(
            values,
            vert=False,
            tick_labels=labels,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#222222", "linewidth": 1.5},
        )
        for box, method in zip(boxes["boxes"], methods):
            box.set_facecolor(COLORS.get(method[0], "#8E8E8E"))
            box.set_alpha(0.9)
        axis.invert_yaxis()
        axis.set_title(level_label(level), loc="left", fontsize=11, weight="bold")
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        if logarithmic:
            axis.set_xscale("log")
        axis.margins(x=0.08)

    axes[-1, 0].set_xlabel(
        "Segundos" + (" (escala logarítmica)" if logarithmic else "")
    )
    figure.suptitle("Tiempo de búsqueda: distribución de repeticiones", fontsize=15, weight="bold")
    figure.tight_layout()
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
            color = "white" if rates[row_index][column_index] < 0.35 else "#202020"
            axis.text(
                column_index,
                row_index,
                annotation,
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                weight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Proporción de éxitos")
    axis.set_title("Resultados por método y nivel", fontsize=15, weight="bold", pad=15)
    axis.set_xlabel("S = success · C = cutoff · F = failure")
    figure.tight_layout()
    return figure


def _median_and_iqr(values: Sequence[float]) -> Tuple[Optional[float], float, float]:
    if not values:
        return None, 0, 0
    median = statistics.median(values)
    first_quartile = _percentile(values, 0.25)
    third_quartile = _percentile(values, 0.75)
    return median, median - first_quartile, third_quartile - median


def _percentile(values: Sequence[float], proportion: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * proportion
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


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
