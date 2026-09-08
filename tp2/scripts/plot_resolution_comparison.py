#!/usr/bin/env python3
"""Create the presentation chart for the working-resolution experiment."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = PROJECT_ROOT / "output" / "studies" / "resolution-v2"
DEFAULT_RESOLUTIONS = (32, 48, 64)
COLORS = ("#667983", "#e69f00", "#1874b4", "#2ca25f", "#9467bd")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot convergence and final NMSE for selected resolutions."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=DEFAULT_STUDY / "records" / "resolution.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_STUDY
            / "figures"
            / "resolution"
            / "resolution-comparison-32-48-64.png"
        ),
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
    )
    args = parser.parse_args(argv)

    records = _read_records(args.records, args.resolutions)
    _validate(records, args.resolutions)
    _render(records, tuple(args.resolutions), args.output)
    print(f"Gráfico: {args.output}")
    return 0


def _read_records(
    path: Path, resolutions: Sequence[int]
) -> Tuple[Mapping[str, str], ...]:
    wanted = {f"side_{resolution}" for resolution in resolutions}
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(
            row
            for row in csv.DictReader(stream)
            if row.get("phase") == "resolution" and row.get("condition") in wanted
        )


def _validate(
    records: Sequence[Mapping[str, str]], resolutions: Sequence[int]
) -> None:
    present = {int(row["condition"].removeprefix("side_")) for row in records}
    missing = sorted(set(resolutions) - present)
    if missing:
        raise ValueError(f"faltan resultados para las resoluciones: {missing}")


def _historical_curves(
    records: Sequence[Mapping[str, str]],
) -> Dict[int, DefaultDict[int, List[float]]]:
    curves: Dict[int, DefaultDict[int, List[float]]] = {}
    for record in records:
        resolution = int(record["condition"].removeprefix("side_"))
        metrics_path = Path(record["run_directory"]) / "metrics.csv"
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            metrics = list(csv.DictReader(stream))
        initial_error = float(metrics[0]["best_error"])
        historical_best = float("inf")
        resolution_curve = curves.setdefault(resolution, defaultdict(list))
        for row in metrics:
            historical_best = min(historical_best, float(row["best_error"]))
            resolution_curve[int(row["generation"])].append(
                historical_best / initial_error
            )
    return curves


def _render(
    records: Sequence[Mapping[str, str]],
    resolutions: Sequence[int],
    output: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "faltan las dependencias para gráficos; instalá el grupo opcional 'plot'"
        ) from error

    curves = _historical_curves(records)
    final_errors: Dict[int, List[float]] = defaultdict(list)
    for row in records:
        resolution = int(row["condition"].removeprefix("side_"))
        final_errors[resolution].append(float(row["best_error"]) * 1_000)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": (1.55, 1)},
    )
    convergence_axis, boxplot_axis = axes

    for index, resolution in enumerate(resolutions):
        color = COLORS[index % len(COLORS)]
        points = curves[resolution]
        generations = sorted(points)
        means = np.array([statistics.fmean(points[g]) for g in generations])
        deviations = np.array(
            [statistics.stdev(points[g]) if len(points[g]) > 1 else 0.0 for g in generations]
        )
        convergence_axis.plot(
            generations,
            means,
            color=color,
            linewidth=2.2,
            label=f"{resolution} px",
        )
        convergence_axis.fill_between(
            generations,
            np.maximum(0, means - deviations),
            means + deviations,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    convergence_axis.set_title("Convergencia del error relativo", fontweight="bold")
    convergence_axis.set_xlabel("Generación", fontweight="bold")
    convergence_axis.set_ylabel(
        "Mejor NMSE histórico / NMSE inicial", fontweight="bold"
    )
    convergence_axis.legend(frameon=False)
    convergence_axis.grid(alpha=0.22)
    convergence_axis.spines[["top", "right"]].set_visible(False)

    values = [final_errors[resolution] for resolution in resolutions]
    medians = [statistics.median(group) for group in values]
    boxes = boxplot_axis.boxplot(
        values,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#d81b36", "linewidth": 2.4},
        whiskerprops={"color": "#62747d", "linewidth": 1.3},
        capprops={"color": "#62747d", "linewidth": 1.3},
        flierprops={
            "marker": "o",
            "markersize": 4.5,
            "markerfacecolor": "#62747d",
            "markeredgecolor": "#62747d",
            "alpha": 0.75,
        },
    )
    for index, box in enumerate(boxes["boxes"]):
        color = COLORS[index % len(COLORS)]
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.25)
        box.set_linewidth(1.6)

    boxplot_axis.set_xticks(
        range(1, len(resolutions) + 1),
        [f"{resolution} px" for resolution in resolutions],
    )
    boxplot_axis.set_title("Distribución del NMSE final", fontweight="bold")
    boxplot_axis.set_xlabel("Resolución de trabajo", fontweight="bold")
    boxplot_axis.set_ylabel(r"NMSE final ($\times 10^3$; menor es mejor)", fontweight="bold")
    boxplot_axis.grid(axis="y", alpha=0.22)
    boxplot_axis.spines[["top", "right"]].set_visible(False)

    value_range = max(max(max(group) for group in values) - min(min(group) for group in values), 0.1)
    for position, median in enumerate(medians, start=1):
        boxplot_axis.text(
            position,
            median + value_range * 0.035,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#18384d",
        )

    counts = {resolution: len(final_errors[resolution]) for resolution in resolutions}
    if len(set(counts.values())) == 1:
        figure.suptitle(
            f"Resolución de trabajo · {next(iter(counts.values()))} semillas por alternativa",
            fontsize=15,
            fontweight="bold",
        )
    else:
        figure.suptitle("Resolución de trabajo", fontsize=15, fontweight="bold")

    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = output.suffix.lower().removeprefix(".") or "png"
    figure.savefig(output, dpi=220, format=output_format, bbox_inches="tight")
    if output.suffix.lower() != ".svg":
        figure.savefig(output.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
