#!/usr/bin/env python3
"""Render a boxplot of final NMSE for each tested working resolution."""

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = (
    PROJECT_ROOT / "output" / "studies" / "resolution-v2" / "records" / "resolution.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "studies"
    / "resolution-v2"
    / "figures"
    / "resolution"
    / "resolution-nmse-boxplot.png"
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare final NMSE distributions across working resolutions."
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        help="Resoluciones que se incluirán; por defecto se muestran todas.",
    )
    args = parser.parse_args(argv)

    groups = _read_resolution_errors(args.records)
    if args.resolutions:
        requested = set(args.resolutions)
        groups = tuple(item for item in groups if item[0] in requested)
        missing = sorted(requested - {resolution for resolution, _ in groups})
        if missing:
            raise ValueError(f"faltan resultados para las resoluciones: {missing}")
    _render_boxplot(groups, args.output)
    _print_summary(groups)
    print(f"Gráfico: {args.output}")
    return 0


def _read_resolution_errors(path: Path) -> Tuple[Tuple[int, Tuple[float, ...]], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"no existe el archivo de resultados: {path}")

    grouped: DefaultDict[int, List[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("phase") != "resolution":
                continue
            condition = row.get("condition", "")
            if not condition.startswith("side_"):
                raise ValueError(f"condición de resolución inválida: {condition!r}")
            resolution = int(condition.removeprefix("side_"))
            grouped[resolution].append(float(row["best_error"]))

    if not grouped:
        raise ValueError(f"no hay corridas de resolución en {path}")

    return tuple(
        (resolution, tuple(errors))
        for resolution, errors in sorted(grouped.items())
    )


def _render_boxplot(
    groups: Sequence[Tuple[int, Sequence[float]]], output: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "falta matplotlib; instalá el grupo opcional 'plot'"
        ) from error

    resolutions = [resolution for resolution, _ in groups]
    scaled_errors = [
        [error * 1_000 for error in errors]
        for _, errors in groups
    ]
    medians = [statistics.median(errors) for errors in scaled_errors]

    colors = ("#667983", "#1874b4", "#e99513", "#2ca25f", "#9467bd")
    figure, axis = plt.subplots(figsize=(12, 7.2))
    boxes = axis.boxplot(
        scaled_errors,
        patch_artist=True,
        widths=0.48,
        showfliers=False,
        medianprops={"color": "#18384d", "linewidth": 2.2},
        whiskerprops={"color": "#82939b", "linewidth": 1.4},
        capprops={"color": "#82939b", "linewidth": 1.4},
    )
    for index, box in enumerate(boxes["boxes"]):
        color = colors[index % len(colors)]
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.23)
        box.set_linewidth(1.5)

    # Every point is one seed. A fixed horizontal spread avoids hiding values
    # without suggesting that the x coordinate has an additional meaning.
    for index, errors in enumerate(scaled_errors, start=1):
        color = colors[(index - 1) % len(colors)]
        if len(errors) == 1:
            offsets = [0.0]
        else:
            step = 0.28 / (len(errors) - 1)
            offsets = [-0.14 + position * step for position in range(len(errors))]
            random.Random(resolutions[index - 1]).shuffle(offsets)
        axis.scatter(
            [index + offset for offset in offsets],
            sorted(errors),
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    vertical_range = max(
        max(max(values) for values in scaled_errors)
        - min(min(values) for values in scaled_errors),
        0.1,
    )
    for position, (median, errors) in enumerate(
        zip(medians, scaled_errors), start=1
    ):
        third_quartile = statistics.quantiles(
            errors, n=4, method="inclusive"
        )[2]
        axis.text(
            position,
            third_quartile + vertical_range * 0.025,
            f"mediana {median:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#18384d",
        )

    axis.set_title(
        "Calidad final según la resolución de trabajo",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    axis.set_xlabel("Resolución de trabajo (px)", fontsize=13, fontweight="bold")
    axis.set_ylabel(
        r"NMSE final $\times$ 1000 (menor es mejor)",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_xticks(
        range(1, len(resolutions) + 1),
        [str(resolution) for resolution in resolutions],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_ylim(bottom=0)
    axis.tick_params(labelsize=11)
    axis.text(
        0.985,
        0.975,
        "Puntos: semillas individuales\nCaja: 50 % central\nLínea dentro de la caja: mediana",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#536b77",
    )
    figure.text(
        0.5,
        0.025,
        "Bandera de Colombia · 10 semillas por condición · 1000 generaciones",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#607782",
    )
    figure.tight_layout(rect=(0.025, 0.065, 0.985, 0.98))

    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = output.suffix.lower().removeprefix(".") or "png"
    figure.savefig(output, dpi=220, format=output_format, bbox_inches="tight")
    plt.close(figure)


def _print_summary(groups: Sequence[Tuple[int, Sequence[float]]]) -> None:
    print("Resolución | corridas | mediana NMSE | cambio vs. anterior")
    previous_median: Optional[float] = None
    for resolution, errors in groups:
        median = statistics.median(errors)
        if previous_median is None:
            change = "—"
        else:
            percentage = (median - previous_median) / previous_median * 100
            change = f"{percentage:+.2f}%"
        print(f"{resolution:>10} | {len(errors):>8} | {median:.6f}     | {change}")
        previous_median = median


if __name__ == "__main__":
    raise SystemExit(main())
