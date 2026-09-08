#!/usr/bin/env python3
"""Plot relative historical NMSE for the focused triangle-count study."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT.parent / ".context" / "tp2-triangle-count-study-5000-focused"
DEFAULT_DESTINATION = (
    PROJECT_ROOT / "output" / "presentation-figures" / "triangle-evolution-relative"
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot historical NMSE relative to each run's initial NMSE."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)

    record_path = args.output / "records" / "capacity.csv"
    if not record_path.is_file():
        parser.error(f"capacity records not found: {record_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render figures") from error

    records = _read_csv(record_path)
    series, seeds_by_target = _collect_relative_series(records)
    args.destination.mkdir(parents=True, exist_ok=True)

    colors = {50: "#176FB1", 100: "#E38608"}
    labels = {"sign": "Señal", "icon": "Ícono", "flag": "Colombia"}
    for target in sorted({key[0] for key in series}):
        figure, axis = plt.subplots(figsize=(12, 6.3))
        for triangle_count in sorted(count for curve_target, count in series if curve_target == target):
            points = series[(target, triangle_count)]
            generations = [generation for generation, _ in points]
            values = [value for _, value in points]
            axis.plot(
                generations,
                values,
                color=colors.get(triangle_count),
                linewidth=2.8,
                label=f"{triangle_count} triángulos",
            )

        target_label = labels.get(target, target)
        axis.set_title(
            f"{target_label}: evolución relativa del error",
            fontsize=19,
            fontweight="bold",
            pad=16,
        )
        axis.set_xlabel("Generación", fontsize=14, fontweight="bold", labelpad=10)
        axis.set_ylabel(
            "NMSE histórico / NMSE inicial (menor es mejor)",
            fontsize=14,
            fontweight="bold",
            labelpad=10,
        )
        axis.set_xlim(0, 5000)
        axis.set_ylim(0, 1.03)
        axis.set_xticks(range(0, 5001, 1000))
        axis.grid(True, color="#CBD2D9", alpha=0.42, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(loc="upper right", frameon=False, fontsize=12)
        axis.tick_params(labelsize=11)

        seed_text = ", ".join(str(seed) for seed in sorted(seeds_by_target[target]))
        figure.text(
            0.5,
            0.015,
            f"Semilla {seed_text} · una corrida por condición · métricas registradas cada 25 generaciones",
            ha="center",
            fontsize=10.5,
            color="#5C6B76",
            fontweight="bold",
        )
        figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.98))

        destination = args.destination / f"{target}-triangle-evolution-relative.png"
        figure.savefig(destination, dpi=200, facecolor="white")
        plt.close(figure)
        print(f"Figure: {destination}")

    return 0


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _collect_relative_series(
    records: Sequence[Dict[str, str]],
) -> Tuple[Dict[Tuple[str, int], List[Tuple[int, float]]], Dict[str, set]]:
    series: Dict[Tuple[str, int], List[Tuple[int, float]]] = {}
    seeds_by_target = defaultdict(set)
    for record in records:
        condition = record["condition"]
        if not condition.startswith("triangles_"):
            continue
        target = record["target"]
        triangle_count = int(condition.removeprefix("triangles_"))
        metrics = _read_csv(Path(record["run_directory"]) / "metrics.csv")
        initial_error = float(metrics[0]["best_error"])
        historical_best = float("inf")
        points = []
        for row in metrics:
            historical_best = min(historical_best, float(row["best_error"]))
            points.append((int(row["generation"]), historical_best / initial_error))
        series[(target, triangle_count)] = points
        seeds_by_target[target].add(int(record["seed"]))
    return series, seeds_by_target


if __name__ == "__main__":
    raise SystemExit(main())
