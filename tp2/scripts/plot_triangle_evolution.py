#!/usr/bin/env python3
"""Plot NMSE evolution for each triangle count in a capacity study."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT.parent / ".context" / "tp2-triangle-count-study"
DEFAULT_DESTINATION = PROJECT_ROOT / "output" / "triangle-count-study" / "figures"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot median historical NMSE by generation and triangle count."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)

    record_path = args.output / "records" / "capacity.csv"
    if not record_path.is_file():
        parser.error(f"capacity records not found: {record_path}")

    records = _read_csv(record_path)
    series = _collect_series(records)
    if not series:
        parser.error("capacity records contain no triangle-count conditions")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render figures") from error

    args.destination.mkdir(parents=True, exist_ok=True)
    targets = sorted({target for target, _ in series})
    for target in targets:
        figure, axis = plt.subplots(figsize=(10, 6))
        triangle_counts = sorted(
            count for curve_target, count in series if curve_target == target
        )
        for triangle_count in triangle_counts:
            points = series[(target, triangle_count)]
            generations = sorted(points)
            medians = [statistics.median(points[generation]) for generation in generations]
            q1 = [_quantile(points[generation], 0.25) for generation in generations]
            q3 = [_quantile(points[generation], 0.75) for generation in generations]
            line = axis.plot(
                generations,
                medians,
                linewidth=2,
                label=f"{triangle_count} triángulos",
            )[0]
            axis.fill_between(
                generations,
                q1,
                q3,
                color=line.get_color(),
                alpha=0.10,
                linewidth=0,
            )

        target_label = {"sign": "Señal", "icon": "Ícono", "flag": "Colombia"}.get(
            target, target
        )
        axis.set_title(f"{target_label}: evolución del NMSE según la cantidad de triángulos")
        axis.set_xlabel("Generación")
        axis.set_ylabel("NMSE histórico (menor es mejor)")
        axis.grid(alpha=0.25)
        axis.legend(title="Representación")
        figure.tight_layout()

        destination = args.destination / f"{target}-triangle-evolution.png"
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        print(f"Figure: {destination}")

    return 0


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _collect_series(
    records: Sequence[Dict[str, str]],
) -> Dict[Tuple[str, int], Dict[int, List[float]]]:
    values: DefaultDict[Tuple[str, int], DefaultDict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        condition = record["condition"]
        if not condition.startswith("triangles_"):
            continue
        triangle_count = int(condition.removeprefix("triangles_"))
        metrics = _read_csv(Path(record["run_directory"]) / "metrics.csv")
        historical_best = float("inf")
        for row in metrics:
            historical_best = min(historical_best, float(row["best_error"]))
            values[(record["target"], triangle_count)][int(row["generation"])].append(
                historical_best
            )
    return {
        key: {generation: errors for generation, errors in points.items()}
        for key, points in values.items()
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


if __name__ == "__main__":
    raise SystemExit(main())
