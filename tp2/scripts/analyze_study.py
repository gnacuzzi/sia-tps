#!/usr/bin/env python3
"""Aggregate TP2 experiment records and render presentation-ready outputs."""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from sia_tp2.study import (
    read_records,
    select_conditions,
    summarize_records,
    write_summaries,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT.parent / ".context" / "tp2-comparative-study"
DEFAULT_PUBLISHED_OUTPUT = PROJECT_ROOT / "experiments" / "results"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a TP2 comparative-study phase."
    )
    parser.add_argument(
        "phase", choices=("profile", "selection", "crossover", "mutation", "validation", "showcase")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--published-output",
        type=Path,
        default=DEFAULT_PUBLISHED_OUTPUT,
        help="Versioned destination for summaries, decisions, and final figures.",
    )
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)

    if args.phase == "profile":
        profile_path = args.output / "profile" / "selection-pressure.csv"
        _render_selection_profile(profile_path, args.output / "figures" / "selection-pressure.svg")
        _copy_artifact(profile_path, args.published_output / "profile" / profile_path.name)
        _copy_artifact(
            args.output / "figures" / "selection-pressure.svg",
            args.published_output / "figures" / "selection-pressure.svg",
        )
        print(f"Figure: {args.output / 'figures' / 'selection-pressure.svg'}")
        return 0

    records = read_records(args.output / "records" / f"{args.phase}.csv")
    summaries = summarize_records(records)
    summary_path = args.output / "summaries" / f"{args.phase}.csv"
    write_summaries(summaries, summary_path)
    _write_report_fragment(summaries, args.output / "reports" / f"{args.phase}.md")
    decisions = _decisions(args.phase, summaries)
    decision_path = args.output / "decisions" / f"{args.phase}.json"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
    if args.figures:
        _render_figures(records, args.output / "figures" / args.phase)
    _copy_artifact(summary_path, args.published_output / "summaries" / summary_path.name)
    _copy_artifact(
        args.output / "reports" / f"{args.phase}.md",
        args.published_output / "reports" / f"{args.phase}.md",
    )
    _copy_artifact(decision_path, args.published_output / "decisions" / decision_path.name)
    figure_directory = args.output / "figures" / args.phase
    if figure_directory.is_dir():
        for figure in figure_directory.glob("*.svg"):
            _copy_artifact(figure, args.published_output / "figures" / args.phase / figure.name)
    print(f"Summary: {summary_path}")
    print(f"Decisions: {decision_path}")
    return 0


def _decisions(phase: str, summaries: Sequence[Mapping[str, object]]) -> Mapping[str, Sequence[str]]:
    if phase == "selection":
        return {"selection": select_conditions(summaries, phase=phase, count=2)}
    if phase == "crossover":
        winner = select_conditions(summaries, phase=phase, count=1)[0]
        selector, crossover = winner.rsplit("__", 1)
        return {"selection": (selector,), "crossover": (crossover,)}
    if phase == "mutation":
        winner = select_conditions(summaries, phase=phase, count=1)[0]
        mutation = "multigene_local" if winner.startswith("multigene") else "single_global"
        survival = winner.rsplit("__", 1)[-1]
        return {"mutation": (mutation,), "survival": (survival,)}
    return {}


def _write_report_fragment(summaries: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# Resultados de la fase",
        "",
        "| Objetivo | Condición | Corridas | Mediana NMSE | Mediana AUC normalizada | Diversidad final | Éxitos 90% |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        rows.append(
            "| {target} | {condition} | {runs} | {error:.6f} | {auc:.6f} | {diversity:.6f} | {successes} |".format(
                target=item["target"],
                condition=item["condition"],
                runs=item["runs"],
                error=float(item["median_best_error"]),
                auc=float(item["median_normalized_auc"]),
                diversity=float(item["median_final_diversity"]),
                successes=item["successes_90_percent_reduction"],
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _render_figures(records: Sequence[Mapping[str, str]], directory: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install the 'plot' dependency group to render figures") from error

    directory.mkdir(parents=True, exist_ok=True)
    curves: Dict[Tuple[str, str], List[Tuple[int, float, float]]] = defaultdict(list)
    for record in records:
        run_directory = Path(record["run_directory"])
        with (run_directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        initial = float(rows[0]["best_error"])
        best = float("inf")
        for row in rows:
            best = min(best, float(row["best_error"]))
            curves[(record["target"], record["condition"])].append(
                (int(row["generation"]), best / initial, float(row["diversity"]))
            )

    grouped: Dict[Tuple[str, str], Dict[int, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for key, values in curves.items():
        for generation, error, diversity in values:
            grouped[key][generation].append((error, diversity))

    for target in sorted({key[0] for key in grouped}):
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for (curve_target, condition), points in sorted(grouped.items()):
            if curve_target != target:
                continue
            generations = sorted(points)
            errors = [sum(value[0] for value in points[generation]) / len(points[generation]) for generation in generations]
            diversities = [sum(value[1] for value in points[generation]) / len(points[generation]) for generation in generations]
            axes[0].plot(generations, errors, label=condition)
            axes[1].plot(generations, diversities, label=condition)
        axes[0].set(title=f"{target}: error histórico normalizado", xlabel="Generación", ylabel="NMSE / NMSE inicial")
        axes[1].set(title=f"{target}: diversidad", xlabel="Generación", ylabel="Diversidad")
        for axis in axes:
            axis.legend(fontsize="small")
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / f"{target}-curves.svg", format="svg")
        plt.close(figure)


def _render_selection_profile(profile_path: Path, figure_path: Path) -> None:
    """Plot empirical selector pressure using the engine's own implementations."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install the 'plot' dependency group to render figures") from error

    with profile_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = [row["condition"] for row in rows]
    top_quartile = [float(row["top_quartile_share"]) for row in rows]
    best = [float(row["best_individual_share"]) for row in rows]
    entropy = [float(row["selection_entropy"]) for row in rows]
    positions = list(range(len(labels)))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(positions, top_quartile, label="cuartil superior")
    axes[0].bar(positions, best, label="mejor individuo")
    axes[0].set(
        title="Presión selectiva empírica",
        xlabel="Selector",
        ylabel="Proporción de padres elegidos",
        xticks=positions,
        xticklabels=labels,
    )
    axes[1].bar(positions, entropy)
    axes[1].set(
        title="Diversidad de la muestra de padres",
        xlabel="Selector",
        ylabel="Entropía (bits)",
        xticks=positions,
        xticklabels=labels,
    )
    for axis in axes:
        axis.tick_params(axis="x", rotation=35, labelsize="small")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize="small")
    figure.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, format="svg")
    plt.close(figure)


def _copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


if __name__ == "__main__":
    raise SystemExit(main())
