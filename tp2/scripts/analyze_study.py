#!/usr/bin/env python3
"""Aggregate TP2 experiment records and render presentation-ready outputs."""

import argparse
import csv
import json
import shutil
import statistics
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
DEFAULT_OUTPUT = PROJECT_ROOT.parent / ".context" / "tp2-colombia-final-study"
DEFAULT_PUBLISHED_OUTPUT = PROJECT_ROOT / "experiments" / "results"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a TP2 comparative-study phase."
    )
    parser.add_argument(
        "phase",
        choices=(
            "profile",
            "resolution",
            "capacity",
            "selection",
            "crossover",
            "mutation",
            "survival",
            "validation",
            "showcase",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--published-output",
        type=Path,
        default=DEFAULT_PUBLISHED_OUTPUT,
        help="Versioned destination for summaries, decisions, and final figures.",
    )
    parser.add_argument("--figures", action="store_true")
    parser.add_argument(
        "--capacity-counts",
        type=int,
        nargs="+",
        help="Triangle counts to include in the capacity boxplot only.",
    )
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
        if args.phase == "capacity":
            _render_capacity_boxplot(
                records,
                args.output
                / "figures"
                / args.phase
                / "capacity-nmse-boxplot",
                included_counts=args.capacity_counts,
            )
        elif args.phase == "selection":
            _render_selection_barplot(
                records,
                args.output
                / "figures"
                / args.phase
                / "selection-nmse-bars",
            )
        elif args.phase == "crossover":
            _render_crossover_evolution(
                records,
                args.output / "figures" / args.phase,
            )
    _copy_artifact(summary_path, args.published_output / "summaries" / summary_path.name)
    _copy_artifact(
        args.output / "reports" / f"{args.phase}.md",
        args.published_output / "reports" / f"{args.phase}.md",
    )
    _copy_artifact(decision_path, args.published_output / "decisions" / decision_path.name)
    figure_directory = args.output / "figures" / args.phase
    if figure_directory.is_dir():
        for extension in ("*.svg", "*.png"):
            for figure in figure_directory.glob(extension):
                _copy_artifact(
                    figure,
                    args.published_output / "figures" / args.phase / figure.name,
                )
    print(f"Summary: {summary_path}")
    print(f"Decisions: {decision_path}")
    return 0


def _decisions(phase: str, summaries: Sequence[Mapping[str, object]]) -> Mapping[str, Sequence[str]]:
    if phase in ("resolution", "capacity", "mutation", "survival"):
        return {phase: select_conditions(summaries, phase=phase, count=1)}
    if phase == "selection":
        return {"selection": select_conditions(summaries, phase=phase, count=2)}
    if phase == "crossover":
        winner = select_conditions(summaries, phase=phase, count=1)[0]
        selector, crossover = winner.rsplit("__", 1)
        return {"selection": (selector,), "crossover": (crossover,)}
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
        import matplotlib

        matplotlib.use("Agg")
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


def _render_capacity_boxplot(
    records: Sequence[Mapping[str, str]],
    output_stem: Path,
    *,
    included_counts: Optional[Sequence[int]] = None,
) -> None:
    """Show final-error distributions for every tested triangle count."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install the 'plot' dependency group to render figures") from error

    grouped: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for record in records:
        condition = record["condition"]
        if not condition.startswith("triangles_"):
            raise ValueError(f"invalid capacity condition: {condition!r}")
        triangle_count = int(condition.removeprefix("triangles_"))
        grouped[triangle_count].append(
            (int(record["seed"]), float(record["best_error"]) * 1_000)
        )

    if included_counts:
        requested = set(included_counts)
        missing = sorted(requested - set(grouped))
        if missing:
            raise ValueError(f"missing capacity results for triangle counts: {missing}")
        grouped = {
            count: grouped[count]
            for count in included_counts
        }

    counts = sorted(grouped)
    values = [
        [error for _, error in sorted(grouped[count])]
        for count in counts
    ]
    medians = [statistics.median(errors) for errors in values]
    colors = ("#667983", "#1874b4", "#e99513", "#2ca25f", "#9467bd", "#c44e52")

    figure, axis = plt.subplots(figsize=(12, 7.2))
    boxes = axis.boxplot(
        values,
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

    for index, errors in enumerate(values, start=1):
        color = colors[(index - 1) % len(colors)]
        if len(errors) == 1:
            offsets = [0.0]
        else:
            step = 0.28 / (len(errors) - 1)
            offsets = [-0.14 + position * step for position in range(len(errors))]
        axis.scatter(
            [index + offset for offset in offsets],
            errors,
            s=48,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )

    value_range = max(
        max(max(errors) for errors in values)
        - min(min(errors) for errors in values),
        0.1,
    )
    for position, (median, errors) in enumerate(zip(medians, values), start=1):
        third_quartile = statistics.quantiles(
            errors, n=4, method="inclusive"
        )[2]
        axis.text(
            position,
            third_quartile + value_range * 0.025,
            f"mediana {median:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#18384d",
        )

    axis.set_title(
        "Calidad final según la cantidad de triángulos",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    axis.set_xlabel("Cantidad de triángulos", fontsize=13, fontweight="bold")
    axis.set_ylabel(
        r"NMSE final $\times$ 1000 (menor es mejor)",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_xticks(range(1, len(counts) + 1), [str(count) for count in counts])
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
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

    runs_per_condition = min(len(errors) for errors in values)
    generations = max(int(record["final_generation"]) for record in records)
    effective_config_path = Path(records[0]["run_directory"]) / "config.effective.json"
    effective_config = json.loads(effective_config_path.read_text(encoding="utf-8"))
    working_max_side = int(effective_config["input"]["working_max_side"])
    figure.text(
        0.5,
        0.025,
        (
            "Bandera de Colombia · "
            f"{runs_per_condition} semillas por condición · "
            f"{generations} generaciones · {working_max_side} px"
        ),
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#607782",
    )
    figure.tight_layout(rect=(0.025, 0.065, 0.985, 0.98))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(figure)


def _render_selection_barplot(
    records: Sequence[Mapping[str, str]], output_stem: Path
) -> None:
    """Compare selector quality through median final NMSE and its IQR."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install the 'plot' dependency group to render figures") from error

    grouped: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(float(record["best_error"]) * 1_000)

    labels = {
        "tournament_5": "Tournament 5",
        "ranking": "Ranking",
        "elite": "Elite",
        "boltzmann": "Boltzmann",
        "tournament_2": "Tournament 2",
        "probabilistic_0_6": "Torneo probabilístico (p=0,6)",
        "universal": "Universal",
        "roulette": "Ruleta",
    }
    statistics_by_condition = []
    for condition, errors in grouped.items():
        quartiles = statistics.quantiles(errors, n=4, method="inclusive")
        statistics_by_condition.append(
            (
                condition,
                statistics.median(errors),
                quartiles[0],
                quartiles[2],
            )
        )
    statistics_by_condition.sort(key=lambda item: item[1])

    conditions = [item[0] for item in statistics_by_condition]
    medians = [item[1] for item in statistics_by_condition]
    lower_errors = [item[1] - item[2] for item in statistics_by_condition]
    upper_errors = [item[3] - item[1] for item in statistics_by_condition]
    positions = list(range(len(conditions)))
    colors = ["#98a6ad"] * len(conditions)
    if colors:
        colors[0] = "#287fb8"
    if len(colors) > 1:
        colors[1] = "#3d9885"

    figure, axis = plt.subplots(figsize=(13, 7.2))
    bars = axis.bar(
        positions,
        medians,
        color=colors,
        width=0.68,
        yerr=[lower_errors, upper_errors],
        capsize=5,
        error_kw={"ecolor": "#405b69", "elinewidth": 1.5, "capthick": 1.5},
    )
    vertical_range = max(max(medians) - min(medians), 0.25)
    for bar, median, upper_error in zip(bars, medians, upper_errors):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            median + upper_error + vertical_range * 0.035,
            f"{median:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#18384d",
        )

    axis.set_title(
        "NMSE final mediano según el método de selección",
        fontsize=20,
        fontweight="bold",
        pad=20,
    )
    axis.set_xlabel("Método de selección", fontsize=13, fontweight="bold")
    axis.set_ylabel(
        r"NMSE final mediano $\times$ 1000 (menor es mejor)",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_xticks(
        positions,
        [labels.get(condition, condition) for condition in conditions],
        rotation=24,
        ha="right",
    )
    top_quartile = max(item[3] for item in statistics_by_condition)
    axis.set_ylim(0, top_quartile + max(vertical_range * 0.38, 0.35))
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=11)
    axis.text(
        0.015,
        0.975,
        "Barra: mediana · Raya: 50 % central de las semillas (IQR)",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="#536b77",
    )

    runs_per_method = min(len(errors) for errors in grouped.values())
    generations = max(int(record["final_generation"]) for record in records)
    effective_config_path = Path(records[0]["run_directory"]) / "config.effective.json"
    effective_config = json.loads(effective_config_path.read_text(encoding="utf-8"))
    side = int(effective_config["input"]["working_max_side"])
    triangle_count = int(effective_config["representation"]["triangle_count"])
    figure.text(
        0.5,
        0.025,
        (
            "Bandera de Colombia · "
            f"{runs_per_method} semillas por método · "
            f"{generations} generaciones · {side} px · "
            f"{triangle_count} triángulos"
        ),
        ha="center",
        fontsize=10.5,
        fontweight="bold",
        color="#607782",
    )
    figure.tight_layout(rect=(0.025, 0.075, 0.985, 0.98))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
    plt.close(figure)


def _render_crossover_evolution(
    records: Sequence[Mapping[str, str]], directory: Path
) -> None:
    """Plot median historical error and IQR for selector/crossover pairs."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("install the 'plot' dependency group to render figures") from error

    curves: Dict[Tuple[str, str], Dict[int, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    run_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for record in records:
        target = record["target"]
        condition = record["condition"]
        run_counts[(target, condition)] += 1
        metrics_path = Path(record["run_directory"]) / "metrics.csv"
        with metrics_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        initial_error = float(rows[0]["best_error"])
        historical_best = float("inf")
        for row in rows:
            historical_best = min(historical_best, float(row["best_error"]))
            curves[(target, condition)][int(row["generation"])].append(
                historical_best / initial_error
            )

    selector_labels = {
        "tournament_5": "Tournament 5",
        "tournament_2": "Tournament 2",
        "ranking": "Ranking",
    }
    crossover_labels = {
        "uniform": "uniforme",
        "one_point": "un punto",
    }
    selector_colors = {
        "tournament_5": "#1874b4",
        "tournament_2": "#3d9885",
        "ranking": "#3d9885",
    }
    fallback_colors = ("#1874b4", "#3d9885", "#e99513", "#9467bd")
    line_styles = {"uniform": "-", "one_point": "--"}
    target_labels = {"flag": "Bandera de Colombia", "sign": "Señal"}

    directory.mkdir(parents=True, exist_ok=True)
    targets = sorted({target for target, _ in curves})
    for target in targets:
        figure, axis = plt.subplots(figsize=(12, 7.2))
        target_conditions = sorted(
            condition
            for curve_target, condition in curves
            if curve_target == target
        )
        fallback_by_selector: Dict[str, str] = {}
        for condition in target_conditions:
            selector, crossover = condition.rsplit("__", 1)
            if selector not in fallback_by_selector:
                fallback_by_selector[selector] = fallback_colors[
                    len(fallback_by_selector) % len(fallback_colors)
                ]
            color = selector_colors.get(selector, fallback_by_selector[selector])
            points = curves[(target, condition)]
            generations = sorted(points)
            medians = []
            first_quartiles = []
            third_quartiles = []
            for generation in generations:
                values = points[generation]
                quartiles = statistics.quantiles(values, n=4, method="inclusive")
                medians.append(statistics.median(values))
                first_quartiles.append(quartiles[0])
                third_quartiles.append(quartiles[2])

            axis.fill_between(
                generations,
                first_quartiles,
                third_quartiles,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            axis.plot(
                generations,
                medians,
                color=color,
                linestyle=line_styles.get(crossover, "-"),
                linewidth=2.4,
                label=(
                    f"{selector_labels.get(selector, selector)} + "
                    f"{crossover_labels.get(crossover, crossover)}"
                ),
            )

        axis.set_title(
            "Evolución del error para los métodos de cruza",
            fontsize=20,
            fontweight="bold",
            pad=20,
        )
        axis.set_xlabel("Generación", fontsize=13, fontweight="bold")
        axis.set_ylabel(
            "NMSE histórico / NMSE inicial (menor es mejor)",
            fontsize=13,
            fontweight="bold",
        )
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=11)
        axis.legend(frameon=False, loc="upper right", fontsize=10.5)
        axis.text(
            0.02,
            0.08,
            "Línea continua: cruza uniforme\nLínea punteada: cruza de un punto",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#536b77",
        )

        runs = min(
            count
            for (run_target, _), count in run_counts.items()
            if run_target == target
        )
        figure.text(
            0.5,
            0.025,
            (
                f"{target_labels.get(target, target)} · "
                f"curva: mediana de {runs} semillas · "
                "banda: 50 % central (IQR)"
            ),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color="#607782",
        )
        figure.tight_layout(rect=(0.025, 0.075, 0.985, 0.98))
        output_stem = directory / f"crossover-error-evolution-{target}"
        figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
        figure.savefig(output_stem.with_suffix(".svg"), format="svg", bbox_inches="tight")
        plt.close(figure)


def _render_selection_profile(profile_path: Path, figure_path: Path) -> None:
    """Plot empirical selector pressure using the engine's own implementations."""

    try:
        import matplotlib

        matplotlib.use("Agg")
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
