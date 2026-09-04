"""Reproducible experiment orchestration and aggregation for TP2."""

import csv
import json
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sia_tp2.config import load_config
from sia_tp2.ga.selection import select_population
from sia_tp2.workflow import evolve_image


@dataclass(frozen=True)
class ExperimentSpec:
    """One deterministic configuration in an experiment phase."""

    run_id: str
    phase: str
    target: str
    condition: str
    seed: int
    payload: Mapping[str, object]


def load_manifest(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_specs(
    manifest: Mapping[str, object],
    *,
    phase: str,
    output_root: Path,
    selected_methods: Optional[Mapping[str, Sequence[str]]] = None,
) -> Tuple[ExperimentSpec, ...]:
    """Build the requested phase without changing the genetic-engine contract."""

    seeds = tuple(int(seed) for seed in manifest["seeds"])
    targets = _mapping(manifest["targets"])
    base = dict(_mapping(manifest["base"]))
    base["selection_variants"] = manifest["selection_variants"]
    selected_methods = selected_methods or {}

    if phase == "selection":
        target_names = ("flag",)
        variants = _mapping(manifest["selection_variants"])
        conditions = tuple((name, {"selector": name}) for name in variants)
    elif phase == "crossover":
        selectors = tuple(selected_methods.get("selection", ()))
        if not selectors:
            raise ValueError("crossover phase requires two selected parent selectors")
        target_names = ("flag", "sign")
        conditions = tuple(
            (f"{selector}__{method}", {"selector": selector, "crossover": method})
            for selector in selectors
            for method in ("uniform", "one_point")
        )
    elif phase == "mutation":
        selector = _one_selected(selected_methods, "selection")
        crossover = _one_selected(selected_methods, "crossover")
        target_names = ("flag", "sign")
        conditions = (
            (
                "multigene_local__additive",
                {
                    "selector": selector,
                    "crossover": crossover,
                    "mutation": "multigene_local",
                    "survival": "additive",
                },
            ),
            (
                "single_global__additive",
                {
                    "selector": selector,
                    "crossover": crossover,
                    "mutation": "single_global",
                    "survival": "additive",
                },
            ),
            (
                "single_global__exclusive",
                {
                    "selector": selector,
                    "crossover": crossover,
                    "mutation": "single_global",
                    "survival": "exclusive",
                },
            ),
        )
    elif phase == "validation":
        selector = _one_selected(selected_methods, "selection")
        crossover = _one_selected(selected_methods, "crossover")
        mutation = _one_selected(selected_methods, "mutation")
        survival = _one_selected(selected_methods, "survival")
        target_names = tuple(targets)
        conditions = (
            (
                "winner",
                {
                    "selector": selector,
                    "crossover": crossover,
                    "mutation": mutation,
                    "survival": survival,
                },
            ),
        )
    else:
        raise ValueError(f"unsupported phase: {phase}")

    specs = []
    for target_name in target_names:
        for condition_name, overrides in conditions:
            for seed in seeds:
                run_id = f"{phase}__{target_name}__{condition_name}__seed-{seed}"
                payload = _payload(
                    base=base,
                    target=_mapping(targets[target_name]),
                    overrides=overrides,
                    seed=seed,
                    output_directory=output_root / "raw" / phase / run_id,
                )
                specs.append(
                    ExperimentSpec(
                        run_id=run_id,
                        phase=phase,
                        target=target_name,
                        condition=condition_name,
                        seed=seed,
                        payload=payload,
                    )
                )
    return tuple(specs)


def build_showcase_specs(
    manifest: Mapping[str, object],
    *,
    output_root: Path,
    selected_methods: Mapping[str, Sequence[str]],
    validation_records: Sequence[Mapping[str, object]],
) -> Tuple[ExperimentSpec, ...]:
    """Build one extended, visually representative run for each target.

    The seed is the median validation result for that target, rather than the
    best-looking lucky run.  This keeps a slide-worthy image honest about the
    configuration's typical behavior.
    """

    selector = _one_selected(selected_methods, "selection")
    crossover = _one_selected(selected_methods, "crossover")
    mutation = _one_selected(selected_methods, "mutation")
    survival = _one_selected(selected_methods, "survival")
    targets = _mapping(manifest["targets"])
    base = dict(_mapping(manifest["base"]))
    base["selection_variants"] = manifest["selection_variants"]

    records_by_target: Dict[str, List[Mapping[str, object]]] = {}
    for record in validation_records:
        if str(record.get("phase")) == "validation":
            records_by_target.setdefault(str(record["target"]), []).append(record)

    specs = []
    for target_name, target in targets.items():
        candidates = records_by_target.get(target_name, [])
        if not candidates:
            raise ValueError(f"no validation records available for target {target_name}")
        median_record = sorted(candidates, key=lambda row: float(row["best_error"]))[
            len(candidates) // 2
        ]
        showcase_target = dict(_mapping(target))
        showcase_target["max_generations"] = int(showcase_target["showcase_generations"])
        showcase_target["checkpoint_every"] = 500
        seed = int(median_record["seed"])
        run_id = f"showcase__{target_name}__median-seed-{seed}"
        payload = _payload(
            base=base,
            target=showcase_target,
            overrides={
                "selector": selector,
                "crossover": crossover,
                "mutation": mutation,
                "survival": survival,
            },
            seed=seed,
            output_directory=output_root / "raw" / "showcase" / run_id,
        )
        specs.append(
            ExperimentSpec(
                run_id=run_id,
                phase="showcase",
                target=str(target_name),
                condition="winner_median_seed",
                seed=seed,
                payload=payload,
            )
        )
    return tuple(specs)


def materialize_specs(specs: Sequence[ExperimentSpec], directory: Path) -> Tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in specs:
        path = directory / f"{spec.run_id}.json"
        path.write_text(json.dumps(spec.payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def run_specs(
    specs: Sequence[ExperimentSpec],
    *,
    config_directory: Path,
    project_root: Path,
    resume: bool,
    progress: Optional[Callable[[str], None]] = None,
) -> Tuple[Mapping[str, object], ...]:
    """Run specs sequentially so every condition receives the same CPU budget."""

    # Reject every bad configuration before spending CPU on the first run.
    configs = {
        spec.run_id: load_config(
            config_directory / f"{spec.run_id}.json", project_root=project_root
        )
        for spec in specs
    }
    records = []
    for number, spec in enumerate(specs, start=1):
        run_parent = Path(spec.payload["output"]["directory"])
        existing = run_parent / f"evolution-seed-{spec.seed}" / "metadata.json"
        if resume and existing.is_file():
            record = _record_from_directory(spec, existing.parent)
            _verify_evaluation_budget(spec, record)
            records.append(record)
            if progress:
                progress(f"[{number}/{len(specs)}] reused {spec.run_id}")
            continue
        if progress:
            progress(f"[{number}/{len(specs)}] running {spec.run_id}")
        result = evolve_image(configs[spec.run_id])
        record = _record_from_directory(spec, result.run_directory)
        _verify_evaluation_budget(spec, record)
        records.append(record)
        if progress:
            progress(
                f"[{number}/{len(specs)}] done {spec.run_id} "
                f"(NMSE={float(record['best_error']):.6g}, {float(record['elapsed_seconds']):.1f}s)"
            )
    return tuple(records)


def write_records(records: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "run_id",
        "phase",
        "target",
        "condition",
        "seed",
        "run_directory",
        "best_error",
        "best_fitness",
        "final_generation",
        "best_generation",
        "evaluations",
        "elapsed_seconds",
        "initial_error",
        "normalized_auc",
        "final_diversity",
        "generation_to_90_percent_reduction",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_records(path: Path) -> Tuple[Mapping[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as stream:
        return tuple(csv.DictReader(stream))


def summarize_records(records: Sequence[Mapping[str, object]]) -> Tuple[Mapping[str, object], ...]:
    groups: Dict[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for record in records:
        key = (str(record["phase"]), str(record["target"]), str(record["condition"]))
        groups.setdefault(key, []).append(record)

    summaries = []
    metrics = (
        "best_error",
        "best_fitness",
        "elapsed_seconds",
        "normalized_auc",
        "final_diversity",
    )
    for (phase, target, condition), rows in sorted(groups.items()):
        summary: Dict[str, object] = {
            "phase": phase,
            "target": target,
            "condition": condition,
            "runs": len(rows),
            "successes_90_percent_reduction": sum(
                row["generation_to_90_percent_reduction"] != "" for row in rows
            ),
        }
        for metric in metrics:
            values = sorted(float(row[metric]) for row in rows)
            summary[f"median_{metric}"] = statistics.median(values)
            summary[f"q1_{metric}"] = _quantile(values, 0.25)
            summary[f"q3_{metric}"] = _quantile(values, 0.75)
        summaries.append(summary)
    return tuple(summaries)


def write_summaries(summaries: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summaries:
        raise ValueError("cannot write an empty experiment summary")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def select_conditions(
    summaries: Sequence[Mapping[str, object]], *, phase: str, count: int
) -> Tuple[str, ...]:
    """Rank by quality, then convergence, then diversity; all are medians."""

    candidates = [summary for summary in summaries if summary["phase"] == phase]
    if not candidates:
        raise ValueError(f"no summaries exist for phase {phase}")
    scores: Dict[str, List[Tuple[float, float, float]]] = {}
    for item in candidates:
        scores.setdefault(str(item["condition"]), []).append(
            (
                float(item["median_best_error"]),
                float(item["median_normalized_auc"]),
                -float(item["median_final_diversity"]),
            )
        )
    ranked = sorted(
        (
            (
                condition,
                tuple(sum(score[index] for score in values) / len(values) for index in range(3)),
            )
            for condition, values in scores.items()
        ),
        key=lambda item: item[1],
    )
    return tuple(condition for condition, _ in ranked[:count])


def selection_profile(
    manifest: Mapping[str, object], *, draws: int = 1000
) -> Tuple[Mapping[str, object], ...]:
    """Measure the exact selector implementations over a known fitness ranking."""

    variants = _mapping(manifest["selection_variants"])
    population = tuple(range(20))
    top_quartile = set(population[-5:])
    profiles = []
    for index, (name, config) in enumerate(variants.items()):
        rng = random.Random(10_000 + index)
        selected = select_population(
            population,
            count=draws,
            method=str(config["method"]),
            params=_mapping(config["params"]),
            generation=int(config.get("profile_generation", 500)),
            fitness=float,
            rng=rng,
        )
        frequency = {member: selected.count(member) / draws for member in population}
        profiles.append(
            {
                "condition": name,
                "top_quartile_share": sum(frequency[member] for member in top_quartile),
                "best_individual_share": frequency[population[-1]],
                "selection_entropy": -sum(
                    probability * _log2(probability)
                    for probability in frequency.values()
                    if probability > 0.0
                ),
            }
        )
    return tuple(profiles)


def write_profile(profile: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(profile[0]))
        writer.writeheader()
        writer.writerows(profile)


def _payload(
    *,
    base: Mapping[str, object],
    target: Mapping[str, object],
    overrides: Mapping[str, object],
    seed: int,
    output_directory: Path,
) -> Dict[str, object]:
    selection = _selection(str(overrides.get("selector", base["selector"])), base)
    crossover_name = str(overrides.get("crossover", base["crossover"]))
    mutation_name = str(overrides.get("mutation", base["mutation"]))
    survival_name = str(overrides.get("survival", base["survival"]))
    return {
        "config_version": 1,
        "input": {"image": target["image"], "working_max_side": target["working_max_side"]},
        "representation": {
            "triangle_count": target["triangle_count"],
            "canvas_rgb": [255, 255, 255],
            "alpha_range": [1, 255],
            "initialization": "uniform_random",
        },
        "genetic": {
            "population_size": target["population_size"],
            "offspring_count": target["offspring_count"],
            "parent_selection": selection,
            "crossover": _crossover(crossover_name),
            "mutation": _mutation(mutation_name),
            "survival": {
                "strategy": survival_name,
                "selection": {"method": "elite", "params": {}},
            },
        },
        "fitness": {"metric": "normalized_mse", "epsilon": 1e-12},
        "termination": {
            "max_generations": target["max_generations"],
            "target_nmse": None,
            "stagnation": None,
            "max_seconds": None,
        },
        "run": {"seed": seed},
        "output": {
            "directory": str(output_directory),
            "metrics_every": int(base["metrics_every"]),
            "checkpoint_every": target["checkpoint_every"],
            "render_original_size": True,
        },
    }


def _selection(name: str, base: Mapping[str, object]) -> Dict[str, object]:
    variant = _mapping(_mapping(base["selection_variants"])[name])
    return {"method": variant["method"], "params": variant["params"]}


def _crossover(name: str) -> Dict[str, object]:
    if name == "uniform":
        return {"method": "uniform", "probability": 0.9, "params": {"swap_probability": 0.5}}
    if name == "one_point":
        return {"method": "one_point", "probability": 0.9, "params": {}}
    raise ValueError(f"unsupported crossover: {name}")


def _mutation(name: str) -> Dict[str, object]:
    if name == "multigene_local":
        return {
            "method": "multigene_uniform",
            "probability": 0.1,
            "allele_change": {
                "mode": "local_delta",
                "position_delta": 0.1,
                "color_delta": 25,
                "alpha_delta": 20,
            },
        }
    if name == "single_global":
        return {
            "method": "single_gene",
            "probability": 0.25,
            "allele_change": {"mode": "global_resample"},
        }
    raise ValueError(f"unsupported mutation: {name}")


def _record_from_directory(spec: ExperimentSpec, run_directory: Path) -> Mapping[str, object]:
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    with (run_directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    historical_error = float("inf")
    normalized_errors = []
    initial_error = float(metrics[0]["best_error"])
    reduction_generation = ""
    for row in metrics:
        historical_error = min(historical_error, float(row["best_error"]))
        normalized = historical_error / initial_error
        normalized_errors.append(normalized)
        if reduction_generation == "" and normalized <= 0.1:
            reduction_generation = row["generation"]
    diversity = float(metrics[-1]["diversity"])
    elapsed = float(metrics[-1]["elapsed_seconds"])
    return {
        "run_id": spec.run_id,
        "phase": spec.phase,
        "target": spec.target,
        "condition": spec.condition,
        "seed": spec.seed,
        "run_directory": str(run_directory),
        "best_error": metadata["best_error"],
        "best_fitness": metadata["best_fitness"],
        "final_generation": metadata["final_generation"],
        "best_generation": metadata["best_generation"],
        "evaluations": metadata["evaluations"],
        "elapsed_seconds": elapsed,
        "initial_error": initial_error,
        "normalized_auc": statistics.fmean(normalized_errors),
        "final_diversity": diversity,
        "generation_to_90_percent_reduction": reduction_generation,
    }


def _verify_evaluation_budget(spec: ExperimentSpec, record: Mapping[str, object]) -> None:
    genetic = _mapping(spec.payload["genetic"])
    termination = _mapping(spec.payload["termination"])
    expected = int(genetic["population_size"]) + (
        int(termination["max_generations"]) * int(genetic["offspring_count"])
    )
    observed = int(record["evaluations"])
    if observed != expected:
        raise RuntimeError(
            f"{spec.run_id} used {observed} evaluations; expected {expected}. "
            "The configured budget was not completed."
        )


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError("manifest objects must be dictionaries")
    return value


def _one_selected(selected: Mapping[str, Sequence[str]], key: str) -> str:
    values = tuple(selected.get(key, ()))
    if len(values) != 1:
        raise ValueError(f"phase requires exactly one selected {key}")
    return values[0]


def _quantile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _log2(value: float) -> float:
    import math

    return math.log(value, 2)
