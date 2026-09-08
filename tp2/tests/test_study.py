from pathlib import Path

import pytest

from sia_tp2.config import load_config
from sia_tp2.study import (
    build_specs,
    build_showcase_specs,
    load_manifest,
    materialize_specs,
    order_specs,
    select_conditions,
    selection_profile,
    summarize_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "experiments" / "study-manifest.json"


def test_selection_specs_are_strictly_valid_and_cover_all_seeds(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    specs = build_specs(
        manifest, phase="selection", output_root=tmp_path / "study"
    )

    assert len(specs) == 8 * 5
    paths = materialize_specs(specs, tmp_path / "configs")
    assert len(paths) == len(specs)
    for path in paths:
        config = load_config(path, project_root=PROJECT_ROOT)
        assert config.genetic.population_size == 50
        assert config.genetic.survival.strategy == "additive"
        assert config.genetic.survival.selection.method == "elite"


def test_later_stages_require_their_prior_decisions(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    with pytest.raises(ValueError, match="requires two selected parent selectors"):
        build_specs(manifest, phase="crossover", output_root=tmp_path)

    specs = build_specs(
        manifest,
        phase="crossover",
        output_root=tmp_path,
        selected_methods={"selection": ("ranking", "tournament_2")},
    )
    assert len(specs) == 2 * 2 * 2 * 5
    assert {spec.target for spec in specs} == {"flag", "sign"}


def test_resolution_and_capacity_change_only_the_studied_representation_factor(
    tmp_path: Path,
) -> None:
    manifest = load_manifest(MANIFEST)
    resolution = build_specs(manifest, phase="resolution", output_root=tmp_path)
    capacity = build_specs(manifest, phase="capacity", output_root=tmp_path)

    assert len(resolution) == 2 * 5
    assert {spec.payload["input"]["working_max_side"] for spec in resolution} == {32, 64}
    assert {spec.payload["representation"]["triangle_count"] for spec in resolution} == {25}
    assert len(capacity) == 3 * 5
    assert {spec.payload["representation"]["triangle_count"] for spec in capacity} == {
        10,
        25,
        50,
    }
    assert {spec.payload["input"]["working_max_side"] for spec in capacity} == {64}


def test_capacity_can_compare_multiple_manifest_targets(tmp_path: Path) -> None:
    manifest = dict(load_manifest(MANIFEST))
    manifest["capacity_targets"] = ["sign", "icon"]
    manifest["triangle_count_variants"] = [25, 50, 75, 100, 125]

    specs = build_specs(manifest, phase="capacity", output_root=tmp_path)

    assert len(specs) == 2 * 5 * 5
    assert {spec.target for spec in specs} == {"sign", "icon"}
    assert {spec.payload["representation"]["triangle_count"] for spec in specs} == {
        25,
        50,
        75,
        100,
        125,
    }


def test_mutations_have_equal_local_scope_and_one_expected_changed_gene(
    tmp_path: Path,
) -> None:
    specs = build_specs(
        load_manifest(MANIFEST),
        phase="mutation",
        output_root=tmp_path,
        selected_methods={"selection": ("ranking",), "crossover": ("uniform",)},
    )

    assert len(specs) == 2 * 2 * 5
    by_condition = {spec.condition: spec for spec in specs}
    single = by_condition["single_local"].payload["genetic"]["mutation"]
    multigene = by_condition["multigene_local_balanced"].payload["genetic"]["mutation"]
    triangle_count = by_condition["multigene_local_balanced"].payload["representation"][
        "triangle_count"
    ]
    assert single["method"] == "single_gene"
    assert single["probability"] == 1.0
    assert multigene["method"] == "multigene_uniform"
    assert multigene["probability"] == pytest.approx(1.0 / triangle_count)
    assert single["allele_change"] == multigene["allele_change"]
    for path in materialize_specs(specs, tmp_path / "mutation-configs"):
        load_config(path, project_root=PROJECT_ROOT)


def test_survival_phase_changes_only_replacement_strategy(tmp_path: Path) -> None:
    specs = build_specs(
        load_manifest(MANIFEST),
        phase="survival",
        output_root=tmp_path,
        selected_methods={
            "selection": ("ranking",),
            "crossover": ("uniform",),
            "mutation": ("single_local",),
        },
    )

    assert len(specs) == 2 * 2 * 5
    assert {spec.payload["genetic"]["survival"]["strategy"] for spec in specs} == {
        "additive",
        "exclusive",
    }
    common = []
    for spec in specs:
        genetic = dict(spec.payload["genetic"])
        genetic["survival"] = dict(genetic["survival"])
        genetic["survival"].pop("strategy")
        common.append(genetic)
    assert all(genetic == common[0] for genetic in common)
    for path in materialize_specs(specs, tmp_path / "survival-configs"):
        load_config(path, project_root=PROJECT_ROOT)


def test_run_order_is_randomized_but_reproducible(tmp_path: Path) -> None:
    specs = build_specs(load_manifest(MANIFEST), phase="resolution", output_root=tmp_path)

    first = order_specs(specs, seed=123)
    second = order_specs(specs, seed=123)
    third = order_specs(specs, seed=456)
    assert first == second
    assert first != tuple(specs)
    assert first != third
    assert {spec.run_id for spec in first} == {spec.run_id for spec in specs}


def test_showcase_uses_the_median_validation_seed_and_checkpoints(tmp_path: Path) -> None:
    manifest = load_manifest(MANIFEST)
    records = tuple(
        {
            "phase": "validation",
            "target": target,
            "seed": seed,
            "best_error": error,
        }
        for target in ("flag", "sign", "icon")
        for seed, error in ((101, 0.50), (202, 0.10), (303, 0.30), (404, 0.20), (505, 0.40))
    )
    specs = build_showcase_specs(
        manifest,
        output_root=tmp_path,
        selected_methods={
            "selection": ("ranking",),
            "crossover": ("uniform",),
            "mutation": ("multigene_local",),
            "survival": ("additive",),
        },
        validation_records=records,
    )

    assert {spec.seed for spec in specs} == {303}
    by_target = {spec.target: spec for spec in specs}
    assert by_target["flag"].payload["termination"]["max_generations"] == 3000
    assert by_target["sign"].payload["termination"]["max_generations"] == 5000
    assert by_target["icon"].payload["output"]["checkpoint_every"] == 500
    for path in materialize_specs(specs, tmp_path / "configs"):
        load_config(path, project_root=PROJECT_ROOT)


def test_selector_profile_exposes_the_actual_elite_cycle_semantics() -> None:
    profile = {item["condition"]: item for item in selection_profile(load_manifest(MANIFEST))}

    assert profile["elite"]["top_quartile_share"] == pytest.approx(0.25)
    assert profile["tournament_5"]["top_quartile_share"] > profile["tournament_2"]["top_quartile_share"]
    assert profile["boltzmann"]["best_individual_share"] < 0.2


def test_summary_and_selection_are_ranked_by_quality_then_auc() -> None:
    records = (
        _record("selection", "flag", "alpha", 0.02, 0.3, 0.01),
        _record("selection", "flag", "alpha", 0.03, 0.4, 0.02),
        _record("selection", "flag", "beta", 0.04, 0.1, 0.5),
        _record("selection", "flag", "beta", 0.05, 0.1, 0.6),
    )
    summaries = summarize_records(records)

    assert select_conditions(summaries, phase="selection", count=1) == ("alpha",)


def test_condition_selection_aggregates_within_target_ranks_not_raw_nmse() -> None:
    summaries = tuple(
        {
            "phase": "mutation",
            "target": target,
            "condition": condition,
            "median_best_error": error,
            "median_normalized_auc": 0.5,
            "median_final_diversity": 0.5,
        }
        for target, values in {
            "high_scale": {"alpha": 0.0, "beta": 0.9},
            "low_scale_1": {"alpha": 0.02, "beta": 0.01},
            "low_scale_2": {"alpha": 0.02, "beta": 0.01},
        }.items()
        for condition, error in values.items()
    )

    assert select_conditions(summaries, phase="mutation", count=1) == ("beta",)


def _record(
    phase: str,
    target: str,
    condition: str,
    best_error: float,
    normalized_auc: float,
    diversity: float,
) -> dict:
    return {
        "phase": phase,
        "target": target,
        "condition": condition,
        "best_error": best_error,
        "best_fitness": 1.0 - best_error,
        "elapsed_seconds": 1.0,
        "normalized_auc": normalized_auc,
        "final_diversity": diversity,
        "generation_to_90_percent_reduction": "",
    }
