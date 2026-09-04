from pathlib import Path

import pytest

from sia_tp2.config import load_config
from sia_tp2.study import (
    build_specs,
    build_showcase_specs,
    load_manifest,
    materialize_specs,
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
        assert config.genetic.population_size == 100
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
    assert by_target["sign"].payload["termination"]["max_generations"] == 2000
    assert by_target["icon"].payload["output"]["checkpoint_every"] == 500
    for path in materialize_specs(specs, tmp_path / "configs"):
        load_config(path, project_root=PROJECT_ROOT)


def test_selector_profile_exposes_the_actual_elite_cycle_semantics() -> None:
    profile = {item["condition"]: item for item in selection_profile(load_manifest(MANIFEST))}

    assert profile["elite"]["top_quartile_share"] == pytest.approx(0.25)
    assert profile["tournament_5"]["top_quartile_share"] > profile["tournament_2"]["top_quartile_share"]


def test_summary_and_selection_are_ranked_by_quality_then_auc() -> None:
    records = (
        _record("selection", "flag", "alpha", 0.02, 0.3, 0.01),
        _record("selection", "flag", "alpha", 0.03, 0.4, 0.02),
        _record("selection", "flag", "beta", 0.04, 0.1, 0.5),
        _record("selection", "flag", "beta", 0.05, 0.1, 0.6),
    )
    summaries = summarize_records(records)

    assert select_conditions(summaries, phase="selection", count=1) == ("alpha",)


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
