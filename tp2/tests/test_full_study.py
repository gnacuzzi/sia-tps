import json
from pathlib import Path

import pytest

import scripts.run_full_study as full_study
from scripts.run_full_study import (
    _apply_base_decisions,
    _set_target_integer,
    _validate_output_identity,
)


def test_representation_decisions_are_applied_to_the_flag() -> None:
    manifest = {"targets": {"flag": {"working_max_side": 32, "triangle_count": 10}}}

    _set_target_integer(manifest, "working_max_side", "side_64", "side_")
    _set_target_integer(manifest, "triangle_count", "triangles_25", "triangles_")

    assert manifest["targets"]["flag"] == {
        "working_max_side": 64,
        "triangle_count": 25,
    }


def test_operator_decisions_are_applied_to_the_final_baseline(tmp_path: Path) -> None:
    decision = tmp_path / "crossover.json"
    decision.write_text(
        json.dumps({"selection": ["ranking"], "crossover": ["uniform"]}),
        encoding="utf-8",
    )
    manifest = {"base": {"selector": "elite", "crossover": "one_point"}}

    _apply_base_decisions(manifest, decision)

    assert manifest["base"] == {"selector": "ranking", "crossover": "uniform"}


def test_resume_rejects_results_from_a_different_manifest(tmp_path: Path) -> None:
    output = tmp_path / "study"
    _validate_output_identity({"study_name": "first"}, output)

    _validate_output_identity({"study_name": "first"}, output)
    with pytest.raises(ValueError, match="different manifest"):
        _validate_output_identity({"study_name": "second"}, output)


def test_main_carries_each_automatic_decision_to_the_final_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "study_name": "test",
                "base": {
                    "selector": "elite",
                    "crossover": "one_point",
                    "mutation": "single_local",
                    "survival": "exclusive",
                },
                "targets": {
                    "flag": {"working_max_side": 32, "triangle_count": 10}
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    phases = []
    decisions = {
        "resolution": {"resolution": ["side_64"]},
        "capacity": {"capacity": ["triangles_25"]},
        "selection": {"selection": ["ranking", "tournament_2"]},
        "crossover": {"selection": ["tournament_2"], "crossover": ["uniform"]},
        "mutation": {"mutation": ["multigene_local_balanced"]},
        "survival": {"survival": ["additive"]},
    }

    def fake_run_and_analyze(phase, manifest, phase_output, published, **kwargs):
        phases.append(phase)
        if phase in decisions:
            decision_path = phase_output / "decisions" / f"{phase}.json"
            decision_path.parent.mkdir(parents=True, exist_ok=True)
            decision_path.write_text(json.dumps(decisions[phase]), encoding="utf-8")

    monkeypatch.setattr(full_study, "_run_and_analyze", fake_run_and_analyze)
    monkeypatch.setattr(full_study, "_call", lambda *args: None)

    assert full_study.main(("--manifest", str(manifest_path), "--output", str(output))) == 0

    assert phases == [
        "profile",
        "resolution",
        "capacity",
        "selection",
        "crossover",
        "mutation",
        "survival",
        "validation",
        "showcase",
    ]
    final_manifest = json.loads((output / "working-manifest.json").read_text())
    assert final_manifest["targets"]["flag"]["working_max_side"] == 64
    assert final_manifest["targets"]["flag"]["triangle_count"] == 25
    assert final_manifest["base"] == {
        "selector": "tournament_2",
        "crossover": "uniform",
        "mutation": "multigene_local_balanced",
        "survival": "additive",
    }
