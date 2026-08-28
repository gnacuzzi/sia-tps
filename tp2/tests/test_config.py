import json
from pathlib import Path

import pytest
from PIL import Image

from sia_tp2.config import ConfigError, load_config


def _valid_payload(image_path: Path, output_path: Path) -> dict:
    return {
        "config_version": 1,
        "input": {"image": str(image_path), "working_max_side": 16},
        "representation": {
            "triangle_count": 3,
            "canvas_rgb": [255, 255, 255],
            "alpha_range": [1, 255],
            "initialization": "uniform_random",
        },
        "genetic": {
            "population_size": 4,
            "offspring_count": 4,
            "parent_selection": {
                "method": "tournament_deterministic",
                "params": {"tournament_size": 2},
            },
            "crossover": {
                "method": "uniform",
                "probability": 0.9,
                "params": {"swap_probability": 0.5},
            },
            "mutation": {
                "method": "single_gene",
                "probability": 0.1,
                "allele_change": {"mode": "global_resample"},
            },
            "survival": {
                "strategy": "additive",
                "selection": {"method": "elite", "params": {}},
            },
        },
        "fitness": {"metric": "normalized_mse", "epsilon": 1e-12},
        "termination": {
            "max_generations": 10,
            "target_nmse": None,
            "stagnation": None,
            "max_seconds": None,
        },
        "run": {"seed": 0},
        "output": {
            "directory": str(output_path),
            "metrics_every": 1,
            "checkpoint_every": None,
            "render_original_size": False,
        },
    }


def _write_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _create_image(tmp_path: Path) -> Path:
    path = tmp_path / "target.png"
    Image.new("RGB", (8, 6), "white").save(path)
    return path


def test_valid_configuration_is_loaded_strictly(tmp_path: Path) -> None:
    image = _create_image(tmp_path)
    payload = _valid_payload(image, tmp_path / "runs")

    config = load_config(_write_config(tmp_path, payload), project_root=tmp_path)

    assert config.input.image == image.resolve()
    assert config.representation.triangle_count == 3
    assert config.genetic.offspring_count == 4
    assert config.effective_dict() == payload


def test_validation_reports_unknown_keys_and_odd_offspring_together(
    tmp_path: Path,
) -> None:
    image = _create_image(tmp_path)
    payload = _valid_payload(image, tmp_path / "runs")
    payload["unknown"] = True
    payload["genetic"]["offspring_count"] = 3

    with pytest.raises(ConfigError) as captured:
        load_config(_write_config(tmp_path, payload), project_root=tmp_path)

    message = str(captured.value)
    assert "config.unknown is unknown" in message
    assert "offspring_count must be even" in message


def test_method_rejects_parameters_from_another_method(tmp_path: Path) -> None:
    image = _create_image(tmp_path)
    payload = _valid_payload(image, tmp_path / "runs")
    payload["genetic"]["parent_selection"] = {
        "method": "roulette",
        "params": {"threshold": 0.8},
    }

    with pytest.raises(ConfigError, match="params.threshold is unknown"):
        load_config(_write_config(tmp_path, payload), project_root=tmp_path)


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"config_version": 1, "config_version": 1}', encoding="utf-8")

    with pytest.raises(ConfigError, match="duplicate JSON key"):
        load_config(path, project_root=tmp_path)

