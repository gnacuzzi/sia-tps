import json
from pathlib import Path

from PIL import Image

from sia_tp2.config import load_config
from sia_tp2.workflow import evolve_image, render_random_population

from .test_config import _create_image, _valid_payload, _write_config


def test_random_workflow_writes_complete_reproducible_artifacts(
    tmp_path: Path,
) -> None:
    image_path = _create_image(tmp_path)
    payload = _valid_payload(image_path, tmp_path / "runs")
    config = load_config(_write_config(tmp_path, payload), project_root=tmp_path)

    first = render_random_population(config)
    second = render_random_population(config)

    required = {
        "config.effective.json",
        "metadata.json",
        "metrics.csv",
        "triangles.json",
        "best.png",
    }
    assert {path.name for path in first.run_directory.iterdir()} == required
    assert first.run_directory != second.run_directory
    for name in required:
        assert (first.run_directory / name).read_bytes() == (
            second.run_directory / name
        ).read_bytes()

    triangles = json.loads((first.run_directory / "triangles.json").read_text())
    assert triangles["individual"]["triangle_count"] == 3
    assert len(triangles["individual"]["triangles"]) == 3
    with Image.open(first.run_directory / "best.png") as best:
        assert best.size == (16, 12)


def test_evolution_workflow_preserves_best_and_writes_generation_metrics(
    tmp_path: Path,
) -> None:
    image_path = _create_image(tmp_path)
    payload = _valid_payload(image_path, tmp_path / "runs")
    payload["genetic"]["mutation"] = {
        "method": "multigene_uniform",
        "probability": 0.2,
        "allele_change": {
            "mode": "local_delta",
            "position_delta": 0.1,
            "color_delta": 20,
            "alpha_delta": 20,
        },
    }
    payload["termination"]["max_generations"] = 3
    payload["output"]["checkpoint_every"] = 2
    config = load_config(_write_config(tmp_path, payload), project_root=tmp_path)

    result = evolve_image(config)

    assert result.evolution.final_generation == 3
    assert result.evolution.evaluations == 16
    assert result.evolution.metrics[-1].best_error <= result.evolution.metrics[0].best_error
    metrics_lines = (result.run_directory / "metrics.csv").read_text().splitlines()
    assert len(metrics_lines) == 5
    metadata = json.loads((result.run_directory / "metadata.json").read_text())
    assert metadata["mode"] == "evolution"
    assert metadata["stop_reason"] == "max_generations"
    checkpoint = result.run_directory / "checkpoints" / "generation-000002"
    assert checkpoint.with_suffix(".json").is_file()
    assert checkpoint.with_suffix(".png").is_file()
    assert metrics_lines[-1].endswith(",max_generations")
