import json
from pathlib import Path

from PIL import Image

from sia_tp2.config import load_config
from sia_tp2.workflow import render_random_population

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

