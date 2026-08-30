"""Write stable JSON, CSV, and image artifacts for one run."""

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence, Union

from PIL import Image

from sia_tp2.domain.model import Individual


def create_run_directory(parent: Path, base_name: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / base_name
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = parent / f"{base_name}-{suffix}"


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_metrics(
    path: Path,
    rows: Union[Mapping[str, object], Sequence[Mapping[str, object]]],
) -> None:
    normalized_rows = [rows] if isinstance(rows, Mapping) else list(rows)
    if not normalized_rows:
        raise ValueError("at least one metrics row is required")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(normalized_rows[0]))
        writer.writeheader()
        writer.writerows(normalized_rows)


def write_run_artifacts(
    *,
    run_directory: Path,
    effective_config: Mapping[str, object],
    metadata: Mapping[str, object],
    metrics: Union[Mapping[str, object], Sequence[Mapping[str, object]]],
    best: Individual,
    best_image: Image.Image,
) -> None:
    write_json(run_directory / "config.effective.json", effective_config)
    write_json(run_directory / "metadata.json", metadata)
    write_metrics(run_directory / "metrics.csv", metrics)
    write_json(
        run_directory / "triangles.json",
        {"schema_version": 1, "individual": best.to_dict()},
    )
    best_image.save(run_directory / "best.png", format="PNG")


def write_checkpoint(
    *,
    run_directory: Path,
    generation: int,
    best: Individual,
    best_image: Image.Image,
) -> None:
    checkpoint_directory = run_directory / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    stem = f"generation-{generation:06d}"
    write_json(
        checkpoint_directory / f"{stem}.json",
        {"schema_version": 1, "generation": generation, "individual": best.to_dict()},
    )
    best_image.save(checkpoint_directory / f"{stem}.png", format="PNG")
