#!/usr/bin/env python3
"""Generar los JSON del experimento de inicialización de Kohonen."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sia_tp4.kohonen_epoch_experiment import (  # noqa: E402
    EPOCH_PROFILES,
    SELECTED_EPOCHS,
)
from sia_tp4.kohonen_initialization_experiment import INITIALIZATIONS  # noqa: E402


OUTPUT = PROJECT_ROOT / "configs" / "kohonen" / "06-initialization"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = set()
    for profile, grid, rate_start, rate_end, radius_fraction in EPOCH_PROFILES:
        diameter = sqrt(2) * (grid - 1)
        radius_start = 1 + radius_fraction * (diameter - 1)
        for initialization in INITIALIZATIONS:
            for seed in range(5):
                filename = f"{profile}-{initialization}-seed-{seed:02d}.json"
                expected.add(filename)
                document = {
                    "name": filename.removesuffix(".json"),
                    "grid_size": grid,
                    "epochs": SELECTED_EPOCHS,
                    "learning_rate_start": rate_start,
                    "learning_rate_end": rate_end,
                    "radius_start": radius_start,
                    "radius_end": 1.0,
                    "initialization": initialization,
                    "shuffle": True,
                    "seed": seed,
                }
                (OUTPUT / filename).write_text(
                    json.dumps(document, indent=2) + "\n",
                    encoding="utf-8",
                )
    unexpected = {path.name for path in OUTPUT.glob("*.json")} - expected
    if unexpected:
        raise ValueError(f"hay configuraciones inesperadas: {sorted(unexpected)}")
    print(f"Se generaron {len(expected)} configuraciones en {OUTPUT}")


if __name__ == "__main__":
    main()
