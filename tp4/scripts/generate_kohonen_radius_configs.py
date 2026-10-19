#!/usr/bin/env python3
"""Generar los JSON del barrido de radio para Kohonen."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "configs" / "kohonen" / "04-radius"
RADIUS_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = range(5)
RATE_PROFILES = {
    4: ((0.05, 0.05), (0.1, 0.1), (0.3, 0.03), (0.7, 0.07)),
    5: ((0.1, 0.1), (0.2, 0.2), (0.7, 0.07), (0.9, 0.09)),
    6: ((0.1, 0.1), (0.2, 0.2), (0.7, 0.07), (0.9, 0.09)),
}


def _rate_label(start: float, end: float) -> str:
    schedule = "constant" if start == end else "decay"
    return f"{schedule}-eta-{round(start * 100):02d}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = set()
    for grid_size, rate_profiles in RATE_PROFILES.items():
        diameter = sqrt(2) * (grid_size - 1)
        for learning_rate_start, learning_rate_end in rate_profiles:
            rate_label = _rate_label(learning_rate_start, learning_rate_end)
            for radius_fraction in RADIUS_FRACTIONS:
                radius_label = round(radius_fraction * 100)
                radius_start = 1 + radius_fraction * (diameter - 1)
                for seed in SEEDS:
                    filename = (
                        f"k{grid_size:02d}-{rate_label}-radius-{radius_label:03d}"
                        f"-seed-{seed:02d}.json"
                    )
                    expected.add(filename)
                    document = {
                        "name": filename.removesuffix(".json"),
                        "grid_size": grid_size,
                        "epochs": 150,
                        "learning_rate_start": learning_rate_start,
                        "learning_rate_end": learning_rate_end,
                        "radius_start": radius_start,
                        "radius_end": 1.0,
                        "initialization": "sample",
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
