#!/usr/bin/env python3
"""Generar los JSON del barrido con tasa decreciente de Kohonen."""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "configs" / "kohonen" / "03-rate-decay"
GRID_SIZES = range(2, 9)
LEARNING_RATES = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
SEEDS = range(5)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = set()
    for grid_size in GRID_SIZES:
        for learning_rate in LEARNING_RATES:
            rate_label = f"{round(learning_rate * 100):02d}"
            for seed in SEEDS:
                filename = (
                    f"k{grid_size:02d}-eta-{rate_label}-decay-seed-{seed:02d}.json"
                )
                expected.add(filename)
                document = {
                    "name": filename.removesuffix(".json"),
                    "grid_size": grid_size,
                    "epochs": 150,
                    "learning_rate_start": learning_rate,
                    "learning_rate_end": learning_rate * 0.1,
                    "radius_start": sqrt(2) * (grid_size - 1),
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
