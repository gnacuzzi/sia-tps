"""Pruebas de la primera etapa experimental de Oja."""

import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp4.oja_rate_experiment import (
    OjaRunMetrics,
    aggregate_runs,
    load_design,
    run_count,
)


def test_stage_one_design_covers_rates_schedules_and_ten_seeds():
    path = Path(__file__).parents[1] / "configs" / "oja" / "01-rate.json"

    design = load_design(path)

    assert design.learning_rates == (
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
    )
    assert design.schedules == ("constant", "linear-10pct")
    assert design.seeds == tuple(range(10))
    assert design.epochs == 250
    assert run_count(design) == 160


def test_design_requires_ten_unique_seeds(tmp_path):
    raw = {
        "name": "invalid",
        "epochs": 10,
        "learning_rates": [0.01],
        "schedules": ["constant"],
        "seeds": list(range(9)),
        "shuffle": True,
    }
    path = tmp_path / "design.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="10 semillas"):
        load_design(path)


def test_aggregation_uses_sample_deviation_and_counts_divergence():
    metrics = [
        OjaRunMetrics("a", 0.1, "constant", 0, True, 0.8, 0.9, 1.0, 0.0, ""),
        OjaRunMetrics("b", 0.1, "constant", 1, True, 1.0, 1.0, 1.2, 0.2, ""),
        OjaRunMetrics(
            "c", 0.1, "constant", 2, False, np.nan, np.nan, np.nan, np.nan, "divergió"
        ),
    ]

    rows = aggregate_runs(metrics)

    assert len(rows) == 1
    assert rows[0]["successful_runs"] == 2
    assert rows[0]["divergent_runs"] == 1
    assert rows[0]["direction_similarity_mean"] == pytest.approx(0.9)
    assert rows[0]["direction_similarity_std"] == pytest.approx(np.sqrt(0.02))
