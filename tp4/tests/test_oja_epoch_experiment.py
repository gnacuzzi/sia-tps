"""Pruebas de la segunda etapa experimental de Oja."""

from pathlib import Path

import numpy as np
import pytest

from sia_tp4.oja_epoch_experiment import (
    OjaEpochMetrics,
    aggregate_runs,
    load_design,
    run_count,
)


def test_stage_two_design_covers_finalists_epochs_and_ten_seeds():
    path = Path(__file__).parents[1] / "configs" / "oja" / "02-epochs.json"

    design = load_design(path)

    assert design.epochs == (5, 10, 25, 50, 100, 150, 250, 500)
    assert {profile.name for profile in design.profiles} == {
        "decay-eta-0001",
        "decay-eta-0005",
        "constant-eta-0001",
    }
    assert design.seeds == tuple(range(10))
    assert run_count(design) == 240


def test_epoch_aggregation_uses_sample_deviation():
    metrics = [
        OjaEpochMetrics(
            "a", "profile", 10, 0.001, 0.0001, 0, True, 0.8, 0.9, 1.0, 0.0, ""
        ),
        OjaEpochMetrics(
            "b", "profile", 10, 0.001, 0.0001, 1, True, 1.0, 1.0, 1.2, 0.2, ""
        ),
    ]

    rows = aggregate_runs(metrics)

    assert len(rows) == 1
    assert rows[0]["direction_similarity_mean"] == pytest.approx(0.9)
    assert rows[0]["direction_similarity_std"] == pytest.approx(np.sqrt(0.02))
    assert rows[0]["norm_error_mean"] == pytest.approx(0.1)
