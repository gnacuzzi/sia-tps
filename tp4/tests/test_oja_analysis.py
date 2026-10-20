"""Pruebas del análisis final de Oja."""

from pathlib import Path

import numpy as np

from sia_tp4.oja_analysis import OjaSeedResult, load_design, select_representative


def _result(seed, similarity, correlation, norm_error):
    return OjaSeedResult(
        seed=seed,
        weights=np.asarray([1.0, 0.0]),
        weight_history=np.asarray([[1.0, 0.0]]),
        indices=np.asarray([0.0]),
        direction_similarity=similarity,
        score_correlation=correlation,
        weight_norm=1.0 + norm_error,
        norm_error=norm_error,
    )


def test_final_design_uses_selected_configuration_and_ten_seeds():
    path = Path(__file__).parents[1] / "configs" / "oja" / "final.json"

    design = load_design(path)

    assert design.epochs == 100
    assert design.learning_rate_start == 0.001
    assert design.learning_rate_end == 0.0001
    assert design.seeds == tuple(range(10))
    assert design.shuffle


def test_representative_is_central_instead_of_best_run():
    results = [
        _result(0, 0.90, 0.91, 0.10),
        _result(1, 0.95, 0.96, 0.05),
        _result(2, 1.00, 1.00, 0.00),
    ]

    representative = select_representative(results)

    assert representative.seed == 1
