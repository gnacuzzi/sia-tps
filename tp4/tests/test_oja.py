"""Pruebas unitarias y sintéticas de la regla de Oja."""

import numpy as np
import pytest

from sia_tp4.oja import (
    OjaConfig,
    _oja_update,
    align_sign,
    direction_similarity,
    fit_oja,
)
from sia_tp4.pca import fit_pca


def _config(**overrides):
    values = {
        "epochs": 20,
        "learning_rate_start": 0.01,
        "learning_rate_end": 0.001,
        "shuffle": True,
        "seed": 7,
    }
    values.update(overrides)
    return OjaConfig(**values)


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="epochs"):
        _config(epochs=0)
    with pytest.raises(ValueError, match="tasas"):
        _config(learning_rate_start=0.6)
    with pytest.raises(ValueError, match="tasas"):
        _config(learning_rate_start=0.01, learning_rate_end=0.02)
    with pytest.raises(ValueError, match="shuffle"):
        _config(shuffle=1)
    with pytest.raises(ValueError, match="seed"):
        _config(seed=True)


def test_update_matches_the_rule_seen_in_class():
    weights = np.asarray([0.2, 0.4])
    sample = np.asarray([1.5, -0.5])
    rate = 0.1
    output = float(sample @ weights)

    updated = _oja_update(weights, sample, rate)

    expected = weights + rate * output * (sample - output * weights)
    np.testing.assert_allclose(updated, expected)


def test_training_is_reproducible_and_preserves_epoch_history():
    values = np.asarray(
        [[1.0, 2.0], [2.0, 4.1], [3.0, 5.9], [4.0, 8.2]]
    )

    first = fit_oja(values, ("a", "b"), _config())
    second = fit_oja(values, ("a", "b"), _config())

    np.testing.assert_allclose(first.weights, second.weights)
    np.testing.assert_allclose(first.weight_history, second.weight_history)
    assert first.weight_history.shape == (first.config.epochs + 1, 2)
    np.testing.assert_allclose(first.weight_history[-1], first.weights)
    assert first.weight_norm_history.shape == (first.config.epochs + 1,)


def test_projection_reuses_training_standardization():
    values = np.asarray(
        [[1.0, 10.0], [2.0, 8.0], [4.0, 3.0], [5.0, 1.0]]
    )
    model = fit_oja(values, ("a", "b"), _config())

    expected = model.standardizer.transform(values) @ model.weights

    np.testing.assert_allclose(model.transform(values), expected)


def test_sign_alignment_treats_opposite_eigenvectors_as_equivalent():
    reference = np.asarray([1.0, -2.0, 3.0])

    np.testing.assert_array_equal(align_sign(-reference, reference), reference)
    assert direction_similarity(-reference, reference) == pytest.approx(1.0)


def test_synthetic_data_converges_to_the_first_pca_direction():
    generator = np.random.default_rng(19)
    latent = generator.normal(size=600)
    values = np.column_stack(
        (
            latent + generator.normal(scale=0.08, size=latent.size),
            0.8 * latent + generator.normal(scale=0.10, size=latent.size),
            generator.normal(size=latent.size),
        )
    )
    names = ("correlated_a", "correlated_b", "noise")
    pca = fit_pca(values, names)
    model = fit_oja(
        values,
        names,
        _config(
            epochs=80,
            learning_rate_start=0.005,
            learning_rate_end=0.0005,
        ),
    )

    similarity = direction_similarity(model.weights, pca.components[:, 0])
    oja_scores = model.transform(values)
    if model.weights @ pca.components[:, 0] < 0:
        oja_scores = -oja_scores
    score_correlation = np.corrcoef(oja_scores, pca.scores[:, 0])[0, 1]

    assert similarity >= 0.995
    assert score_correlation >= 0.995
    assert np.linalg.norm(model.weights) == pytest.approx(1.0, abs=0.08)
