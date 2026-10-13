"""Pruebas unitarias de la implementación de Kohonen."""

import numpy as np
import pytest

from sia_tp4.kohonen import (
    KohonenConfig,
    KohonenModel,
    _neighborhood_mask,
    fit_kohonen,
)
from sia_tp4.pca import Standardizer


def _config(**overrides):
    values = {
        "grid_size": 2,
        "epochs": 2,
        "learning_rate_start": 0.5,
        "learning_rate_end": 0.1,
        "radius_start": 1.5,
        "radius_end": 1.0,
        "seed": 7,
    }
    values.update(overrides)
    return KohonenConfig(**values)


def test_config_rejects_invalid_schedules():
    with pytest.raises(ValueError, match="tasas"):
        _config(learning_rate_start=1.0)
    with pytest.raises(ValueError, match="radios"):
        _config(radius_start=0.5)
    with pytest.raises(ValueError, match="grid_size"):
        _config(grid_size=1)
    with pytest.raises(ValueError, match="grid_size"):
        _config(grid_size=2.5)


def test_radius_one_includes_winner_and_four_orthogonal_neighbors():
    coordinates = np.indices((3, 3)).transpose(1, 2, 0)

    neighborhood = _neighborhood_mask(
        coordinates, np.asarray([1, 1]), radius=1.0
    )

    expected = np.asarray(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ]
    )
    np.testing.assert_array_equal(neighborhood, expected)


def test_training_is_reproducible_and_keeps_expected_shape():
    values = np.asarray(
        [[1.0, 10.0], [2.0, 8.0], [4.0, 3.0], [5.0, 1.0]]
    )

    first = fit_kohonen(values, ("a", "b"), _config())
    second = fit_kohonen(values, ("a", "b"), _config())

    assert first.weights.shape == (2, 2, 2)
    assert np.isfinite(first.weights).all()
    np.testing.assert_allclose(first.weights, second.weights)


def test_hit_map_assigns_every_sample_once():
    values = np.asarray(
        [[1.0, 10.0], [2.0, 8.0], [4.0, 3.0], [5.0, 1.0]]
    )
    model = fit_kohonen(values, ("a", "b"), _config())

    hits = model.hit_map(values)

    assert hits.shape == (2, 2)
    assert hits.sum() == values.shape[0]
    assert np.all(hits >= 0)


def test_u_matrix_uses_four_orthogonal_neighbors():
    reference = np.asarray(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
    )
    standardizer = Standardizer.fit(reference, ("a", "b"))
    weights = np.asarray(
        [
            [[0.0, 0.0], [3.0, 0.0]],
            [[0.0, 4.0], [3.0, 4.0]],
        ]
    )
    model = KohonenModel(weights, ("a", "b"), standardizer, _config())

    expected = np.full((2, 2), 3.5)
    np.testing.assert_allclose(model.u_matrix(), expected)


def test_component_plane_can_return_original_units():
    values = np.asarray(
        [[10.0, 1.0], [20.0, 3.0], [30.0, 5.0], [40.0, 7.0]]
    )
    standardizer = Standardizer.fit(values, ("reach", "price"))
    weights = np.zeros((2, 2, 2))
    model = KohonenModel(weights, ("reach", "price"), standardizer, _config())

    np.testing.assert_allclose(model.component_plane("reach"), 25.0)
    np.testing.assert_allclose(
        model.component_plane("reach", original_scale=False), 0.0
    )
    with pytest.raises(ValueError, match="variable desconocida"):
        model.component_plane("missing")
