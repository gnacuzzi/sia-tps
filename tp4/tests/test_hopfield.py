"""Pruebas unitarias de la red de Hopfield discreta."""

import numpy as np
import pytest

from sia_tp4.hopfield import HopfieldModel, _update_state, fit_hopfield


def test_fit_rejects_non_bipolar_or_malformed_patterns():
    with pytest.raises(ValueError, match="matriz no vacía"):
        fit_hopfield(np.asarray([]))
    with pytest.raises(ValueError, match="únicamente -1 y 1"):
        fit_hopfield(np.asarray([[1, 0, -1]]))
    with pytest.raises(ValueError, match="finitos"):
        fit_hopfield(np.asarray([[1, np.nan, -1]]))
    with pytest.raises(ValueError, match="numérica"):
        fit_hopfield(np.asarray([["1", "-1"]]))


def test_weights_match_the_numerical_example_seen_in_class():
    patterns = np.asarray(
        [
            [1, 1, -1, -1],
            [-1, -1, 1, 1],
        ]
    )

    model = fit_hopfield(patterns)

    expected = np.asarray(
        [
            [0.0, 0.5, -0.5, -0.5],
            [0.5, 0.0, -0.5, -0.5],
            [-0.5, -0.5, 0.0, 0.5],
            [-0.5, -0.5, 0.5, 0.0],
        ]
    )
    np.testing.assert_allclose(model.weights, expected)
    np.testing.assert_array_equal(model.stored_patterns, patterns)
    np.testing.assert_allclose(model.weights, model.weights.T)
    np.testing.assert_allclose(np.diag(model.weights), 0.0)


def test_recall_matches_the_numerical_example_seen_in_class():
    first = np.asarray([1, 1, -1, -1])
    second = -first
    model = fit_hopfield(np.asarray([first, second]))

    result = model.recall(np.asarray([1, -1, -1, -1]))

    assert result.status == "fixed_point"
    assert result.converged
    assert result.iterations == 2
    assert result.cycle_length is None
    np.testing.assert_array_equal(result.final_state, first)
    np.testing.assert_array_equal(
        result.states,
        np.asarray(
            [
                [1, -1, -1, -1],
                [1, 1, -1, -1],
                [1, 1, -1, -1],
            ]
        ),
    )
    assert result.energies.shape == (3,)
    assert model.matching_pattern_index(result.final_state) == 0


def test_zero_local_field_preserves_the_previous_neuron_state():
    weights = np.zeros((3, 3), dtype=np.float64)
    state = np.asarray([1, -1, 1], dtype=np.int8)

    updated = _update_state(weights, state)

    np.testing.assert_array_equal(updated, state)


def test_roundoff_near_zero_preserves_the_previous_neuron_state():
    weights = np.asarray([[0.0, 5e-16], [5e-16, 0.0]])
    state = np.asarray([1, -1], dtype=np.int8)

    updated = _update_state(weights, state)

    np.testing.assert_array_equal(updated, state)


def test_energy_matches_its_definition():
    patterns = np.asarray([[1, 1, -1, -1], [-1, -1, 1, 1]])
    state = np.asarray([1, -1, -1, -1])
    model = fit_hopfield(patterns)

    expected = -0.5 * state @ model.weights @ state

    assert model.energy(state) == pytest.approx(expected)


def test_synchronous_update_detects_a_two_state_cycle():
    model = HopfieldModel(
        weights=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        stored_patterns=np.asarray([[1, 1]], dtype=np.int8),
    )

    result = model.recall(np.asarray([1, -1]), max_iterations=10)

    assert result.status == "cycle"
    assert not result.converged
    assert result.iterations == 2
    assert result.cycle_length == 2
    np.testing.assert_array_equal(result.states[0], result.states[-1])
    np.testing.assert_allclose(result.energies, result.energies[0])


def test_iteration_limit_stops_a_trajectory_before_repeating():
    model = HopfieldModel(
        weights=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        stored_patterns=np.asarray([[1, 1]], dtype=np.int8),
    )

    result = model.recall(np.asarray([1, -1]), max_iterations=1)

    assert result.status == "max_iterations"
    assert result.iterations == 1
    assert result.cycle_length is None
    assert result.states.shape == (2, 2)


def test_recall_validates_query_and_iteration_limit():
    model = fit_hopfield(np.asarray([[1, -1, 1]]))

    with pytest.raises(ValueError, match="longitud 3"):
        model.recall(np.asarray([1, -1]))
    with pytest.raises(ValueError, match="únicamente -1 y 1"):
        model.recall(np.asarray([1, 0, -1]))
    with pytest.raises(ValueError, match="max_iterations"):
        model.recall(np.asarray([1, -1, 1]), max_iterations=0)


def test_matching_pattern_returns_none_for_a_spurious_state():
    model = fit_hopfield(np.asarray([[1, 1, -1], [-1, 1, 1]]))

    assert model.matching_pattern_index(np.asarray([1, -1, 1])) is None


def test_model_rejects_weights_that_break_hopfield_invariants():
    patterns = np.asarray([[1, -1]])

    with pytest.raises(ValueError, match="simétrica"):
        HopfieldModel(np.asarray([[0.0, 1.0], [0.0, 0.0]]), patterns)
    with pytest.raises(ValueError, match="diagonal"):
        HopfieldModel(np.eye(2), patterns)
    with pytest.raises(ValueError, match="misma cantidad"):
        HopfieldModel(np.zeros((3, 3)), patterns)
