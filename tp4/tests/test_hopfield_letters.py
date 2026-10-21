"""Pruebas del experimento controlado de Hopfield con letras."""

import json
import math

import numpy as np
import pytest

from sia_tp4.hopfield_letters import (
    LETTER_PATTERNS,
    LetterExperimentDesign,
    _bounded_error,
    add_flip_noise,
    gram_matrix,
    load_design,
    rank_letter_groups,
    run_experiment,
    summarize_by_letter,
    summarize_runs,
)


def _design():
    return LetterExperimentDesign(
        group_size=4,
        noise_levels=(0.0, 0.2),
        seeds=tuple(range(10)),
        max_iterations=20,
    )


def test_alphabet_contains_26_distinct_bipolar_patterns():
    assert tuple(sorted(LETTER_PATTERNS)) == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    flattened = []
    for pattern in LETTER_PATTERNS.values():
        assert pattern.shape == (5, 5)
        assert set(np.unique(pattern)) <= {-1, 1}
        flattened.append(tuple(pattern.reshape(-1)))
    assert len(set(flattened)) == 26


def test_gram_matrix_has_expected_properties_for_odd_patterns():
    matrix = gram_matrix(("A", "L", "T", "V"))

    assert matrix.shape == (4, 4)
    np.testing.assert_array_equal(matrix, matrix.T)
    np.testing.assert_array_equal(np.diag(matrix), 25)
    off_diagonal = matrix[np.triu_indices(4, 1)]
    assert np.all(np.abs(off_diagonal) >= 1)
    assert np.all(off_diagonal % 2 != 0)


def test_ranking_evaluates_every_group_and_is_ordered():
    ranking = rank_letter_groups(4)

    assert len(ranking) == math.comb(26, 4)
    keys = [
        (
            score.mean_absolute_inner_product,
            score.maximum_absolute_inner_product,
            score.maximum_pair_count,
            score.letters,
        )
        for score in ranking
    ]
    assert keys == sorted(keys)
    assert ranking[0].mean_absolute_inner_product <= ranking[-1].mean_absolute_inner_product


def test_flip_noise_is_exact_and_reproducible():
    pattern = LETTER_PATTERNS["A"].reshape(-1)

    first = add_flip_noise(pattern, 0.2, seed=3)
    second = add_flip_noise(pattern, 0.2, seed=3)
    other = add_flip_noise(pattern, 0.2, seed=4)

    assert np.count_nonzero(first != pattern) == 5
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, other)


def test_error_bars_are_bounded_to_the_valid_metric_range():
    means = np.asarray([0.0, 0.5, 0.9, 1.0])
    deviations = np.asarray([0.2, 0.1, 0.3, 0.2])

    errors = _bounded_error(means, deviations, 0.0, 1.0)

    np.testing.assert_allclose(errors[0], [0.0, 0.1, 0.3, 0.2])
    np.testing.assert_allclose(errors[1], [0.2, 0.1, 0.1, 0.0])
    assert np.all(means - errors[0] >= 0.0)
    assert np.all(means + errors[1] <= 1.0)


def test_design_requires_at_least_ten_unique_seeds(tmp_path):
    path = tmp_path / "design.json"
    path.write_text(
        json.dumps(
            {
                "group_size": 4,
                "noise_levels": [0, 0.1],
                "seeds": list(range(9)),
                "max_iterations": 20,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="al menos 10"):
        load_design(path)


def test_experiment_is_reproducible_and_summary_uses_ten_seeds():
    first_ranking, first_groups, first_rows = run_experiment(_design())
    second_ranking, second_groups, second_rows = run_experiment(_design())

    assert first_ranking == second_ranking
    assert first_groups == second_groups
    assert first_rows == second_rows
    assert len(first_rows) == 2 * 4 * 2 * 10
    summary = summarize_runs(first_rows)
    assert len(summary) == 2 * 2
    assert all(row["seed_count"] == 10 for row in summary)
    letter_summary = summarize_by_letter(first_rows)
    assert len(letter_summary) == 2 * 4 * 2
    assert all(row["seed_count"] == 10 for row in letter_summary)
