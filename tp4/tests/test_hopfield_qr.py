"""Pruebas de carga y representación de los QR para Hopfield."""

from pathlib import Path

import numpy as np
import pytest

from sia_tp4.hopfield_qr import (
    QR_FILENAMES,
    _binary_grayscale,
    energy_trajectory_rows,
    gram_matrices,
    load_qr_patterns,
    pairwise_similarity_rows,
    pixel_balance_rows,
    run_zero_noise_recall,
)


DATA_DIRECTORY = Path(__file__).parents[1] / "datos" / "hopfield"


def test_loads_the_five_qr_as_bipolar_matrices_and_vectors():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    assert patterns.names == (
        "beverages",
        "dessert",
        "food",
        "specials",
        "wine",
    )
    assert patterns.images.shape == (5, 37, 37)
    assert patterns.vectors.shape == (5, 37 * 37)
    assert patterns.images.dtype == np.int8
    assert set(np.unique(patterns.images)) == {-1, 1}
    np.testing.assert_array_equal(
        patterns.vectors,
        patterns.images.reshape(5, -1),
    )


def test_black_is_positive_and_white_is_negative():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    # En el primer QR, la esquina pertenece al patrón de búsqueda negro y el
    # módulo siguiente a ese patrón es blanco.
    assert patterns.images[0, 0, 0] == 1
    assert patterns.images[0, 0, 7] == -1


def test_binary_validation_rejects_wrong_shape_channels_and_gray_values():
    path = Path("invalid.png")

    with pytest.raises(ValueError, match="37x37"):
        _binary_grayscale(np.zeros((5, 5)), path)
    with pytest.raises(ValueError, match="escala de grises"):
        _binary_grayscale(np.zeros((37, 37, 3)), path)
    with pytest.raises(ValueError, match="debe ser binaria"):
        _binary_grayscale(np.full((37, 37), 0.5), path)


def test_loader_reports_a_missing_expected_file(tmp_path):
    with pytest.raises(ValueError, match=QR_FILENAMES[0]):
        load_qr_patterns(tmp_path)


def test_gram_matrix_is_symmetric_normalized_and_does_not_overflow():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    gram, normalized = gram_matrices(patterns)

    assert gram.dtype == np.int64
    np.testing.assert_array_equal(gram, gram.T)
    np.testing.assert_array_equal(np.diag(gram), 37 * 37)
    np.testing.assert_allclose(normalized, gram / (37 * 37))
    np.testing.assert_allclose(np.diag(normalized), 1.0)


def test_pairwise_rows_cover_every_pair_and_match_bipolar_distance():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    rows = pairwise_similarity_rows(patterns)

    assert len(rows) == 10
    for row in rows:
        assert (
            row["inner_product"]
            == 37 * 37 - 2 * row["different_pixels"]
        )
    closest = max(rows, key=lambda row: row["absolute_inner_product"])
    assert (closest["first"], closest["second"]) == ("food", "wine")
    assert closest["inner_product"] == 1177


def test_pixel_balance_accounts_for_every_pixel():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    rows = pixel_balance_rows(patterns)

    assert len(rows) == 5
    assert all(
        row["black_pixels"] + row["white_pixels"] == 37 * 37
        for row in rows
    )


def test_five_stored_qr_collapse_to_one_spurious_state_without_noise():
    patterns = load_qr_patterns(DATA_DIRECTORY)

    rows, cases = run_zero_noise_recall(patterns)

    assert len(rows) == len(cases) == 5
    assert all(not row["remains_stable"] for row in rows)
    assert all(row["outcome"] == "spurious_fixed" for row in rows)
    assert all(row["status"] == "fixed_point" for row in rows)
    assert all(row["matched_pattern"] == "" for row in rows)
    assert [row["iterations"] for row in rows] == [2, 2, 2, 2, 2]
    assert [row["final_error_pixels"] for row in rows] == [225, 225, 71, 103, 91]
    assert len({case["result"].final_state.tobytes() for case in cases}) == 1
    for case in cases:
        result = case["result"]
        assert len(result.energies) == result.iterations + 1
        assert np.all(np.diff(result.energies) <= 1e-9)

    energy_rows = energy_trajectory_rows(cases)
    assert len(energy_rows) == 5 * 3
    assert [row["iteration"] for row in energy_rows[:3]] == [0, 1, 2]
    assert all(
        row["error_pixels"] == 0
        for row in energy_rows
        if row["iteration"] == 0
    )
