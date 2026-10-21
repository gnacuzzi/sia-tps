"""Pruebas de carga y representación de los QR para Hopfield."""

from pathlib import Path

import numpy as np
import pytest

from sia_tp4.hopfield_qr import (
    QR_FILENAMES,
    _binary_grayscale,
    gram_matrices,
    load_qr_patterns,
    pairwise_similarity_rows,
    pixel_balance_rows,
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
