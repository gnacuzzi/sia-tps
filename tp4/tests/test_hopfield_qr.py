"""Pruebas de carga y representación de los QR para Hopfield."""

from pathlib import Path

import numpy as np
import pytest

from sia_tp4.hopfield_qr import (
    QR_FILENAMES,
    _binary_grayscale,
    load_qr_patterns,
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
