import numpy as np
import pytest

from sia_tp5.data import load_font, row_to_bits


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0x00, [0, 0, 0, 0, 0]),
        (0x04, [0, 0, 1, 0, 0]),
        (0x0E, [0, 1, 1, 1, 0]),
        (0x0D, [0, 1, 1, 0, 1]),
        (0x13, [1, 0, 0, 1, 1]),
        (0x1F, [1, 1, 1, 1, 1]),
    ],
)
def test_row_to_bits(value, expected):
    np.testing.assert_array_equal(row_to_bits(value), expected)


def test_row_to_bits_rejects_values_that_do_not_fit():
    with pytest.raises(ValueError):
        row_to_bits(0x20)


def test_load_font_returns_the_expected_dataset():
    dataset = load_font()

    assert dataset.images.shape == (32, 7, 5)
    assert dataset.patterns.shape == (32, 35)
    assert set(np.unique(dataset.images)) == {0.0, 1.0}
    assert dataset.codepoints == tuple(range(0x60, 0x80))
    assert dataset.labels[:3] == ("`", "a", "b")
    assert dataset.labels[-1] == "DEL"


def test_letter_a_matches_the_rows_converted_by_hand():
    dataset = load_font()
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 0, 1],
        [1, 0, 0, 1, 1],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 1],
    ])

    np.testing.assert_array_equal(dataset.images[1], expected)
    np.testing.assert_array_equal(dataset.patterns[1], expected.reshape(35))


def test_most_significant_bit_is_the_left_pixel():
    dataset = load_font()

    # La primera fila de la b es 0x10: el pixel izquierdo debe estar encendido.
    np.testing.assert_array_equal(dataset.images[2, 0], [1, 0, 0, 0, 0])
