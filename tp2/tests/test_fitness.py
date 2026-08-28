import pytest
from PIL import Image

from sia_tp2.domain.fitness import fitness_from_error, normalized_mse


def test_identical_images_have_zero_error_and_maximum_fitness() -> None:
    image = Image.new("RGB", (4, 3), (12, 34, 56))

    error = normalized_mse(image, image.copy())

    assert error == 0.0
    assert fitness_from_error(error, 1e-12) == 1.0


def test_white_against_black_has_maximum_normalized_error() -> None:
    white = Image.new("RGB", (2, 2), (255, 255, 255))
    black = Image.new("RGB", (2, 2), (0, 0, 0))

    error = normalized_mse(white, black)

    assert error == 1.0
    assert fitness_from_error(error, 1e-12) == 1e-12


def test_nmse_requires_equal_dimensions() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        normalized_mse(
            Image.new("RGB", (2, 2), "white"),
            Image.new("RGB", (3, 2), "white"),
        )

