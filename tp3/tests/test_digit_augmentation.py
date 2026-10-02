from functools import partial

import numpy as np
import pytest

from sia_tp3 import MultilayerPerceptron, fit
from sia_tp3.digit_augmentation import (random_rotate_images,
                                        random_translate_images,
                                        random_translate_rotate_images)


def test_translation_moves_pixels_with_zero_padding_and_no_wrap():
    image = np.zeros((28, 28))
    image[14, 14] = 1
    image[0, 0] = 2
    X = image.reshape(1, -1)
    translated = random_translate_images(
        X, np.random.default_rng(7), max_shift=1, probability=1).reshape(28, 28)

    center_position = np.argwhere(translated == 1)
    assert center_position.shape == (1, 2)
    dy, dx = center_position[0] - np.array([14, 14])
    assert -1 <= dy <= 1 and -1 <= dx <= 1 and (dy, dx) != (0, 0)
    expected = np.zeros((28, 28))
    expected[14 + dy, 14 + dx] = 1
    if dy >= 0 and dx >= 0:
        expected[dy, dx] = 2
    np.testing.assert_array_equal(translated, expected)


def test_translation_is_reproducible_does_not_mutate_and_probability_zero_is_identity():
    X = np.arange(3 * 784, dtype=np.float64).reshape(3, 784)
    original = X.copy()
    first = random_translate_images(
        X, np.random.default_rng(11), max_shift=2, probability=0.75)
    second = random_translate_images(
        X, np.random.default_rng(11), max_shift=2, probability=0.75)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(
        random_translate_images(
            X, np.random.default_rng(3), max_shift=1, probability=0),
        X,
    )


@pytest.mark.parametrize("max_shift,probability", [(0, 0.5), (28, 0.5), (1, -0.1), (1, 1.1)])
def test_translation_rejects_invalid_parameters(max_shift, probability):
    with pytest.raises(ValueError):
        random_translate_images(
            np.zeros((1, 784)), np.random.default_rng(0),
            max_shift=max_shift, probability=probability)


def test_zero_probability_preserves_training_exactly():
    X = np.zeros((4, 784))
    X[np.arange(4), np.arange(4)] = 1
    y = np.eye(10)[[0, 1, 2, 3]]
    models = [MultilayerPerceptron(
        [784, 4, 10], activations=["tanh", "softmax"],
        loss="categorical_cross_entropy", seed=5, init_scale=0.1)
        for _ in range(2)]
    options = dict(learning_rate=0.01, batch_size=2, max_epochs=2,
                   target_mse=0, shuffle=True, seed=9)
    plain = fit(models[0], X, y, **options)
    augmented = fit(
        models[1], X, y, **options,
        batch_transform=partial(
            random_translate_images, max_shift=1, probability=0),
        batch_transform_seed=13,
    )
    assert plain == augmented
    for first, second in zip(models[0].weights + models[0].biases,
                             models[1].weights + models[1].biases):
        np.testing.assert_array_equal(first, second)


def test_rotation_preserves_center_and_uses_bilinear_interpolation():
    image = np.zeros((28, 28))
    image[13, 13] = 1
    rotated = random_rotate_images(
        image.reshape(1, -1), np.random.default_rng(4),
        max_angle_degrees=5, probability=1).reshape(28, 28)

    assert rotated.min() >= 0
    assert rotated.max() <= 1
    assert rotated.sum() == pytest.approx(1, abs=0.02)
    assert rotated[12:15, 12:15].sum() == pytest.approx(rotated.sum())


def test_rotation_is_reproducible_non_mutating_and_zero_probability_is_identity():
    X = np.arange(2 * 784, dtype=np.float64).reshape(2, 784)
    original = X.copy()
    first = random_rotate_images(
        X, np.random.default_rng(9), max_angle_degrees=4,
        probability=0.75)
    second = random_rotate_images(
        X, np.random.default_rng(9), max_angle_degrees=4,
        probability=0.75)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(random_rotate_images(
        X, np.random.default_rng(2), max_angle_degrees=4, probability=0), X)


def test_combined_augmentation_zero_probabilities_is_identity():
    X = np.arange(784, dtype=np.float64).reshape(1, -1)
    result = random_translate_rotate_images(
        X, np.random.default_rng(1), max_shift=1,
        translation_probability=0, max_angle_degrees=4,
        rotation_probability=0)
    np.testing.assert_array_equal(result, X)


@pytest.mark.parametrize("angle,probability", [(0, 0.5), (46, 0.5), (4, -0.1), (4, 1.1)])
def test_rotation_rejects_invalid_parameters(angle, probability):
    with pytest.raises(ValueError):
        random_rotate_images(
            np.zeros((1, 784)), np.random.default_rng(0),
            max_angle_degrees=angle, probability=probability)
