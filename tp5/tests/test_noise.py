import numpy as np
import pytest

from sia_tp5 import bit_flip


def test_bit_flip_is_reproducible_for_the_same_seed():
    inputs = np.tile([[0.0, 1.0, 0.0, 1.0]], (8, 1))

    first = bit_flip(inputs, 0.35, generator=np.random.default_rng(17))
    second = bit_flip(inputs, 0.35, generator=np.random.default_rng(17))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(inputs, np.tile([[0.0, 1.0, 0.0, 1.0]], (8, 1)))


def test_bit_flip_extreme_probabilities_have_known_results():
    inputs = np.array([[0.0, 1.0, 1.0]])

    unchanged = bit_flip(inputs, 0.0, generator=np.random.default_rng(0))
    inverted = bit_flip(inputs, 1.0, generator=np.random.default_rng(0))

    np.testing.assert_array_equal(unchanged, inputs)
    np.testing.assert_array_equal(inverted, 1.0 - inputs)


def test_bit_flip_generator_produces_fresh_reproducible_corruptions():
    inputs = np.tile([[0.0, 1.0]], (32, 16))
    first_generator = np.random.default_rng(4)
    first = bit_flip(inputs, 0.5, generator=first_generator)
    second = bit_flip(inputs, 0.5, generator=first_generator)

    replay_generator = np.random.default_rng(4)
    replay_first = bit_flip(inputs, 0.5, generator=replay_generator)
    replay_second = bit_flip(inputs, 0.5, generator=replay_generator)

    assert not np.array_equal(first, second)
    np.testing.assert_array_equal(first, replay_first)
    np.testing.assert_array_equal(second, replay_second)


@pytest.mark.parametrize("probability", [-0.1, 1.1, np.nan, True])
def test_bit_flip_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        bit_flip(
            np.array([[0.0, 1.0]]),
            probability,
            generator=np.random.default_rng(0),
        )


def test_bit_flip_rejects_non_binary_inputs():
    with pytest.raises(ValueError, match="binarias"):
        bit_flip(
            np.array([[0.0, 0.5]]),
            0.1,
            generator=np.random.default_rng(0),
        )
