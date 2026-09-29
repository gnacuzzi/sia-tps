import numpy as np
import pytest

from sia_tp3 import (Adam, AdaptiveLearningRate, GradientDescent, Momentum,
                     Perceptron, RMSProp, fit)


def test_gradient_descent_matches_manual_update():
    parameter = np.array([1.0, -1.0])
    GradientDescent(0.1).step([parameter], [np.array([0.5, -2.0])])
    np.testing.assert_allclose(parameter, [0.95, -0.8])


def test_momentum_reuses_previous_change():
    parameter = np.array([1.0])
    optimizer = Momentum(0.1, alpha=0.9)
    optimizer.step([parameter], [np.array([0.5])])
    np.testing.assert_allclose(parameter, [0.95])
    optimizer.step([parameter], [np.array([0.5])])
    np.testing.assert_allclose(parameter, [0.855])


def test_adaptive_learning_rate_follows_consecutive_epoch_losses():
    optimizer = AdaptiveLearningRate(
        0.1, increase_by=0.05, decrease_fraction=0.5, patience=2)
    for loss in [3.0, 2.0, 1.0]:
        optimizer.epoch_end(loss)
    assert optimizer.learning_rate == pytest.approx(0.15)
    for loss in [2.0, 3.0]:
        optimizer.epoch_end(loss)
    assert optimizer.learning_rate == pytest.approx(0.075)


def test_rmsprop_places_epsilon_inside_square_root_as_in_class():
    parameter = np.array([1.0])
    optimizer = RMSProp(0.1, gamma=0.9, epsilon=0.01)
    optimizer.step([parameter], [np.array([2.0])])
    expected = 1.0 - 0.1 * 2.0 / np.sqrt(0.4 + 0.01)
    np.testing.assert_allclose(parameter, [expected])


def test_adam_applies_bias_correction_and_epsilon_outside_root():
    parameter = np.array([1.0])
    optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
    optimizer.step([parameter], [np.array([2.0])])
    np.testing.assert_allclose(parameter, [1.0 - 0.001 * 2.0 / (2.0 + 1e-8)])


def test_online_and_batch_use_different_gradient_groupings():
    X = np.array([[1.0], [2.0]])
    y = X.copy()
    online = Perceptron(1, activation="linear", init_scale=0)
    batch = Perceptron(1, activation="linear", init_scale=0)
    options = dict(max_epochs=1, target_mse=0, shuffle=False, seed=0)
    fit(online, X, y, learning_rate=0.1, batch_size=1, **options)
    fit(batch, X, y, learning_rate=0.1, batch_size=len(X), **options)
    np.testing.assert_allclose(online.weights[0], [[0.44]])
    np.testing.assert_allclose(online.biases[0], [0.27])
    np.testing.assert_allclose(batch.weights[0], [[0.25]])
    np.testing.assert_allclose(batch.biases[0], [0.15])


def test_none_batch_size_means_full_batch():
    X = np.array([[1.0], [2.0]])
    y = X.copy()
    models = [Perceptron(1, activation="linear", init_scale=0) for _ in range(2)]
    options = dict(learning_rate=0.1, max_epochs=1, target_mse=0,
                   shuffle=False, seed=0)
    fit(models[0], X, y, batch_size=None, **options)
    fit(models[1], X, y, batch_size=len(X), **options)
    for actual, expected in zip(models[0].weights + models[0].biases,
                                models[1].weights + models[1].biases):
        np.testing.assert_array_equal(actual, expected)


def test_optimizer_and_learning_rate_cannot_both_be_given():
    model = Perceptron(1, activation="linear")
    with pytest.raises(ValueError, match="no indicar ambos"):
        fit(model, [[1]], [[1]], learning_rate=0.1,
            optimizer=Adam(), max_epochs=1, target_mse=0, shuffle=False, seed=0)


@pytest.mark.parametrize("optimizer,batch_size", [(Momentum(0.1, alpha=0.9), 1),
                                                   (GradientDescent(0.1), 2)])
def test_step_perceptron_remains_rosenblatt_online(optimizer, batch_size):
    model = Perceptron(1, activation="step")
    with pytest.raises(ValueError, match="Rosenblatt"):
        fit(model, [[-1], [1]], [[-1], [1]], optimizer=optimizer,
            batch_size=batch_size, max_epochs=1, target_mse=0,
            shuffle=False, seed=0)
