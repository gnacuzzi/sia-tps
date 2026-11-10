import numpy as np

from sia_tp5 import (Adam, ExponentialDecay, GradientDescent, Momentum,
                     RMSProp, optimizer_from_config)


def test_gradient_descent_uses_the_current_gradient():
    parameter = np.array([1.0])
    optimizer = GradientDescent(learning_rate=0.1)

    optimizer.step([parameter], [np.array([0.4])])

    np.testing.assert_allclose(parameter, [0.96])


def test_momentum_accumulates_the_previous_change():
    parameter = np.array([1.0])
    optimizer = Momentum(learning_rate=0.1, alpha=0.9)

    optimizer.step([parameter], [np.array([0.4])])
    # Primer cambio: -0.1 * 0.4 = -0.04.
    np.testing.assert_allclose(parameter, [0.96])

    optimizer.step([parameter], [np.array([0.4])])
    # Segundo cambio: -0.04 + 0.9 * -0.04 = -0.076.
    np.testing.assert_allclose(parameter, [0.884])


def test_rmsprop_reduces_the_step_using_the_squared_average():
    parameter = np.array([1.0])
    optimizer = RMSProp(learning_rate=0.1, gamma=0.9, epsilon=1e-8)
    gradient = np.array([0.4])

    optimizer.step([parameter], [gradient])

    squared_average = 0.9 * 0.0 + 0.1 * 0.4 ** 2
    expected = 1.0 - 0.1 * 0.4 / np.sqrt(squared_average + 1e-8)
    np.testing.assert_allclose(parameter, [expected])


def test_adam_combines_first_and_second_moments():
    parameter = np.array([1.0])
    optimizer = Adam(
        learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8
    )

    optimizer.step([parameter], [np.array([0.4])])

    # En el primer paso, la correccion de sesgo recupera 0.4 y 0.4^2.
    expected = 1.0 - 0.1 * 0.4 / (np.sqrt(0.4 ** 2) + 1e-8)
    np.testing.assert_allclose(parameter, [expected])


def test_exponential_decay_reaches_both_learning_rate_endpoints():
    parameter = np.array([1.0])
    optimizer = ExponentialDecay(
        GradientDescent(learning_rate=0.1),
        final_learning_rate=0.001,
        total_steps=3,
    )

    observed_rates = []
    for _ in range(3):
        optimizer.step([parameter], [np.array([0.0])])
        observed_rates.append(optimizer.learning_rate)

    np.testing.assert_allclose(observed_rates, [0.1, 0.01, 0.001])


def test_optimizer_config_can_build_exponential_decay():
    optimizer = optimizer_from_config({
        "name": "gradient_descent",
        "learning_rate": 0.03,
        "schedule": {
            "name": "exponential",
            "final_learning_rate": 0.003,
            "total_steps": 10,
        },
    })

    assert isinstance(optimizer, ExponentialDecay)
    assert optimizer.learning_rate == 0.03
