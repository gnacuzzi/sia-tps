import numpy as np

from sia_tp5 import Adam, GradientDescent, Momentum, RMSProp


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
