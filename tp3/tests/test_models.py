import numpy as np
import pytest

from sia_tp3 import MultilayerPerceptron, Perceptron, fit


@pytest.mark.parametrize('activation,target,expected_w,expected_b', [
    ('step', -1.0, [0.2, -0.2], -0.2),
    ('linear', 0.5, [0.025], 0.05),
    ('tanh', np.tanh(0.5), [0.05 * np.tanh(0.5)], 0.1 * np.tanh(0.5)),
])
def test_manual_first_update(activation, target, expected_w, expected_b):
    X = [[-1, 1]] if activation == 'step' else [[0.5]]
    model = Perceptron(len(X[0]), activation=activation, init_scale=0)
    model.train_sample(X, [[target]], learning_rate=0.1)
    np.testing.assert_allclose(model.weights[0][0], expected_w)
    np.testing.assert_allclose(model.biases[0], [expected_b])


def test_step_boundary_and_correct_sample_do_not_update():
    model = Perceptron(2, activation='step', init_scale=0)
    assert model.predict([[0, 0]])[0, 0] == 1
    model.train_sample([[1, 1]], [[1]], learning_rate=0.1)
    np.testing.assert_array_equal(model.weights[0], [[0, 0]])
    np.testing.assert_array_equal(model.biases[0], [0])
    with pytest.raises(ValueError, match='Rosenblatt'):
        model.gradients([[1, 1]], [[1]])


@pytest.mark.parametrize('architecture,activations,beta', [
    ([2, 1], ['linear'], 1.0),
    ([2, 1], ['tanh'], 0.7),
    ([2, 1], ['logistic'], 0.7),
    ([2, 2, 1], ['tanh', 'tanh'], 1.0),
    ([2, 3, 2, 1], ['tanh', 'tanh', 'tanh'], 1.0),
    ([3, 4, 2], ['tanh', 'logistic'], 0.8),
])
def test_every_weight_and_bias_against_finite_differences(architecture, activations, beta):
    model = MultilayerPerceptron(architecture, activations=activations, beta=beta, seed=8)
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3, architecture[0]))
    y = rng.uniform(-0.8, 0.8, size=(3, architecture[-1]))
    before = [a.copy() for a in model.weights + model.biases]
    dw, db = model.gradients(X, y)
    for original, current in zip(before, model.weights + model.biases):
        np.testing.assert_array_equal(original, current)
    epsilon = 1e-6
    for parameter, derivative in zip(model.weights + model.biases, dw + db):
        numeric = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            initial = parameter[index]
            parameter[index] = initial + epsilon
            plus = model.loss(X, y)
            parameter[index] = initial - epsilon
            minus = model.loss(X, y)
            parameter[index] = initial
            numeric[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(derivative, numeric, atol=1e-8, rtol=1e-5)


def test_multilayer_manual_forward_and_update():
    model = MultilayerPerceptron([2, 2, 1], activations=['tanh', 'tanh'])
    model.weights[0][...] = [[1, -1], [-1, 1]]
    model.biases[0][...] = [-1, -1]
    model.weights[1][...] = [[1, 1]]
    model.biases[1][...] = [1]
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    np.testing.assert_allclose(model.predict(X)[:, 0], [0.6449, 0.6449, -0.4802, -0.4802], atol=5e-5)
    hidden = np.array([np.tanh(-3), np.tanh(1)])
    output = np.tanh(hidden.sum() + 1)
    delta_out = (1 - output) * (1 - output ** 2)
    delta_hidden = delta_out * (1 - hidden ** 2)
    expected_w0 = model.weights[0] + 0.1 * np.outer(delta_hidden, [-1, 1])
    expected_w1 = model.weights[1] + 0.1 * delta_out * hidden
    expected_b0 = model.biases[0] + 0.1 * delta_hidden
    expected_b1 = model.biases[1] + 0.1 * delta_out
    model.train_sample([[-1, 1]], [[1]], learning_rate=0.1)
    for actual, expected in zip(model.weights + model.biases,
                                [expected_w0, expected_w1, expected_b0, expected_b1]):
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('activation', ['step', 'linear', 'tanh', 'logistic'])
def test_save_load_and_continue_training(tmp_path, activation):
    model = Perceptron(2, activation=activation, beta=0.7, seed=3)
    X, y = [[-1, 1]], [[1]]
    model.train_sample(X, y, learning_rate=0.1)
    path = tmp_path / 'model.npz'
    model.save(path)
    restored = MultilayerPerceptron.load(path)
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    for network in [model, restored]:
        network.train_sample(X, y, learning_rate=0.1)
    for actual, expected in zip(restored.weights + restored.biases, model.weights + model.biases):
        np.testing.assert_array_equal(actual, expected)


def test_reproducible_fit_and_history_are_measured_after_epoch():
    X = np.array([[-1], [0], [1]])
    models = [Perceptron(1, activation='linear', seed=7) for _ in range(2)]
    options = dict(learning_rate=0.1, max_epochs=30, target_mse=1e-6, shuffle=True, seed=9)
    histories = [fit(model, X, X, **options) for model in models]
    assert histories[0] == histories[1]
    np.testing.assert_array_equal(models[0].weights[0], models[1].weights[0])
    assert histories[0][-1]['mse'] == pytest.approx(np.mean((models[0].predict(X) - X) ** 2))


def test_step_cannot_solve_xor_and_does_not_report_convergence():
    X = np.array([[-1, 1], [1, -1], [-1, -1], [1, 1]])
    y = np.array([[1], [1], [-1], [-1]])
    model = Perceptron(2, activation='step', seed=0)
    history = fit(model, X, y, learning_rate=0.1, max_epochs=100, target_mse=0,
                  shuffle=False, seed=0, require_bipolar_accuracy=True)
    assert history[-1]['epoch'] == 100
    assert not history[-1]['converged']
    assert history[-1]['accuracy'] < 1


@pytest.mark.parametrize('X,y', [([[1, 2]], [1]), ([[1]], [[1]]),
                                ([[float('nan'), 1]], [[1]]), ([], [])])
def test_invalid_data_are_rejected(X, y):
    model = Perceptron(2, activation='linear')
    with pytest.raises(ValueError):
        model.train_sample(X, y, learning_rate=0.1)


def test_logistic_remains_finite_at_extreme_inputs():
    model = Perceptron(1, activation='logistic')
    model.weights[0][...] = 1
    np.testing.assert_allclose(model.predict([[-1000], [0], [1000]])[:, 0], [0, 0.5, 1])


def test_multioutput_forward_shape():
    model = MultilayerPerceptron([784, 8, 10], activations=['tanh', 'logistic'])
    assert model.predict(np.zeros((2, 784))).shape == (2, 10)
