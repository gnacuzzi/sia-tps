import numpy as np

from sia_tp5 import Adam, Autoencoder, fit_autoencoder, load_font


def _small_autoencoder(seed=0, init_scale=0.2):
    return Autoencoder(
        [3, 2, 1, 2, 3],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=seed,
        init_scale=init_scale,
    )


def test_encode_and_decode_match_the_complete_forward_pass():
    model = Autoencoder(
        [35, 15, 2, 15, 35],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=3,
    )
    inputs = load_font().patterns[:4]

    latent = model.encode(inputs)
    decoded = model.decode(latent)

    assert latent.shape == (4, 2)
    assert decoded.shape == (4, 35)
    np.testing.assert_allclose(decoded, model.predict(inputs))
    assert np.all((decoded >= 0) & (decoded <= 1))


def test_backpropagation_matches_a_numerical_gradient():
    model = _small_autoencoder()
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    weight_gradients, bias_gradients = model.gradients(inputs)
    analytical = weight_gradients + bias_gradients
    epsilon = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss(inputs)
            parameter[index] = original - epsilon
            minus = model.loss(inputs)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-6)


def test_bce_backpropagation_matches_a_numerical_gradient():
    model = Autoencoder(
        [3, 2, 1, 2, 3],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        loss="bce",
        seed=0,
        init_scale=0.2,
    )
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    weight_gradients, bias_gradients = model.gradients(inputs)
    analytical = weight_gradients + bias_gradients
    epsilon = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss(inputs)
            parameter[index] = original - epsilon
            minus = model.loss(inputs)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-6)


def test_bce_is_finite_for_saturated_probabilities():
    model = Autoencoder(
        [2, 1, 2],
        latent_layer=1,
        activations=["linear", "logistic"],
        loss="bce",
        seed=0,
    )
    model.weights[-1][:] = 1_000.0
    model.biases[-1][:] = 1_000.0

    assert np.isfinite(model.loss(np.array([[0.0, 1.0]])))


def test_saved_model_preserves_predictions_when_loaded(tmp_path):
    model = _small_autoencoder(seed=4)
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    expected = model.predict(inputs)
    path = tmp_path / "model.npz"

    model.save(path)
    loaded = Autoencoder.load(path)

    assert loaded.loss_name == model.loss_name
    np.testing.assert_allclose(loaded.predict(inputs), expected)


def test_autoencoder_can_memorize_one_letter():
    letter_a = load_font().patterns[1:2]
    model = Autoencoder(
        [35, 15, 2, 15, 35],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=0,
    )
    history = fit_autoencoder(
        model,
        letter_a,
        optimizer=Adam(learning_rate=0.01),
        max_epochs=2_000,
        target_max_pixel_errors=0,
    )

    assert history[-1]["max_pixel_errors"] == 0
    assert history[-1]["epoch"] < 2_000
    assert model.encode(letter_a).shape == (1, 2)
