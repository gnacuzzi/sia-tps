import numpy as np

from sia_tp5 import (Adam, VariationalAutoencoder,
                     fit_variational_autoencoder, toy_bars)


def _small_vae(seed=0, kl_weight=0.0):
    return VariationalAutoencoder(
        [3, 2, 1, 2, 3],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=seed,
        init_scale=0.2,
        kl_weight=kl_weight,
    )


def test_reparameterization_has_expected_shape_and_value():
    model = VariationalAutoencoder(
        [9, 5, 2, 5, 9],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=3,
    )
    inputs = toy_bars()
    epsilon = np.full((6, 2), 0.25)

    mu, log_var = model.encode_distribution(inputs)
    z = model.encode(inputs, epsilon)

    assert mu.shape == (6, 2)
    assert log_var.shape == (6, 2)
    np.testing.assert_allclose(
        z, mu + np.exp(0.5 * log_var) * epsilon
    )
    np.testing.assert_allclose(
        model.decode(z), model.predict(inputs, epsilon)
    )


def test_reparameterized_gradients_match_finite_differences():
    model = _small_vae()
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    sample = np.array([[0.3], [-0.7]])
    weight_gradients, bias_gradients = model.gradients(inputs, epsilon=sample)
    analytical = weight_gradients + bias_gradients
    epsilon = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss(inputs, epsilon=sample)
            parameter[index] = original - epsilon
            minus = model.loss(inputs, epsilon=sample)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-6)


def test_nonzero_epsilon_propagates_reconstruction_gradient_to_log_var():
    model = _small_vae()
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    sample = np.array([[0.3], [-0.7]])
    weight_gradients, bias_gradients = model.gradients(inputs, epsilon=sample)
    log_var_weight_index = len(model.encoder_weights) + 1
    log_var_bias_index = len(model.encoder_biases) + 1

    assert np.linalg.norm(weight_gradients[log_var_weight_index]) > 0.0
    assert np.linalg.norm(bias_gradients[log_var_bias_index]) > 0.0


def test_zero_epsilon_recovers_deterministic_checkpoint():
    inputs = toy_bars()
    model = VariationalAutoencoder(
        [9, 8, 2, 8, 9],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=0,
    )

    epsilon = np.zeros((len(inputs), 2))
    mu, _ = model.encode_distribution(inputs)
    np.testing.assert_array_equal(model.encode(inputs, epsilon), mu)

    weight_gradients, bias_gradients = model.gradients(inputs, epsilon=epsilon)
    log_var_weight_index = len(model.encoder_weights) + 1
    log_var_bias_index = len(model.encoder_biases) + 1
    np.testing.assert_array_equal(
        weight_gradients[log_var_weight_index],
        np.zeros_like(model.log_var_weight),
    )
    np.testing.assert_array_equal(
        bias_gradients[log_var_bias_index], np.zeros_like(model.log_var_bias)
    )

    optimizer = Adam(learning_rate=0.01)
    for _ in range(3_000):
        weight_gradients, bias_gradients = model.gradients(
            inputs, epsilon=epsilon
        )
        optimizer.step(
            model.parameters, weight_gradients + bias_gradients
        )
        if np.max(np.sum(
                (model.predict(inputs, epsilon) >= 0.5) != inputs, axis=1
        )) == 0:
            break
    assert np.max(np.sum(
        (model.predict(inputs, epsilon) >= 0.5) != inputs, axis=1
    )) == 0


def test_sampling_is_reproducible_and_changes_between_draws():
    inputs = toy_bars()
    first = VariationalAutoencoder(
        [9, 5, 2, 5, 9],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=4,
        sampling_seed=12,
    )
    second = VariationalAutoencoder(
        [9, 5, 2, 5, 9],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=4,
        sampling_seed=12,
    )

    first_draw = first.encode(inputs)
    second_draw = first.encode(inputs)

    np.testing.assert_array_equal(first_draw, second.encode(inputs))
    np.testing.assert_array_equal(second_draw, second.encode(inputs))
    assert not np.array_equal(first_draw, second_draw)


def test_loss_components_use_the_closed_form_diagonal_kl():
    model = _small_vae(kl_weight=0.2)
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    epsilon = np.array([[0.3], [-0.7]])
    mu, log_var = model.encode_distribution(inputs)

    components = model.loss_components(inputs, epsilon=epsilon)
    expected_kl = np.mean(-0.5 * np.sum(
        1.0 + log_var - mu ** 2 - np.exp(log_var), axis=1
    ))

    np.testing.assert_allclose(components["kl"], expected_kl)
    np.testing.assert_allclose(
        components["total"],
        components["reconstruction"] + 0.2 * components["kl"],
    )


def test_total_loss_gradients_with_kl_match_finite_differences():
    model = _small_vae(kl_weight=0.2)
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    sample = np.array([[0.3], [-0.7]])
    weight_gradients, bias_gradients = model.gradients(inputs, epsilon=sample)
    analytical = weight_gradients + bias_gradients
    step = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            plus = model.loss(inputs, epsilon=sample)
            parameter[index] = original - step
            minus = model.loss(inputs, epsilon=sample)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-6)


def test_grouped_softmax_vae_gradients_with_kl_match_finite_differences():
    model = VariationalAutoencoder(
        [6, 4, 2, 4, 6],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "grouped_softmax"],
        loss="categorical_cross_entropy",
        categorical_groups=2,
        seed=9,
        init_scale=0.2,
        kl_weight=0.1,
    )
    inputs = np.array([
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ])
    sample = np.array([[0.25, -0.4], [-0.3, 0.6]])
    predictions = model.predict(inputs, sample).reshape(2, 2, 3)
    np.testing.assert_allclose(predictions.sum(axis=2), np.ones((2, 2)))
    weight_gradients, bias_gradients = model.gradients(
        inputs, epsilon=sample
    )
    analytical = weight_gradients + bias_gradients
    step = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            plus = model.loss(inputs, epsilon=sample)
            parameter[index] = original - step
            minus = model.loss(inputs, epsilon=sample)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-6)


def test_kl_updates_log_var_even_when_epsilon_is_zero():
    model = _small_vae(kl_weight=0.2)
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    sample = np.zeros((len(inputs), 1))
    weight_gradients, bias_gradients = model.gradients(inputs, epsilon=sample)
    log_var_weight_index = len(model.encoder_weights) + 1
    log_var_bias_index = len(model.encoder_biases) + 1

    assert np.linalg.norm(weight_gradients[log_var_weight_index]) > 0.0
    assert np.linalg.norm(bias_gradients[log_var_bias_index]) > 0.0


def test_variational_training_records_all_loss_components_reproducibly():
    inputs = toy_bars()

    def run():
        model = VariationalAutoencoder(
            [9, 8, 2, 8, 9],
            latent_layer=2,
            activations=["tanh", "linear", "tanh", "logistic"],
            seed=0,
            kl_weight=0.01,
        )
        return fit_variational_autoencoder(
            model,
            inputs,
            optimizer=Adam(learning_rate=0.01),
            max_epochs=300,
            generator=np.random.default_rng(7),
        )

    first = run()
    second = run()

    assert first == second
    assert set(first[-1]) >= {
        "total_loss", "reconstruction_loss", "kl_loss", "kl_weight"
    }
    assert all(row["kl_loss"] >= 0.0 for row in first)
    assert np.mean([row["reconstruction_loss"] for row in first[-20:]]) < np.mean(
        [row["reconstruction_loss"] for row in first[:20]]
    )


def test_saved_vae_preserves_distribution_decoder_and_kl_weight(tmp_path):
    model = _small_vae(seed=8, kl_weight=0.2)
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    latent = np.array([[0.4], [-0.2]])
    expected_mu, expected_log_var = model.encode_distribution(inputs)
    expected_decoded = model.decode(latent)
    path = tmp_path / "vae.npz"

    model.save(path)
    loaded = VariationalAutoencoder.load(path)

    mu, log_var = loaded.encode_distribution(inputs)
    assert loaded.kl_weight == 0.2
    np.testing.assert_array_equal(mu, expected_mu)
    np.testing.assert_array_equal(log_var, expected_log_var)
    np.testing.assert_array_equal(loaded.decode(latent), expected_decoded)


def test_saved_categorical_vae_preserves_groups_and_decoder(tmp_path):
    model = VariationalAutoencoder(
        [6, 4, 2, 4, 6],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "grouped_softmax"],
        loss="categorical_cross_entropy",
        categorical_groups=2,
        seed=6,
        kl_weight=0.1,
    )
    latent = np.array([[0.2, -0.4]])
    path = tmp_path / "categorical-vae.npz"

    model.save(path)
    loaded = VariationalAutoencoder.load(path)

    assert loaded.categorical_groups == 2
    assert loaded.loss_name == "categorical_cross_entropy"
    np.testing.assert_array_equal(loaded.decode(latent), model.decode(latent))

