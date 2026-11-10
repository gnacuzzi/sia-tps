import numpy as np

from sia_tp5 import (Adam, Autoencoder, fit_autoencoder,
                     fit_denoising_autoencoder,
                     fit_mixed_denoising_autoencoder,
                     fit_multilevel_denoising_autoencoder, load_font)


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
    targets = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    weight_gradients, bias_gradients = model.gradients(inputs, targets)
    analytical = weight_gradients + bias_gradients
    epsilon = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss(inputs, targets)
            parameter[index] = original - epsilon
            minus = model.loss(inputs, targets)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-6)


def test_backpropagation_with_separate_targets_matches_a_numerical_gradient():
    model = _small_autoencoder()
    inputs = np.array([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    targets = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    weight_gradients, bias_gradients = model.gradients(inputs, targets)
    analytical = weight_gradients + bias_gradients
    epsilon = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss(inputs, targets)
            parameter[index] = original - epsilon
            minus = model.loss(inputs, targets)
            parameter[index] = original
            numerical[index] = (plus - minus) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, rtol=2e-4, atol=2e-6)


def test_leaky_relu_backpropagation_matches_a_numerical_gradient():
    model = Autoencoder(
        [3, 3, 1, 3],
        latent_layer=2,
        activations=["leaky_relu", "linear", "logistic"],
        seed=5,
        init_scale=0.2,
        leaky_slope=0.01,
    )
    inputs = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
    targets = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    weight_gradients, bias_gradients = model.gradients(inputs, targets)
    analytical = weight_gradients + bias_gradients
    epsilon = 1e-6

    for parameter, gradient in zip(model.parameters, analytical):
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            plus = model.loss(inputs, targets)
            parameter[index] = original - epsilon
            minus = model.loss(inputs, targets)
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


def test_fit_autoencoder_measures_predictions_against_separate_targets():
    clean = np.array([[0.0, 1.0, 1.0]])
    noisy = np.array([[1.0, 1.0, 1.0]])
    model = _small_autoencoder(init_scale=0.0)

    history = fit_autoencoder(
        model,
        noisy,
        clean,
        optimizer=Adam(learning_rate=0.01),
        max_epochs=1,
        target_max_pixel_errors=None,
    )

    expected_errors = np.sum((model.predict(noisy) >= 0.5) != clean)
    assert history[-1]["max_pixel_errors"] == expected_errors
    assert history[0]["learning_rate"] == 0.01


def test_denoising_fit_uses_fresh_inputs_and_clean_targets_each_epoch():
    clean = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    model = _small_autoencoder(seed=3)
    observed_updates = []
    original_gradients = model.gradients

    def record_gradients(inputs, targets=None):
        observed_updates.append((inputs.copy(), targets.copy()))
        return original_gradients(inputs, targets)

    model.gradients = record_gradients
    fit_denoising_autoencoder(
        model,
        clean,
        optimizer=Adam(learning_rate=0.01),
        max_epochs=3,
        noise_probability=0.5,
        generator=np.random.default_rng(9),
    )

    assert len(observed_updates) == 3
    assert any(
        not np.array_equal(first[0], second[0])
        for first, second in zip(observed_updates, observed_updates[1:])
    )
    for noisy_inputs, targets in observed_updates:
        assert not np.shares_memory(noisy_inputs, clean)
        np.testing.assert_array_equal(targets, clean)


def test_denoising_fit_repeats_the_history_with_the_same_seeds():
    clean = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])

    def run():
        model = _small_autoencoder(seed=6)
        return fit_denoising_autoencoder(
            model,
            clean,
            optimizer=Adam(learning_rate=0.01),
            max_epochs=4,
            noise_probability=0.35,
            generator=np.random.default_rng(12),
        )

    assert run() == run()


def test_denoising_fit_can_use_multiple_corruptions_per_sample():
    clean = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    model = _small_autoencoder(seed=3)
    observed_updates = []
    original_gradients = model.gradients

    def record_gradients(inputs, targets=None):
        observed_updates.append((inputs.copy(), targets.copy()))
        return original_gradients(inputs, targets)

    model.gradients = record_gradients
    fit_denoising_autoencoder(
        model,
        clean,
        optimizer=Adam(learning_rate=0.01),
        max_epochs=1,
        noise_probability=0.5,
        generator=np.random.default_rng(9),
        corruptions_per_sample=4,
    )

    noisy_inputs, targets = observed_updates[0]
    assert noisy_inputs.shape == (8, 3)
    assert len({tuple(row) for row in noisy_inputs[:4]}) > 1
    np.testing.assert_array_equal(targets, np.repeat(clean, 4, axis=0))


def test_denoising_fit_rejects_invalid_corruption_count():
    with np.testing.assert_raises(ValueError):
        fit_denoising_autoencoder(
            _small_autoencoder(),
            np.array([[0.0, 1.0, 1.0]]),
            optimizer=Adam(learning_rate=0.01),
            max_epochs=1,
            noise_probability=0.1,
            generator=np.random.default_rng(0),
            corruptions_per_sample=0,
        )


def test_mixed_denoising_fit_combines_clean_and_fresh_noisy_inputs():
    clean = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    model = _small_autoencoder(seed=3)
    observed_updates = []
    original_gradients = model.gradients

    def record_gradients(inputs, targets=None):
        observed_updates.append((inputs.copy(), targets.copy()))
        return original_gradients(inputs, targets)

    model.gradients = record_gradients
    fit_mixed_denoising_autoencoder(
        model,
        clean,
        optimizer=Adam(learning_rate=0.01),
        max_epochs=2,
        noise_probability=1.0,
        generator=np.random.default_rng(9),
    )

    assert len(observed_updates) == 2
    expected_targets = np.concatenate((clean, clean), axis=0)
    for inputs, targets in observed_updates:
        np.testing.assert_array_equal(inputs[:2], clean)
        np.testing.assert_array_equal(inputs[2:], 1.0 - clean)
        np.testing.assert_array_equal(targets, expected_targets)


def test_multilevel_fit_builds_one_clean_block_and_one_per_noise_level():
    clean = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    model = _small_autoencoder(seed=3)
    observed_updates = []
    original_gradients = model.gradients

    def record_gradients(inputs, targets=None):
        observed_updates.append((inputs.copy(), targets.copy()))
        return original_gradients(inputs, targets)

    model.gradients = record_gradients
    fit_multilevel_denoising_autoencoder(
        model,
        clean,
        optimizer=Adam(learning_rate=0.01),
        max_epochs=1,
        noise_probabilities=[0.0, 1.0],
        generator=np.random.default_rng(9),
    )

    inputs, targets = observed_updates[0]
    np.testing.assert_array_equal(inputs[:2], clean)
    np.testing.assert_array_equal(inputs[2:4], clean)
    np.testing.assert_array_equal(inputs[4:], 1.0 - clean)
    np.testing.assert_array_equal(targets, np.concatenate([clean] * 3, axis=0))
