import json

from sia_tp5.denoising_experiment import run_denoising_experiment


def test_denoising_experiment_preserves_config_and_evaluates_new_noise(tmp_path):
    configuration = {
        "name": "dae-smoke-test",
        "architecture": [35, 3, 2, 3, 35],
        "latent_layer": 2,
        "activations": ["tanh", "linear", "tanh", "logistic"],
        "loss": "bce",
        "optimizer": {"name": "gradient_descent", "learning_rate": 0.01},
        "max_epochs": 1,
        "model_seed": 0,
        "training_noise_probability": 0.1,
        "corruptions_per_sample": 2,
        "training_noise_seed": 10,
        "evaluation_noise_probability": 0.2,
        "evaluation_noise_seed": 20,
        "evaluation_repetitions": 2,
    }

    output, summary = run_denoising_experiment(configuration, tmp_path)

    assert json.loads((output / "config.json").read_text()) == configuration
    assert summary["epochs"] == 1
    assert summary["corruptions_per_sample"] == 2
    assert summary["include_clean_inputs"] is False
    assert summary["training_noise_probabilities"] is None
    assert summary["evaluation"]["evaluated_samples"] == 64
    assert summary["evaluation"]["noise_probability"] == 0.2
    assert (output / "history.csv").is_file()
    assert (output / "evaluation.csv").is_file()
    assert (output / "model.npz").is_file()
