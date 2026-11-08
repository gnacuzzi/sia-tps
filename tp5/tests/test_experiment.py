import json

from sia_tp5.experiment import run_experiment


def test_experiment_preserves_configuration_and_outputs(tmp_path):
    configuration = {
        "name": "smoke-test",
        "architecture": [35, 4, 2, 4, 35],
        "latent_layer": 2,
        "activations": ["tanh", "linear", "tanh", "logistic"],
        "loss": "mse",
        "optimizer": {"name": "gradient_descent", "learning_rate": 0.001},
        "max_epochs": 1,
        "target_max_pixel_errors": 0,
        "seed": 7,
    }

    output, summary = run_experiment(configuration, tmp_path)

    assert json.loads((output / "config.json").read_text()) == configuration
    assert json.loads((output / "metrics.json").read_text())["epochs"] == 1
    assert summary["epochs"] == 1
    assert (output / "history.csv").is_file()
    assert (output / "model.npz").is_file()
    assert (output / "loss.png").is_file()
    assert (output / "reconstructions.png").is_file()
    assert (output / "latent.csv").is_file()
    assert (output / "latent.png").is_file()


def test_experiment_can_disable_early_stopping(tmp_path):
    configuration = {
        "name": "fixed-budget-test",
        "architecture": [35, 4, 2, 4, 35],
        "latent_layer": 2,
        "activations": ["tanh", "linear", "tanh", "logistic"],
        "loss": "mse",
        "optimizer": {"name": "gradient_descent", "learning_rate": 0.001},
        "max_epochs": 2,
        "requirement_max_pixel_errors": 35,
        "early_stopping": False,
        "seed": 7,
    }

    _, summary = run_experiment(configuration, tmp_path)

    assert summary["epochs"] == 2


def test_experiment_accepts_bce(tmp_path):
    configuration = {
        "name": "bce-smoke-test",
        "architecture": [35, 2, 35],
        "latent_layer": 1,
        "activations": ["linear", "logistic"],
        "loss": "bce",
        "optimizer": {"name": "gradient_descent", "learning_rate": 0.001},
        "max_epochs": 1,
        "target_max_pixel_errors": 0,
        "seed": 7,
    }

    _, summary = run_experiment(configuration, tmp_path)

    assert summary["epochs"] == 1
