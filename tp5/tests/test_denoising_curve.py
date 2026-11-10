import json

from sia_tp5 import Autoencoder
from sia_tp5.denoising_curve import run_noise_curve


def test_noise_curve_evaluates_frozen_model_at_each_level(tmp_path):
    model = Autoencoder(
        [35, 2, 35],
        latent_layer=1,
        activations=["linear", "logistic"],
        loss="bce",
        seed=0,
    )
    model_path = tmp_path / "model.npz"
    model.save(model_path)
    configuration = {
        "name": "curve-smoke-test",
        "model_path": str(model_path),
        "noise_probabilities": [0.0, 0.1],
        "evaluation_repetitions": 2,
        "evaluation_seed": 4,
    }

    output, summaries = run_noise_curve(configuration, tmp_path)

    assert [row["noise_probability"] for row in summaries] == [0.0, 0.1]
    assert all(row["evaluated_samples"] == 64 for row in summaries)
    assert json.loads((output / "config.json").read_text()) == configuration
    assert (output / "summary.csv").is_file()
    assert (output / "evaluation.csv").is_file()
    assert (output / "noise-curve.png").is_file()
