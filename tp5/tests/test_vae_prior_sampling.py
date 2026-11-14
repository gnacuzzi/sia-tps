import numpy as np

from sia_tp5 import VariationalAutoencoder
from sia_tp5.vae_prior_sampling import evaluate_prior


def _model():
    return VariationalAutoencoder(
        [9, 5, 2, 5, 9],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=3,
        kl_weight=0.01,
    )


def test_prior_evaluation_is_reproducible_and_summary_is_consistent():
    first = evaluate_prior(_model(), sample_count=25, seed=11)
    second = evaluate_prior(_model(), sample_count=25, seed=11)
    rows, summary, probabilities, binary = first

    assert rows == second[0]
    assert summary == second[1]
    np.testing.assert_array_equal(probabilities, second[2])
    np.testing.assert_array_equal(binary, second[3])
    assert summary["valid_samples"] == sum(
        row["valid_bar"] for row in rows
    )
    assert summary["valid_rate"] == summary["valid_samples"] / 25
    assert sum(summary["valid_counts"].values()) == summary["valid_samples"]
    assert probabilities.shape == (25, 9)
    assert binary.shape == (25, 9)

