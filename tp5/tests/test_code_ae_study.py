import numpy as np

from sia_tp5 import Autoencoder
from sia_tp5.code_ae_study import categorical_source_metrics
from sia_tp5.code_dataset import encode_sources


def test_categorical_source_metrics_use_argmax_per_character():
    sources = (
        "def f(x): return (x+ 3)* 7",
        "def f(x): return (x-12)+99",
    )
    patterns = encode_sources(sources)
    model = Autoencoder(
        [650, 4, 650],
        latent_layer=1,
        activations=["linear", "grouped_softmax"],
        loss="categorical_cross_entropy",
        categorical_groups=26,
        seed=0,
    )
    model.predict = lambda inputs: patterns.copy()
    model.loss = lambda inputs: 0.0

    metrics = categorical_source_metrics(model, patterns, sources)

    assert metrics["exact_sources"] == 2
    assert metrics["exact_source_rate"] == 1.0
    assert metrics["character_accuracy"] == 1.0
    assert metrics["compile_rate"] == 1.0
    assert metrics["grammar_rate"] == 1.0
    np.testing.assert_array_equal(metrics["correct_by_position"], np.ones(26))


def test_categorical_source_metrics_distinguish_character_and_source_accuracy():
    sources = ("def f(x): return (x+ 3)* 7",)
    patterns = encode_sources(sources)
    probabilities = patterns.copy().reshape(1, 26, 25)
    correct_index = np.argmax(probabilities[0, -1])
    probabilities[0, -1, correct_index] = 0.0
    probabilities[0, -1, (correct_index + 1) % 25] = 1.0
    probabilities = probabilities.reshape(1, 650)
    model = Autoencoder(
        [650, 4, 650],
        latent_layer=1,
        activations=["linear", "grouped_softmax"],
        loss="categorical_cross_entropy",
        categorical_groups=26,
        seed=0,
    )
    model.predict = lambda inputs: probabilities
    model.loss = lambda inputs: 1.0

    metrics = categorical_source_metrics(model, patterns, sources)

    assert metrics["exact_sources"] == 0
    assert metrics["character_accuracy"] == 25 / 26
    assert metrics["mean_character_errors"] == 1.0
