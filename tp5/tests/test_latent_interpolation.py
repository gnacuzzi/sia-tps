import numpy as np

from sia_tp5 import Autoencoder, load_font
from sia_tp5.latent_interpolation import interpolate, plot_interpolation


def test_interpolation_uses_the_requested_latent_segment(tmp_path):
    dataset = load_font()
    model = Autoencoder(
        [35, 4, 2, 4, 35],
        latent_layer=2,
        activations=["tanh", "linear", "tanh", "logistic"],
        seed=3,
    )
    records = interpolate(model, dataset, "a", "x", [0.0, 0.5, 1.0])
    codes = model.encode(dataset.patterns)
    left = codes[dataset.labels.index("a")]
    right = codes[dataset.labels.index("x")]

    assert len(records) == 3
    np.testing.assert_allclose(records[0]["latent"], left)
    np.testing.assert_allclose(records[1]["latent"], (left + right) / 2.0)
    np.testing.assert_allclose(records[2]["latent"], right)
    assert all(len(record["binary"]) == 35 for record in records)
    assert all(record["minimum_hamming_distance"] >= 0 for record in records)

    output = tmp_path / "interpolation.png"
    plot_interpolation(output, records, "a", "x", threshold=0.5)
    assert output.is_file()
