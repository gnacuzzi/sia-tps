import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp3.digit_experiment import (load_digit_search_config, one_hot,
                                      parameter_count, run_digit_search)


CONFIG = (Path(__file__).resolve().parents[1] / "docs" / "ejercicio2"
          / "01-learning-rate.json")


def _small_config():
    return {
        "protocol": "search",
        "validation_fraction": 0.5,
        "validation_seed": 0,
        "checkpoints": [0, 1],
        "runs": [{
            "name": "sanity",
            "stage": "sanity",
            "architecture": [784, 2, 10],
            "activations": ["tanh", "logistic"],
            "beta": 1.0,
            "init_scale": 0.1,
            "optimizer": {"name": "gradient_descent", "learning_rate": 0.01},
            "batch_size": 2,
            "max_epochs": 1,
            "model_seed": 0,
            "shuffle_seed": 0,
            "shuffle": True,
        }],
    }


def _write_digits(path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "image"])
        writer.writeheader()
        for label in range(3):
            for copy in range(2):
                image = np.zeros(784)
                image[label] = 1.0
                image[3] = copy
                writer.writerow({"label": label, "image": image.tolist()})


def test_one_hot_and_parameter_count():
    encoded = one_hot(np.array([0, 3, 9]))
    assert encoded.shape == (3, 10)
    np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(3))
    assert np.argmax(encoded, axis=1).tolist() == [0, 3, 9]
    assert parameter_count([784, 32, 10]) == 25450
    assert parameter_count([784, 32, 16, 10]) == 25818


def test_initial_search_config_has_controlled_learning_rate_runs():
    config = load_digit_search_config(CONFIG)
    assert config["protocol"] == "search"
    assert [run["optimizer"]["learning_rate"] for run in config["runs"]] == [
        0.001, 0.01, 0.1]
    assert len({tuple(run["architecture"]) for run in config["runs"]}) == 1
    assert len({run["batch_size"] for run in config["runs"]}) == 1


def test_config_rejects_non_logistic_output(tmp_path):
    config = _small_config()
    config["runs"][0]["activations"][-1] = "linear"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="salida logística"):
        load_digit_search_config(path)


def test_config_accepts_unique_repeat_seeds(tmp_path):
    config = _small_config()
    config["repeat_seeds"] = [0, 1, 2, 3, 4]
    path = tmp_path / "repeated.json"
    path.write_text(json.dumps(config))
    assert load_digit_search_config(path)["repeat_seeds"] == [0, 1, 2, 3, 4]


def test_search_uses_only_development_and_saves_epoch_metrics(tmp_path):
    data = tmp_path / "digits.csv"
    _write_digits(data)
    output = tmp_path / "output"
    summary = run_digit_search(_small_config(), data, output)

    assert len(summary) == 1
    assert summary[0]["status"] == "complete"
    source = json.loads((output / "data-source.json").read_text())
    assert source["test_opened"] is False
    assert set(source) == {"development_path", "development_sha256", "test_opened"}

    with (output / "sanity" / "history.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert "training_macro_f1_present" in rows[0]
    assert "validation_macro_f1_present" in rows[0]
    assert (output / "sanity" / "model-best.npz").exists()
    assert (output / "sanity" / "metrics.json").exists()
    with np.load(output / "split-indices.npz") as split:
        assert len(split["train"]) == 3
        assert len(split["validation"]) == 3
