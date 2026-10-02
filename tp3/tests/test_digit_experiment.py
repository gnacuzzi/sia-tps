import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp3.digit_experiment import (_expanded_runs, load_digit_search_config,
                                      one_hot, parameter_count,
                                      run_digit_search)


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


def test_config_accepts_softmax_with_cross_entropy(tmp_path):
    config = _small_config()
    config["runs"][0]["activations"][-1] = "softmax"
    config["runs"][0]["loss"] = "categorical_cross_entropy"
    path = tmp_path / "softmax.json"
    path.write_text(json.dumps(config))
    loaded = load_digit_search_config(path)
    assert loaded["runs"][0]["activations"][-1] == "softmax"
    assert loaded["runs"][0]["loss"] == "categorical_cross_entropy"


def test_config_accepts_non_negative_l2(tmp_path):
    config = _small_config()
    config["runs"][0]["l2_lambda"] = 0.001
    path = tmp_path / "l2.json"
    path.write_text(json.dumps(config))
    assert load_digit_search_config(path)["runs"][0]["l2_lambda"] == 0.001

    config["runs"][0]["l2_lambda"] = -0.001
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="l2_lambda"):
        load_digit_search_config(path)


def test_config_accepts_translation_augmentation_and_repeats_its_seed(tmp_path):
    config = _small_config()
    config["runs"][0]["augmentation"] = {
        "name": "translation", "max_shift": 1,
        "probability": 0.5, "seed": 17,
    }
    config["repeat_seeds"] = [2, 4]
    path = tmp_path / "augmentation.json"
    path.write_text(json.dumps(config))
    loaded = load_digit_search_config(path)
    expanded = _expanded_runs(loaded)
    assert [run["augmentation"]["seed"] for run in expanded] == [2, 4]
    assert loaded["runs"][0]["augmentation"]["seed"] == 17


def test_config_accepts_combined_translation_and_rotation(tmp_path):
    config = _small_config()
    config["runs"][0]["augmentation"] = {
        "name": "translation_rotation", "max_shift": 1,
        "translation_probability": 0.5, "max_angle_degrees": 4,
        "rotation_probability": 0.5, "seed": 17,
    }
    path = tmp_path / "combined-augmentation.json"
    path.write_text(json.dumps(config))
    loaded = load_digit_search_config(path)
    assert loaded["runs"][0]["augmentation"]["max_angle_degrees"] == 4


@pytest.mark.parametrize("augmentation", [
    {"name": "translation", "max_shift": 0, "probability": 0.5, "seed": 0},
    {"name": "translation", "max_shift": 1, "probability": 2, "seed": 0},
    {"name": "rotation", "max_shift": 1, "probability": 0.5, "seed": 0},
    {"name": "translation_rotation", "max_shift": 1,
     "translation_probability": 0.5, "max_angle_degrees": 46,
     "rotation_probability": 0.5, "seed": 0},
])
def test_config_rejects_invalid_augmentation(tmp_path, augmentation):
    config = _small_config()
    config["runs"][0]["augmentation"] = augmentation
    path = tmp_path / "invalid-augmentation.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_digit_search_config(path)


def test_config_accepts_piecewise_learning_rate_schedule(tmp_path):
    config = _small_config()
    config["runs"][0]["learning_rate_schedule"] = [
        {"start_epoch": 1, "learning_rate": 0.01},
    ]
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps(config))
    loaded = load_digit_search_config(path)
    assert loaded["runs"][0]["learning_rate_schedule"][0]["start_epoch"] == 1


@pytest.mark.parametrize("schedule", [
    [],
    [{"start_epoch": 2, "learning_rate": 0.01}],
    [{"start_epoch": 1, "learning_rate": 0.02}],
    [{"start_epoch": 1, "learning_rate": 0.01},
     {"start_epoch": 1, "learning_rate": 0.001}],
])
def test_config_rejects_invalid_learning_rate_schedule(tmp_path, schedule):
    config = _small_config()
    config["runs"][0]["learning_rate_schedule"] = schedule
    path = tmp_path / "invalid-schedule.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_digit_search_config(path)


def test_config_accepts_cross_validation_fold(tmp_path):
    config = _small_config()
    del config["validation_fraction"]
    del config["validation_seed"]
    config.update(fold_count=5, fold_index=3, fold_seed=17)
    path = tmp_path / "fold.json"
    path.write_text(json.dumps(config))
    loaded = load_digit_search_config(path)
    assert (loaded["fold_count"], loaded["fold_index"], loaded["fold_seed"]) == (5, 3, 17)


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


def test_search_can_combine_and_deduplicate_development_sources(tmp_path):
    data = tmp_path / "digits.csv"
    additional = tmp_path / "more_digits.csv"
    _write_digits(data)
    _write_digits(additional)
    output = tmp_path / "output"

    run_digit_search(
        _small_config(), data, output,
        additional_data_path=additional,
        deduplicate_inputs=True,
    )

    source = json.loads((output / "data-source.json").read_text())
    assert source["additional_development_path"] == str(additional)
    assert source["deduplicate_inputs"] is True
    assert source["test_opened"] is False
    with np.load(output / "split-indices.npz") as split:
        assert len(split["train"]) == 3
        assert len(split["validation"]) == 3
