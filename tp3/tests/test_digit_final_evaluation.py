import csv
import json

import numpy as np

from sia_tp3.digit_final_evaluation import (load_final_digit_config,
                                            run_final_digit_evaluation)


def _write_digits(path, labels=range(10)):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "image"])
        writer.writeheader()
        for label in labels:
            for copy in range(2):
                image = np.zeros(784)
                image[label] = 1.0
                image[10] = copy
                writer.writerow({"label": label, "image": image.tolist()})


def _config():
    return {
        "protocol": "final",
        "architecture": [784, 2, 10],
        "activations": ["tanh", "logistic"],
        "beta": 1.0,
        "init_scale": 0.1,
        "optimizer": {"name": "adam", "learning_rate": 0.001,
                      "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
        "batch_size": 2,
        "epochs": 1,
        "model_seed": 0,
        "shuffle_seed": 0,
        "shuffle": True,
    }


def test_final_config_and_evaluation_records_single_test_use(tmp_path):
    development = tmp_path / "digits.csv"
    test = tmp_path / "digits_test.csv"
    config_path = tmp_path / "final.json"
    output = tmp_path / "output"
    _write_digits(development, labels=[0, 1, 2, 3, 4, 5, 6, 7, 9])
    _write_digits(test)
    config_path.write_text(json.dumps(_config()))

    config = load_final_digit_config(config_path)
    summary = run_final_digit_evaluation(config, development, test, output)

    assert summary["development_samples"] == 18
    assert summary["test_samples"] == 20
    source = json.loads((output / "data-source.json").read_text())
    assert source["test_opened"] is True
    assert source["test_evaluations"] == 1
    assert (output / "model-final.npz").exists()
    assert (output / "test-predictions.csv").exists()
    assert (output / "test-confusion-matrix.csv").exists()


def test_final_config_accepts_frozen_softmax_augmentation_and_schedule(tmp_path):
    config = _config()
    config.update({
        "activations": ["tanh", "softmax"],
        "loss": "categorical_cross_entropy",
        "l2_lambda": 0.0,
        "augmentation": {
            "name": "translation_rotation", "max_shift": 1,
            "translation_probability": 0.5, "max_angle_degrees": 4,
            "rotation_probability": 0.5, "seed": 0,
        },
        "learning_rate_schedule": [
            {"start_epoch": 1, "learning_rate": 0.001},
        ],
    })
    path = tmp_path / "final-softmax.json"
    path.write_text(json.dumps(config))
    loaded = load_final_digit_config(path)
    assert loaded["loss"] == "categorical_cross_entropy"
    assert loaded["augmentation"]["max_angle_degrees"] == 4


def test_final_softmax_combines_and_deduplicates_development(tmp_path):
    primary = tmp_path / "digits.csv"
    additional = tmp_path / "more_digits.csv"
    test = tmp_path / "digits_test.csv"
    output = tmp_path / "output"
    _write_digits(primary)
    _write_digits(additional)
    _write_digits(test)
    config = _config()
    config.update({
        "activations": ["tanh", "softmax"],
        "loss": "categorical_cross_entropy",
        "l2_lambda": 0.0,
        "augmentation": {
            "name": "translation_rotation", "max_shift": 1,
            "translation_probability": 0, "max_angle_degrees": 4,
            "rotation_probability": 0, "seed": 0,
        },
        "learning_rate_schedule": [
            {"start_epoch": 1, "learning_rate": 0.001},
        ],
    })

    summary = run_final_digit_evaluation(
        config, primary, test, output,
        additional_development_path=additional, deduplicate_inputs=True)

    assert summary["development_samples"] == 20
    assert summary["test_digit_8_f1"] >= 0
    source = json.loads((output / "data-source.json").read_text())
    assert source["deduplicate_inputs"] is True
    assert source["test_evaluations"] == 1
