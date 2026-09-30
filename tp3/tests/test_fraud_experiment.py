import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sia_tp3.data import FRAUD_FEATURES, FRAUD_LABEL, FRAUD_TARGET
from sia_tp3.fraud_experiment import (load_fraud_config,
                                      run_fraud_learning)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fraud-learning.json"


def test_fraud_baseline_config_is_valid_and_contains_both_models():
    config = load_fraud_config(CONFIG)
    assert config["protocol"] == "learning"
    assert [model["activation"] for model in config["models"]] == [
        "linear", "logistic"]
    assert config["training"]["optimizer"] == "gradient_descent"


def test_fraud_config_rejects_unknown_fields(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["training"]["learnig_rate"] = 0.1
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="campos de training"):
        load_fraud_config(path)


def test_fraud_config_rejects_step_activation(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["models"][1]["activation"] = "step"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="activación continua"):
        load_fraud_config(path)


def test_learning_protocol_uses_every_sample_and_has_no_validation(tmp_path):
    data = tmp_path / "fraud.csv"
    fields = list(FRAUD_FEATURES) + [FRAUD_TARGET, FRAUD_LABEL]
    with data.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index in range(10):
            row = {name: index + offset for offset, name in enumerate(FRAUD_FEATURES)}
            row.update({FRAUD_TARGET: index / 10, FRAUD_LABEL: index % 2})
            writer.writerow(row)

    config = load_fraud_config(CONFIG)
    config["training"]["max_epochs"] = 2
    config["training"]["batch_size"] = 2
    output = tmp_path / "output"
    summary = run_fraud_learning(config, data, output)

    assert [row["model"] for row in summary] == ["linear", "logistic"]
    assert all(row["validation_mse"] is None for row in summary)
    with np.load(output / "sample-indices.npz") as indices:
        np.testing.assert_array_equal(indices["all"], np.arange(10))
    for model in ["linear", "logistic"]:
        with (output / model / "training-predictions.csv").open() as file:
            assert len(list(csv.DictReader(file))) == 10
        assert not (output / model / "validation-predictions.csv").exists()
    assert not (output / "split-indices.npz").exists()


def test_generalization_requires_only_selected_model(tmp_path):
    config = json.loads(CONFIG.read_text())
    config.update({
        "protocol": "generalization",
        "test_fraction": 0.2,
        "validation_fraction": 0.2,
        "test_seed": 0,
        "validation_seed": 0,
    })
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="exactamente el modelo seleccionado"):
        load_fraud_config(path)
