"""Comprobar estadísticos y aislamiento de training con datos pequeños."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib", reason="el EDA requiere el extra plot")

from sia_tp3.data import FRAUD_FEATURES, FRAUD_LABEL, FRAUD_TARGET, _stratified_indices


spec = importlib.util.spec_from_file_location(
    "analyze_training", Path(__file__).resolve().parents[1] / "scripts/analyze_training.py")
eda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eda)


def test_summary_counts_missing_and_infinite_without_hiding_them():
    result = eda.describe("example", [0, 1, 2, 3, 100, np.nan, np.inf], "units")
    assert result["count"] == 7
    assert result["finite_count"] == 5
    assert result["missing_or_nan"] == result["infinite"] == 1
    assert result["q1"] == 1 and result["q3"] == 3
    assert result["median"] == 2
    assert result["lower_fence"] == -2 and result["upper_fence"] == 6
    assert result["iqr_candidates"] == 1
    assert eda.describe("empty", [np.nan], "units")["mean"] is None


def test_duplicate_counts_distinguish_copies_and_conflicting_targets():
    result = eda.duplicates(np.array([[1, 2], [1, 2], [3, 4], [3, 4], [5, 6]]),
                            np.array([0, 0, 1, 2, 3]))
    assert result == dict(duplicate_input_groups=2, duplicate_input_rows=4,
                          extra_input_copies=2, conflicting_target_groups=1)


def test_fraud_eda_uses_existing_split_without_parsing_test_features(tmp_path):
    labels = np.array([0, 1] * 5)
    training, test = _stratified_indices(labels, 0.2, 0)
    path = tmp_path / "fraud.csv"
    names = list(FRAUD_FEATURES) + [FRAUD_TARGET, FRAUD_LABEL]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        for index, label in enumerate(labels):
            row = {name: index + 1 for name in FRAUD_FEATURES}
            row.update({FRAUD_TARGET: index / 10, FRAUD_LABEL: label})
            if index in test:
                row["amount_usd"] = "NO LEER NI ANALIZAR TEST"
            writer.writerow(row)
    indices, values, observed = eda.fraud_training(path)
    np.testing.assert_array_equal(indices, training)
    np.testing.assert_array_equal(values[:, 0], training + 1)
    np.testing.assert_array_equal(observed, labels[training])
    assert values.shape == (8, 10)  # Nueve entradas y BigModel; sin etiqueta real.


def test_digit_eda_opens_only_training_file(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    with (data / "digits.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "image"])
        writer.writeheader()
        writer.writerow({"label": 0, "image": [0] * 784})
        writer.writerow({"label": 1, "image": [1] * 784})
    # Los otros datasets no existen. Registrar todos los CSV abiertos.
    opened = []
    original_open = Path.open
    def recording_open(path, *args, **kwargs):
        if path.suffix == ".csv" and path.parent == data:
            opened.append(path.name)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr(eda, "save_figure", lambda fig, *args: eda.plt.close(fig))
    output = tmp_path / "output"
    output.mkdir()
    result = eda.analyze_digits(data, output, seed=0)
    assert opened == ["digits.csv"]
    assert result["rows"] == 2
    assert result["statistics"]["missing_or_nan"] == 0
    assert result["blank_images"] == 1
