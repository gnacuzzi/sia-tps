import csv

import numpy as np
import pytest

from sia_tp3 import Standardizer, load_digits_train_test, load_fraud_train_test
from sia_tp3.data import FRAUD_FEATURES, _stratified_indices


FRAUD_FIELDS = [
    "timestamp", "amount_usd", "quantity_purchased",
    "session_duration_seconds", "days_since_last_purchase", "account_age_days",
    "device_screen_resolution", "time_since_last_login_s",
    "items_viewed_before_purchase", "big_model_fraud_probability", "flagged_fraud",
]


def _write_fraud(path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAUD_FIELDS)
        writer.writeheader()
        for index in range(10):
            row = {name: index + offset for offset, name in enumerate(FRAUD_FIELDS[:-2])}
            row.update(big_model_fraud_probability=index / 10,
                       flagged_fraud=index % 2)
            writer.writerow(row)


def _write_digits(path, labels, *, image_length=784):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "image"])
        writer.writeheader()
        for label in labels:
            writer.writerow({"label": label, "image": [float(label)] * image_length})


def test_fraud_loads_reproducible_stratified_train_test(tmp_path):
    path = tmp_path / "fraud.csv"
    _write_fraud(path)
    first = load_fraud_train_test(path, test_fraction=0.2, seed=7)
    second = load_fraud_train_test(path, test_fraction=0.2, seed=7)

    assert first.X_train.shape == (8, 9)
    assert first.y_train.shape == (8, 1)
    assert first.X_test.shape == (2, 9)
    assert first.y_test.shape == (2, 1)
    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    np.testing.assert_allclose(first.X_train.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(first.X_train.std(axis=0), 1, atol=1e-12)
    np.testing.assert_array_equal(np.sort(first.flagged_fraud_test), [0, 1])
    assert "flagged_fraud" not in first.feature_names
    assert "big_model_fraud_probability" not in first.feature_names


def test_fraud_standardization_fits_only_training_and_preserves_targets(tmp_path):
    path = tmp_path / "fraud.csv"
    _write_fraud(path)
    data = load_fraud_train_test(path, test_fraction=0.2, seed=7)

    raw_X = np.asarray([[index + offset for offset in range(len(FRAUD_FEATURES))]
                        for index in range(10)], dtype=np.float64)
    raw_y = np.arange(10, dtype=np.float64).reshape(-1, 1) / 10
    labels = np.arange(10) % 2
    train, test = _stratified_indices(labels, 0.2, 7)
    expected_mean = raw_X[train].mean(axis=0)
    expected_std = raw_X[train].std(axis=0, ddof=0)

    np.testing.assert_allclose(data.standardizer.mean, expected_mean)
    np.testing.assert_allclose(data.standardizer.scale, expected_std)
    np.testing.assert_allclose(data.X_train, (raw_X[train] - expected_mean) / expected_std)
    np.testing.assert_allclose(data.X_test, (raw_X[test] - expected_mean) / expected_std)
    np.testing.assert_array_equal(data.y_train, raw_y[train])
    np.testing.assert_array_equal(data.y_test, raw_y[test])


def test_standardizer_handles_constants_and_can_be_saved(tmp_path):
    scaler = Standardizer.fit([[1, 10], [1, 20]], ["constant", "variable"])
    np.testing.assert_array_equal(scaler.scale, [1, 5])
    np.testing.assert_allclose(scaler.transform([[1, 30]]), [[0, 3]])

    path = tmp_path / "standardizer.npz"
    scaler.save(path)
    restored = Standardizer.load(path)
    np.testing.assert_array_equal(restored.mean, scaler.mean)
    np.testing.assert_array_equal(restored.scale, scaler.scale)
    assert restored.feature_names == scaler.feature_names


@pytest.mark.parametrize("test_fraction", [0, 1, -0.1, float("nan")])
def test_fraud_rejects_invalid_test_fraction(tmp_path, test_fraction):
    path = tmp_path / "fraud.csv"
    _write_fraud(path)
    with pytest.raises(ValueError, match="test_fraction"):
        load_fraud_train_test(path, test_fraction=test_fraction)


def test_digits_keep_external_test_and_add_more_only_to_training(tmp_path):
    train = tmp_path / "digits.csv"
    test = tmp_path / "digits_test.csv"
    more = tmp_path / "more_digits.csv"
    _write_digits(train, [0, 1])
    _write_digits(test, [2, 3, 4])
    _write_digits(more, [5])

    data = load_digits_train_test(train, test, additional_train_path=more)

    assert data.X_train.shape == (3, 784)
    assert data.X_test.shape == (3, 784)
    np.testing.assert_array_equal(data.y_train, [0, 1, 5])
    np.testing.assert_array_equal(data.y_test, [2, 3, 4])
    assert data.X_train.dtype == np.float32


def test_digits_reject_invalid_image_length(tmp_path):
    train = tmp_path / "digits.csv"
    test = tmp_path / "digits_test.csv"
    _write_digits(train, [0], image_length=783)
    _write_digits(test, [0])
    with pytest.raises(ValueError, match="imagen inválida"):
        load_digits_train_test(train, test)
