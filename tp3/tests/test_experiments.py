import csv

import numpy as np
import pytest

from sia_tp3 import (load_digits_development_fold, load_digits_experiment_split,
                     load_fraud_experiment_split)
from sia_tp3.data import FRAUD_FEATURES


FRAUD_FIELDS = [
    *FRAUD_FEATURES,
    "big_model_fraud_probability",
    "flagged_fraud",
]


def _write_fraud(path, sample_count=20):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FRAUD_FIELDS)
        writer.writeheader()
        for index in range(sample_count):
            row = {name: index + offset
                   for offset, name in enumerate(FRAUD_FEATURES)}
            row.update(big_model_fraud_probability=index / sample_count,
                       flagged_fraud=index % 2)
            writer.writerow(row)


def _write_digits(path, labels):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label", "image"])
        writer.writeheader()
        for index, label in enumerate(labels):
            writer.writerow({"label": label,
                             "image": [float(label) + index / 1000] * 784})


def test_fraud_validation_is_reproducible_disjoint_and_stratified(tmp_path):
    path = tmp_path / "fraud.csv"
    _write_fraud(path)
    options = dict(test_fraction=0.2, validation_fraction=0.25,
                   test_seed=7, validation_seed=9)
    first = load_fraud_experiment_split(path, **options)
    second = load_fraud_experiment_split(path, **options)

    assert first.X_train.shape == (12, 9)
    assert first.X_validation.shape == (4, 9)
    assert first.X_test.shape == (4, 9)
    np.testing.assert_array_equal(first.train_indices, second.train_indices)
    np.testing.assert_array_equal(first.validation_indices, second.validation_indices)
    np.testing.assert_array_equal(first.test_indices, second.test_indices)
    assert not set(first.train_indices) & set(first.validation_indices)
    assert not set(first.train_indices) & set(first.test_indices)
    assert not set(first.validation_indices) & set(first.test_indices)
    assert (set(first.train_indices) | set(first.validation_indices)
            | set(first.test_indices)) == set(range(20))
    np.testing.assert_array_equal(np.sort(first.flagged_fraud_validation), [0, 0, 1, 1])
    np.testing.assert_array_equal(np.sort(first.flagged_fraud_test), [0, 0, 1, 1])


def test_fraud_standardizer_uses_only_inner_training(tmp_path):
    path = tmp_path / "fraud.csv"
    _write_fraud(path)
    data = load_fraud_experiment_split(
        path, test_fraction=0.2, validation_fraction=0.25,
        test_seed=7, validation_seed=9)
    raw_X = np.asarray([
        [index + offset for offset in range(len(FRAUD_FEATURES))]
        for index in range(20)
    ], dtype=np.float64)
    expected_mean = raw_X[data.train_indices].mean(axis=0)
    expected_std = raw_X[data.train_indices].std(axis=0, ddof=0)

    np.testing.assert_allclose(data.standardizer.mean, expected_mean)
    np.testing.assert_allclose(data.standardizer.scale, expected_std)
    np.testing.assert_allclose(data.X_train.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(data.X_train.std(axis=0), 1, atol=1e-12)
    np.testing.assert_allclose(
        data.X_validation,
        (raw_X[data.validation_indices] - expected_mean) / expected_std,
    )
    np.testing.assert_allclose(
        data.X_test,
        (raw_X[data.test_indices] - expected_mean) / expected_std,
    )


def test_digits_validation_comes_only_from_development_file(tmp_path):
    train = tmp_path / "digits.csv"
    test = tmp_path / "digits_test.csv"
    labels = np.repeat([0, 1, 2], 4)
    _write_digits(train, labels)
    _write_digits(test, [7, 8, 9])

    data = load_digits_experiment_split(
        train, test, validation_fraction=0.25, validation_seed=4)

    assert data.X_train.shape == (9, 784)
    assert data.X_validation.shape == (3, 784)
    np.testing.assert_array_equal(np.sort(data.y_validation), [0, 1, 2])
    np.testing.assert_array_equal(data.y_test, [7, 8, 9])
    assert not set(data.train_indices) & set(data.validation_indices)
    assert set(data.train_indices) | set(data.validation_indices) == set(range(12))


def test_digits_additional_data_enters_development_not_test(tmp_path):
    train = tmp_path / "digits.csv"
    more = tmp_path / "more_digits.csv"
    test = tmp_path / "digits_test.csv"
    _write_digits(train, np.repeat([0, 1], 4))
    _write_digits(more, np.repeat([2], 4))
    _write_digits(test, [8, 9])

    data = load_digits_experiment_split(
        train, test, additional_train_path=more,
        validation_fraction=0.25, validation_seed=2)

    assert len(data.X_train) + len(data.X_validation) == 12
    assert 2 in data.y_train
    assert 2 in data.y_validation
    np.testing.assert_array_equal(data.y_test, [8, 9])


def test_digit_folds_are_stratified_disjoint_and_exhaustive(tmp_path):
    train = tmp_path / "digits.csv"
    labels = np.repeat([0, 1, 2], 10)
    _write_digits(train, labels)

    folds = [load_digits_development_fold(
        train, fold_count=5, fold_index=index, fold_seed=17)
        for index in range(5)]

    validation_sets = [set(fold.validation_indices) for fold in folds]
    assert set.union(*validation_sets) == set(range(30))
    assert sum(len(indices) for indices in validation_sets) == 30
    for fold in folds:
        assert not set(fold.train_indices) & set(fold.validation_indices)
        np.testing.assert_array_equal(np.bincount(fold.y_validation), [2, 2, 2])
    repeated = load_digits_development_fold(
        train, fold_count=5, fold_index=2, fold_seed=17)
    np.testing.assert_array_equal(
        folds[2].validation_indices, repeated.validation_indices)


@pytest.mark.parametrize("validation_fraction", [0, 1, -0.1, float("nan")])
def test_validation_rejects_invalid_fraction(tmp_path, validation_fraction):
    train = tmp_path / "digits.csv"
    test = tmp_path / "digits_test.csv"
    _write_digits(train, np.repeat([0, 1], 3))
    _write_digits(test, [0, 1])
    with pytest.raises(ValueError, match="validation_fraction"):
        load_digits_experiment_split(
            train, test, validation_fraction=validation_fraction)
