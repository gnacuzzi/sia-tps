"""Particiones temporales de validation para los experimentos obligatorios."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .data import (FRAUD_FEATURES, _load_digit_file, _load_fraud_file,
                   _stratified_indices)
from .preprocessing import Standardizer


@dataclass(frozen=True)
class FraudExperimentSplit:
    """Training/validation temporales y test externo para fraude."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    flagged_fraud_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    flagged_fraud_test: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    feature_names: Tuple[str, ...]
    standardizer: Standardizer


@dataclass(frozen=True)
class DigitExperimentSplit:
    """Training/validation temporales y test externo para dígitos."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _temporary_validation_indices(labels, validation_fraction: float,
                                  seed: int):
    """Separar índices internos sin modificar ni consultar el test reservado."""
    return _stratified_indices(
        labels, validation_fraction, seed,
        fraction_name="validation_fraction",
    )


def load_fraud_experiment_split(path, *, test_fraction: float = 0.2,
                                validation_fraction: float = 0.2,
                                test_seed: int = 0,
                                validation_seed: int = 0) -> FraudExperimentSplit:
    """Dividir fraude antes de ajustar el estandarizador.

    ``validation_fraction`` se aplica sobre el training que queda después de
    reservar test. ``flagged_fraud`` sólo estratifica las particiones y queda
    disponible para evaluar umbrales; nunca se incorpora a ``X`` ni a ``y``.
    """
    X, y, flagged = _load_fraud_file(path)
    development_indices, test_indices = _stratified_indices(
        flagged, test_fraction, test_seed)
    inner_train, inner_validation = _temporary_validation_indices(
        flagged[development_indices], validation_fraction, validation_seed)
    train_indices = development_indices[inner_train]
    validation_indices = development_indices[inner_validation]

    standardizer = Standardizer.fit(X[train_indices], FRAUD_FEATURES)
    return FraudExperimentSplit(
        X_train=standardizer.transform(X[train_indices]),
        y_train=y[train_indices],
        X_validation=standardizer.transform(X[validation_indices]),
        y_validation=y[validation_indices],
        flagged_fraud_validation=flagged[validation_indices],
        X_test=standardizer.transform(X[test_indices]),
        y_test=y[test_indices],
        flagged_fraud_test=flagged[test_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        feature_names=FRAUD_FEATURES,
        standardizer=standardizer,
    )


def load_digits_experiment_split(
        train_path, test_path, *, validation_fraction: float = 0.2,
        validation_seed: int = 0,
        additional_train_path: Optional[Path] = None) -> DigitExperimentSplit:
    """Derivar validation de los archivos de desarrollo y preservar test externo."""
    X_development, y_development = _load_digit_file(train_path)
    if additional_train_path is not None:
        additional_X, additional_y = _load_digit_file(additional_train_path)
        X_development = np.concatenate([X_development, additional_X])
        y_development = np.concatenate([y_development, additional_y])

    train_indices, validation_indices = _temporary_validation_indices(
        y_development, validation_fraction, validation_seed)
    X_test, y_test = _load_digit_file(test_path)
    return DigitExperimentSplit(
        X_train=X_development[train_indices],
        y_train=y_development[train_indices],
        X_validation=X_development[validation_indices],
        y_validation=y_development[validation_indices],
        X_test=X_test,
        y_test=y_test,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
