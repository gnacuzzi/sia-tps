"""Particiones temporales de validation para los experimentos obligatorios."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .data import (FRAUD_FEATURES, _load_digit_file, _load_fraud_file,
                   _stratified_indices)
from .digit_development import load_unique_digit_development
from .preprocessing import Standardizer


@dataclass(frozen=True)
class FraudLearningData:
    """Todas las muestras de fraude para comparar capacidad de aprendizaje."""

    X: np.ndarray
    y: np.ndarray
    source_indices: np.ndarray
    feature_names: Tuple[str, ...]
    standardizer: Standardizer


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


@dataclass(frozen=True)
class DigitDevelopmentSplit:
    """Training/validation derivados sin abrir el test externo."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray


def _temporary_validation_indices(labels, validation_fraction: float,
                                  seed: int):
    """Separar índices internos sin modificar ni consultar el test reservado."""
    return _stratified_indices(
        labels, validation_fraction, seed,
        fraction_name="validation_fraction",
    )


def _stratified_fold_indices(labels, fold_count: int, seed: int):
    """Construir folds disjuntos cuya unión contiene todo development."""
    if type(fold_count) is not int or fold_count < 2:
        raise ValueError("fold_count debe ser un entero mayor o igual a dos")
    if type(seed) is not int or seed < 0:
        raise ValueError("fold_seed debe ser un entero no negativo")
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < fold_count):
        raise ValueError("cada clase necesita al menos fold_count muestras")

    rng = np.random.default_rng(seed)
    validation_parts = [[] for _ in range(fold_count)]
    for label in classes:
        shuffled = rng.permutation(np.flatnonzero(labels == label))
        for fold_index, part in enumerate(np.array_split(shuffled, fold_count)):
            validation_parts[fold_index].append(part)

    all_indices = np.arange(len(labels))
    folds = []
    for parts in validation_parts:
        validation = rng.permutation(np.concatenate(parts))
        in_validation = np.zeros(len(labels), dtype=bool)
        in_validation[validation] = True
        training = rng.permutation(all_indices[~in_validation])
        folds.append((training, validation))
    return folds


def load_fraud_learning_data(path) -> FraudLearningData:
    """Estandarizar y devolver las muestras completas para la primera comparación.

    Esta etapa no estudia generalización: lineal y no lineal se entrenan con las
    mismas filas para analizar aprendizaje, underfitting y saturación. Por eso
    el estandarizador también se ajusta con el conjunto completo.
    """
    X, y, _ = _load_fraud_file(path)
    standardizer = Standardizer.fit(X, FRAUD_FEATURES)
    return FraudLearningData(
        X=standardizer.transform(X),
        y=y,
        source_indices=np.arange(len(X)),
        feature_names=FRAUD_FEATURES,
        standardizer=standardizer,
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
        additional_train_path: Optional[Path] = None,
        deduplicate_inputs: bool = False) -> DigitExperimentSplit:
    """Derivar validation de los archivos de desarrollo y preservar test externo."""
    development = load_digits_development_split(
        train_path,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        additional_train_path=additional_train_path,
        deduplicate_inputs=deduplicate_inputs,
    )
    X_test, y_test = _load_digit_file(test_path)
    return DigitExperimentSplit(
        X_train=development.X_train,
        y_train=development.y_train,
        X_validation=development.X_validation,
        y_validation=development.y_validation,
        X_test=X_test,
        y_test=y_test,
        train_indices=development.train_indices,
        validation_indices=development.validation_indices,
    )


def load_digits_development_split(
        train_path, *, validation_fraction: float = 0.2,
        validation_seed: int = 0,
        additional_train_path: Optional[Path] = None,
        deduplicate_inputs: bool = False) -> DigitDevelopmentSplit:
    """Derivar training/validation sin recibir ni abrir un archivo de test."""
    if deduplicate_inputs and additional_train_path is None:
        raise ValueError("deduplicate_inputs requiere additional_train_path")
    if deduplicate_inputs:
        development = load_unique_digit_development(
            train_path, additional_train_path)
        X_development, y_development = development.X, development.y
    else:
        X_development, y_development = _load_digit_file(train_path)
    if additional_train_path is not None and not deduplicate_inputs:
        additional_X, additional_y = _load_digit_file(additional_train_path)
        X_development = np.concatenate([X_development, additional_X])
        y_development = np.concatenate([y_development, additional_y])

    train_indices, validation_indices = _temporary_validation_indices(
        y_development, validation_fraction, validation_seed)
    return DigitDevelopmentSplit(
        X_train=X_development[train_indices],
        y_train=y_development[train_indices],
        X_validation=X_development[validation_indices],
        y_validation=y_development[validation_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
    )


def load_digits_development_fold(
        train_path, *, fold_count: int, fold_index: int, fold_seed: int = 0,
        additional_train_path: Optional[Path] = None,
        deduplicate_inputs: bool = False) -> DigitDevelopmentSplit:
    """Cargar un fold estratificado de development sin abrir test externo."""
    if type(fold_index) is not int or not 0 <= fold_index < fold_count:
        raise ValueError("fold_index debe pertenecer a [0, fold_count)")
    if deduplicate_inputs and additional_train_path is None:
        raise ValueError("deduplicate_inputs requiere additional_train_path")
    if deduplicate_inputs:
        development = load_unique_digit_development(
            train_path, additional_train_path)
        X_development, y_development = development.X, development.y
    else:
        X_development, y_development = _load_digit_file(train_path)
    if additional_train_path is not None and not deduplicate_inputs:
        additional_X, additional_y = _load_digit_file(additional_train_path)
        X_development = np.concatenate([X_development, additional_X])
        y_development = np.concatenate([y_development, additional_y])

    folds = _stratified_fold_indices(y_development, fold_count, fold_seed)
    train_indices, validation_indices = folds[fold_index]
    return DigitDevelopmentSplit(
        X_train=X_development[train_indices],
        y_train=y_development[train_indices],
        X_validation=X_development[validation_indices],
        y_validation=y_development[validation_indices],
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
