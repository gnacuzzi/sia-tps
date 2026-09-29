"""Cargar los datasets reales y mantener separados training y test."""

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .preprocessing import Standardizer


FRAUD_TARGET = "big_model_fraud_probability"
FRAUD_LABEL = "flagged_fraud"
FRAUD_FEATURES = (
    "timestamp",
    "amount_usd",
    "quantity_purchased",
    "session_duration_seconds",
    "days_since_last_purchase",
    "account_age_days",
    "device_screen_resolution",
    "time_since_last_login_s",
    "items_viewed_before_purchase",
)


@dataclass(frozen=True)
class FraudTrainTest:
    """Fraude estandarizado; ``flagged_fraud`` queda fuera de lo entrenable."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    flagged_fraud_test: np.ndarray
    feature_names: Tuple[str, ...]
    standardizer: Standardizer


@dataclass(frozen=True)
class DigitTrainTest:
    """Imágenes planas y etiquetas enteras, todavía sin codificar ni reescalar."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _test_count(sample_count: int, test_fraction: float) -> int:
    if sample_count < 2:
        raise ValueError("se necesitan al menos dos muestras para separar training y test")
    if not np.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction debe estar entre 0 y 1")
    return min(sample_count - 1, max(1, int(round(sample_count * test_fraction))))


def _stratified_indices(labels: np.ndarray, test_fraction: float,
                        seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Separar índices preservando aproximadamente la proporción de cada clase."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed debe ser un entero no negativo")
    labels = np.asarray(labels)
    wanted_test = _test_count(len(labels), test_fraction)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or np.any(counts < 2):
        raise ValueError("cada clase necesita al menos dos muestras para estratificar")

    exact = counts * wanted_test / len(labels)
    per_class = np.floor(exact).astype(int)
    per_class = np.maximum(per_class, 1)
    per_class = np.minimum(per_class, counts - 1)

    while per_class.sum() < wanted_test:
        candidates = np.where(per_class < counts - 1)[0]
        if not len(candidates):
            break
        choice = candidates[np.argmax(exact[candidates] - per_class[candidates])]
        per_class[choice] += 1
    while per_class.sum() > wanted_test:
        candidates = np.where(per_class > 1)[0]
        if not len(candidates):
            break
        choice = candidates[np.argmax(per_class[candidates] - exact[candidates])]
        per_class[choice] -= 1
    if per_class.sum() != wanted_test:
        raise ValueError("test_fraction incompatible con la estratificación solicitada")

    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for label, class_test_count in zip(classes, per_class):
        indices = rng.permutation(np.flatnonzero(labels == label))
        test_parts.append(indices[:class_test_count])
        train_parts.append(indices[class_test_count:])
    return (rng.permutation(np.concatenate(train_parts)),
            rng.permutation(np.concatenate(test_parts)))


def load_fraud_train_test(path, *, test_fraction: float = 0.2,
                          seed: int = 0) -> FraudTrainTest:
    """Cargar, separar y estandarizar fraude sin usar estadísticas de test.

    El objetivo de entrenamiento es la probabilidad producida por BigModel.
    ``flagged_fraud`` interviene solamente para conservar su proporción al separar
    los datos y se expone únicamente para la evaluación final de test. La media
    y el desvío se calculan con training y luego se aplican a ambos conjuntos.
    """
    path = Path(path)
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        expected = set(FRAUD_FEATURES) | {FRAUD_TARGET, FRAUD_LABEL}
        if reader.fieldnames is None or set(reader.fieldnames) != expected:
            raise ValueError("columnas inesperadas en el dataset de fraude")
        rows = list(reader)
    if not rows:
        raise ValueError("el dataset de fraude está vacío")

    try:
        X = np.asarray([[float(row[name]) for name in FRAUD_FEATURES]
                        for row in rows], dtype=np.float64)
        y = np.asarray([float(row[FRAUD_TARGET]) for row in rows],
                       dtype=np.float64).reshape(-1, 1)
        flagged = np.asarray([int(row[FRAUD_LABEL]) for row in rows], dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("el dataset de fraude contiene valores no numéricos") from error
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("el dataset de fraude contiene valores no finitos")
    if not np.logical_and(y >= 0, y <= 1).all():
        raise ValueError("las probabilidades de BigModel deben estar entre 0 y 1")
    if not np.isin(flagged, [0, 1]).all():
        raise ValueError("flagged_fraud debe contener solamente 0 y 1")

    train_indices, test_indices = _stratified_indices(flagged, test_fraction, seed)
    X_train, X_test = X[train_indices], X[test_indices]
    standardizer = Standardizer.fit(X_train, FRAUD_FEATURES)
    return FraudTrainTest(
        X_train=standardizer.transform(X_train),
        y_train=y[train_indices],
        X_test=standardizer.transform(X_test),
        y_test=y[test_indices],
        flagged_fraud_test=flagged[test_indices],
        feature_names=FRAUD_FEATURES,
        standardizer=standardizer,
    )


def _load_digit_file(path) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    images, labels = [], []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or set(reader.fieldnames) != {"label", "image"}:
            raise ValueError(f"columnas inesperadas en {path.name}")
        for row_number, row in enumerate(reader, start=2):
            try:
                label = int(row["label"])
                image = np.asarray(ast.literal_eval(row["image"]), dtype=np.float32)
            except (SyntaxError, TypeError, ValueError) as error:
                raise ValueError(f"fila {row_number} inválida en {path.name}") from error
            if label not in range(10):
                raise ValueError(f"etiqueta fuera de 0..9 en la fila {row_number} de {path.name}")
            if image.shape != (784,) or not np.isfinite(image).all():
                raise ValueError(f"imagen inválida en la fila {row_number} de {path.name}")
            images.append(image)
            labels.append(label)
    if not images:
        raise ValueError(f"{path.name} está vacío")
    return np.stack(images), np.asarray(labels, dtype=np.int64)


def load_digits_train_test(train_path, test_path, *,
                           additional_train_path: Optional[Path] = None) -> DigitTrainTest:
    """Cargar dígitos usando un archivo externo y reservado como test.

    ``additional_train_path`` permite sumar ``more_digits.csv`` al training del
    ejercicio 3 sin modificar ni mezclar el archivo de test.
    """
    X_train, y_train = _load_digit_file(train_path)
    if additional_train_path is not None:
        additional_X, additional_y = _load_digit_file(additional_train_path)
        X_train = np.concatenate([X_train, additional_X])
        y_train = np.concatenate([y_train, additional_y])
    X_test, y_test = _load_digit_file(test_path)
    return DigitTrainTest(X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test)
