"""Operaciones categoricas agrupadas compartidas por AE y VAE."""

import numpy as np


def validate_categorical_groups(width, groups):
    if (isinstance(groups, bool) or not isinstance(groups, int)
            or groups < 1 or width % groups != 0):
        raise ValueError(
            "categorical_groups debe dividir la dimension de salida"
        )
    return width // groups


def grouped_softmax(logits, groups):
    """Aplicar un softmax estable e independiente en cada grupo."""
    values = np.asarray(logits, dtype=np.float64)
    group_size = validate_categorical_groups(values.shape[1], groups)
    grouped = values.reshape(len(values), groups, group_size)
    shifted = grouped - np.max(grouped, axis=2, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=2, keepdims=True)
    return probabilities.reshape(values.shape)


def validate_categorical_targets(targets, groups):
    """Exigir una distribucion objetivo por grupo y por muestra."""
    values = np.asarray(targets, dtype=np.float64)
    group_size = validate_categorical_groups(values.shape[1], groups)
    grouped = values.reshape(len(values), groups, group_size)
    if (np.any(grouped < 0.0)
            or not np.allclose(grouped.sum(axis=2), 1.0)):
        raise ValueError(
            "cada grupo categorico objetivo debe ser no negativo y sumar 1"
        )


def categorical_cross_entropy(predictions, targets, groups):
    """Media por muestra de la suma de entropias por posicion."""
    validate_categorical_targets(targets, groups)
    group_size = validate_categorical_groups(predictions.shape[1], groups)
    safe = np.clip(predictions, np.finfo(np.float64).tiny, 1.0)
    grouped_predictions = safe.reshape(len(safe), groups, group_size)
    grouped_targets = targets.reshape(len(targets), groups, group_size)
    per_position = -np.sum(
        grouped_targets * np.log(grouped_predictions), axis=2
    )
    return float(np.mean(np.sum(per_position, axis=1)))
