"""Entrenamiento y métricas explícitas para el autoencoder."""

import numpy as np


def reconstruction_metrics(expected, predicted, *, threshold: float = 0.5):
    expected = np.asarray(expected, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if expected.shape != predicted.shape or expected.ndim != 2:
        raise ValueError("expected y predicted deben ser matrices de igual forma")
    binary = (predicted >= threshold).astype(np.float64)
    errors = np.sum(binary != expected, axis=1)
    return {
        "pixel_errors": errors,
        "max_pixel_errors": int(errors.max()),
        "mean_pixel_errors": float(errors.mean()),
        "exact_patterns": int(np.sum(errors == 0)),
        "patterns_with_at_most_one_error": int(np.sum(errors <= 1)),
    }


def fit_autoencoder(model, inputs, *, optimizer, max_epochs: int,
                    target_max_pixel_errors=0, progress=None):
    """Entrenar full-batch y detenerse al cumplir el criterio binario."""
    inputs = np.asarray(inputs, dtype=np.float64)
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs debe ser un entero positivo")
    if (target_max_pixel_errors is not None
            and (isinstance(target_max_pixel_errors, bool)
                 or not isinstance(target_max_pixel_errors, int)
                 or target_max_pixel_errors < 0)):
        raise ValueError("target_max_pixel_errors debe ser None o un entero no negativo")
    history = []
    for epoch in range(max_epochs + 1):
        predictions = model.predict(inputs)
        metrics = reconstruction_metrics(inputs, predictions)
        row = {
            "epoch": epoch,
            "loss": model.loss(inputs),
            "max_pixel_errors": metrics["max_pixel_errors"],
            "mean_pixel_errors": metrics["mean_pixel_errors"],
            "exact_patterns": metrics["exact_patterns"],
        }
        history.append(row)
        if progress is not None:
            progress(row.copy())
        if (target_max_pixel_errors is not None
                and metrics["max_pixel_errors"] <= target_max_pixel_errors):
            break
        if epoch == max_epochs:
            break
        weight_gradients, bias_gradients = model.gradients(inputs)
        optimizer.step(model.parameters, weight_gradients + bias_gradients)
    return history
