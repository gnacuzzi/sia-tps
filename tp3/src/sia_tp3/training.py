"""Entrenar modelos sin conocer el problema que representan los datos."""

from typing import Callable, Optional

import numpy as np

from .models import MultilayerPerceptron


def fit(model: MultilayerPerceptron, X, y, *, learning_rate: float,
        max_epochs: int, target_mse: float, shuffle: bool, seed: int,
        require_bipolar_accuracy: bool = False,
        progress: Optional[Callable[[dict], None]] = None):
    """Entrenar online y registrar métricas con pesos fijos al final de cada época."""
    X, y = model.validate_data(X, y)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo y finito")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs debe ser un entero positivo")
    if not np.isfinite(target_mse) or target_mse < 0:
        raise ValueError("target_mse debe ser no negativo y finito")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle debe ser booleano")
    if require_bipolar_accuracy and not np.isin(y, [-1, 1]).all():
        raise ValueError("accuracy bipolar requiere objetivos -1/+1")
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(max_epochs + 1):
        if epoch:
            order = rng.permutation(len(X)) if shuffle else range(len(X))
            for i in order:
                model._update(X[i:i + 1], y[i:i + 1], learning_rate)
        predictions = model.predict(X)
        if not np.isfinite(predictions).all() or any(
                not np.isfinite(a).all() for a in model.weights + model.biases):
            raise FloatingPointError("entrenamiento divergente: parámetros o predicciones no finitos")
        mse = float(np.mean((predictions - y) ** 2))
        if not np.isfinite(mse):
            raise FloatingPointError("entrenamiento divergente: MSE no finito")
        accuracy = (float(np.mean(np.all(np.where(predictions >= 0, 1, -1) == y, axis=1)))
                    if require_bipolar_accuracy else None)
        converged = mse <= target_mse and (accuracy is None or accuracy == 1.0)
        row = {"epoch": epoch, "mse": mse, "accuracy": accuracy, "converged": converged}
        history.append(row)
        if progress:
            progress(row.copy())
        if converged:
            break
    return history
