"""Entrenar modelos sin conocer el problema que representan los datos."""

from typing import Callable, Optional, Tuple

import numpy as np

from .models import MultilayerPerceptron
from .optimizers import GradientDescent, Optimizer


def fit(model: MultilayerPerceptron, X, y, *, learning_rate: Optional[float] = None,
        max_epochs: int, target_mse: float, shuffle: bool, seed: int,
        optimizer: Optional[Optimizer] = None, batch_size: Optional[int] = 1,
        validation_data: Optional[Tuple[object, object]] = None,
        require_bipolar_accuracy: bool = False,
        progress: Optional[Callable[[dict], None]] = None):
    """Entrenar por lotes y medir training/validation al final de cada época."""
    X, y = model.validate_data(X, y)
    if validation_data is not None and (
            not isinstance(validation_data, (tuple, list)) or len(validation_data) != 2):
        raise ValueError("validation_data debe contener (X_validation, y_validation)")
    validation = (model.validate_data(*validation_data)
                  if validation_data is not None else None)
    if optimizer is None:
        if learning_rate is None:
            raise ValueError("se requiere learning_rate u optimizer")
        optimizer = GradientDescent(learning_rate)
    elif not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer debe ser una instancia de Optimizer")
    elif learning_rate is not None:
        raise ValueError("learning_rate pertenece al optimizer; no indicar ambos")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs debe ser un entero positivo")
    if not np.isfinite(target_mse) or target_mse < 0:
        raise ValueError("target_mse debe ser no negativo y finito")
    if not isinstance(shuffle, bool):
        raise ValueError("shuffle debe ser booleano")
    if batch_size is None:
        batch_size = len(X)
    if (isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer))
            or not 1 <= batch_size <= len(X)):
        raise ValueError("batch_size debe ser un entero entre 1 y la cantidad de muestras")
    batch_size = int(batch_size)
    if require_bipolar_accuracy and not np.isin(y, [-1, 1]).all():
        raise ValueError("accuracy bipolar requiere objetivos -1/+1")
    uses_step = model.activations[-1] == "step"
    if uses_step and (type(optimizer) is not GradientDescent or batch_size != 1):
        raise ValueError("Rosenblatt escalón sólo admite descenso básico online")
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(max_epochs + 1):
        if epoch:
            order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
            for start in range(0, len(X), batch_size):
                indices = order[start:start + batch_size]
                batch_X, batch_y = X[indices], y[indices]
                if uses_step:
                    model._update(batch_X, batch_y, optimizer.learning_rate)
                else:
                    dw, db = model._gradients(batch_X, batch_y)
                    optimizer.step(model.weights + model.biases, dw + db)
        predictions = model.predict(X)
        if not np.isfinite(predictions).all() or any(
                not np.isfinite(a).all() for a in model.weights + model.biases):
            raise FloatingPointError("entrenamiento divergente: parámetros o predicciones no finitos")
        mse = float(np.mean((predictions - y) ** 2))
        if not np.isfinite(mse):
            raise FloatingPointError("entrenamiento divergente: MSE no finito")
        validation_mse = None
        if validation is not None:
            validation_X, validation_y = validation
            validation_predictions = model.predict(validation_X)
            if not np.isfinite(validation_predictions).all():
                raise FloatingPointError("entrenamiento divergente: validation no finita")
            validation_mse = float(np.mean((validation_predictions - validation_y) ** 2))
            if not np.isfinite(validation_mse):
                raise FloatingPointError("entrenamiento divergente: MSE de validation no finito")
        accuracy = (float(np.mean(np.all(np.where(predictions >= 0, 1, -1) == y, axis=1)))
                    if require_bipolar_accuracy else None)
        converged = mse <= target_mse and (accuracy is None or accuracy == 1.0)
        row = {"epoch": epoch, "mse": mse, "validation_mse": validation_mse,
               "accuracy": accuracy, "converged": converged}
        history.append(row)
        if progress:
            progress(row.copy())
        optimizer.epoch_end(mse)
        if converged:
            break
    return history
