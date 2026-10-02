"""Entrenar modelos sin conocer el problema que representan los datos."""

from typing import Callable, Optional, Tuple

import numpy as np

from .models import MultilayerPerceptron
from .optimizers import GradientDescent, Optimizer


def fit(model: MultilayerPerceptron, X, y, *, learning_rate: Optional[float] = None,
        max_epochs: int, target_mse: float, shuffle: bool, seed: int,
        optimizer: Optional[Optimizer] = None, batch_size: Optional[int] = 1,
        validation_data: Optional[Tuple[object, object]] = None,
        batch_transform: Optional[Callable] = None,
        batch_transform_seed: Optional[int] = None,
        learning_rate_schedule=None,
        require_bipolar_accuracy: bool = False,
        epoch_metrics: Optional[Callable] = None,
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
    if learning_rate_schedule is None:
        learning_rate_schedule = {}
    if not isinstance(learning_rate_schedule, dict):
        raise TypeError("learning_rate_schedule debe ser un diccionario")
    for start_epoch, scheduled_rate in learning_rate_schedule.items():
        if (isinstance(start_epoch, bool)
                or not isinstance(start_epoch, (int, np.integer))
                or not 1 <= start_epoch <= max_epochs):
            raise ValueError(
                "las épocas del learning_rate_schedule deben pertenecer a [1, max_epochs]")
        if not np.isfinite(scheduled_rate) or scheduled_rate <= 0:
            raise ValueError("cada learning rate programado debe ser positivo y finito")
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
    if batch_transform is not None and not callable(batch_transform):
        raise TypeError("batch_transform debe ser invocable")
    if batch_transform is None and batch_transform_seed is not None:
        raise ValueError("batch_transform_seed requiere batch_transform")
    if batch_transform is not None and (
            isinstance(batch_transform_seed, bool)
            or not isinstance(batch_transform_seed, (int, np.integer))
            or batch_transform_seed < 0):
        raise ValueError("batch_transform_seed debe ser un entero no negativo")
    rng = np.random.default_rng(seed)
    transform_rng = (None if batch_transform is None
                     else np.random.default_rng(batch_transform_seed))
    history = []
    for epoch in range(max_epochs + 1):
        if epoch:
            if epoch in learning_rate_schedule:
                optimizer.learning_rate = float(learning_rate_schedule[epoch])
            order = rng.permutation(len(X)) if shuffle else np.arange(len(X))
            for start in range(0, len(X), batch_size):
                indices = order[start:start + batch_size]
                batch_X, batch_y = X[indices], y[indices]
                if batch_transform is not None:
                    transformed = np.asarray(batch_transform(batch_X, transform_rng),
                                             dtype=np.float64)
                    if transformed.shape != batch_X.shape:
                        raise ValueError(
                            "batch_transform debe conservar la forma del batch")
                    if not np.isfinite(transformed).all():
                        raise ValueError(
                            "batch_transform debe devolver valores finitos")
                    batch_X = transformed
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
        training_data_loss = model.data_loss(X, y)
        regularization_loss = model.regularization_loss()
        training_loss = training_data_loss + regularization_loss
        if not np.isfinite(mse):
            raise FloatingPointError("entrenamiento divergente: MSE no finito")
        validation_mse = None
        validation_loss = None
        validation_predictions = None
        validation_y = None
        if validation is not None:
            validation_X, validation_y = validation
            validation_predictions = model.predict(validation_X)
            if not np.isfinite(validation_predictions).all():
                raise FloatingPointError("entrenamiento divergente: validation no finita")
            validation_mse = float(np.mean((validation_predictions - validation_y) ** 2))
            # Validation mide sólo ajuste a datos; la penalización depende de
            # los parámetros, no de las muestras de validation.
            validation_loss = model.data_loss(validation_X, validation_y)
            if not np.isfinite(validation_mse):
                raise FloatingPointError("entrenamiento divergente: MSE de validation no finito")
        accuracy = (float(np.mean(np.all(np.where(predictions >= 0, 1, -1) == y, axis=1)))
                    if require_bipolar_accuracy else None)
        converged = mse <= target_mse and (accuracy is None or accuracy == 1.0)
        row = {"epoch": epoch, "learning_rate": optimizer.learning_rate,
               "loss": training_loss,
               "data_loss": training_data_loss,
               "regularization_loss": regularization_loss,
               "validation_loss": validation_loss,
               "mse": mse, "validation_mse": validation_mse,
               "accuracy": accuracy, "converged": converged}
        if epoch_metrics is not None:
            extra_metrics = epoch_metrics(
                y, predictions, validation_y, validation_predictions)
            if not isinstance(extra_metrics, dict):
                raise TypeError("epoch_metrics debe devolver un diccionario")
            repeated = set(row) & set(extra_metrics)
            if repeated:
                raise ValueError(
                    f"epoch_metrics intentó sobrescribir métricas: {sorted(repeated)}")
            row.update(extra_metrics)
        history.append(row)
        if progress:
            progress(row.copy())
        optimizer.epoch_end(mse if model.loss_name == "mse" else training_loss)
        if converged:
            break
    return history
