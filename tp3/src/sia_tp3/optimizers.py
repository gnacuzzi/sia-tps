"""Optimizadores presentados en la clase 12.1."""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


def _positive_finite(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} debe ser positivo y finito")
    return float(value)


def _fraction(value, name):
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError(f"{name} debe pertenecer a [0, 1)")
    return float(value)


class Optimizer(ABC):
    """Definir cómo convertir gradientes en cambios de parámetros."""

    def __init__(self, learning_rate: float):
        self.learning_rate = _positive_finite(learning_rate, "learning_rate")
        self._initial_learning_rate = self.learning_rate

    def _pairs(self, parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]):
        if len(parameters) == 0 or len(parameters) != len(gradients):
            raise ValueError("parameters y gradients deben tener la misma longitud no vacía")
        pairs = []
        for parameter, gradient in zip(parameters, gradients):
            if not isinstance(parameter, np.ndarray):
                raise TypeError("cada parámetro debe ser un ndarray de NumPy")
            gradient = np.asarray(gradient, dtype=np.float64)
            if parameter.shape != gradient.shape:
                raise ValueError("cada gradiente debe tener la forma de su parámetro")
            if not np.isfinite(parameter).all() or not np.isfinite(gradient).all():
                raise FloatingPointError("parámetros y gradientes deben ser finitos")
            pairs.append((parameter, gradient))
        return pairs

    @abstractmethod
    def step(self, parameters: Sequence[np.ndarray], gradients: Sequence[np.ndarray]):
        """Actualizar los parámetros una vez a partir de sus gradientes."""

    def epoch_end(self, loss: float):
        """Recibir la pérdida de época; sólo eta adaptativo la utiliza."""
        if not np.isfinite(loss):
            raise FloatingPointError("loss debe ser finita")

    def reset(self):
        """Descartar el estado acumulado sin restaurar los parámetros del modelo."""


class GradientDescent(Optimizer):
    """Aplicar descenso básico: theta <- theta - eta * gradiente."""

    def step(self, parameters, gradients):
        for parameter, gradient in self._pairs(parameters, gradients):
            parameter -= self.learning_rate * gradient


class Momentum(Optimizer):
    """Acumular una fracción alpha del cambio anterior."""

    def __init__(self, learning_rate: float, *, alpha: float):
        super().__init__(learning_rate)
        self.alpha = _fraction(alpha, "alpha")
        self._previous_changes = None

    def step(self, parameters, gradients):
        pairs = self._pairs(parameters, gradients)
        if self._previous_changes is None:
            self._previous_changes = [np.zeros_like(parameter) for parameter, _ in pairs]
        if len(self._previous_changes) != len(pairs) or any(change.shape != parameter.shape
               for change, (parameter, _) in zip(self._previous_changes, pairs)):
            raise ValueError("el estado de Momentum no coincide con los parámetros")
        for parameter, gradient, previous in (
                (parameter, gradient, previous)
                for (parameter, gradient), previous in zip(pairs, self._previous_changes)):
            change = -self.learning_rate * gradient + self.alpha * previous
            parameter += change
            previous[...] = change

    def reset(self):
        self._previous_changes = None


class AdaptiveLearningRate(GradientDescent):
    """Aumentar o reducir eta tras cambios consistentes de la pérdida por época."""

    def __init__(self, learning_rate: float, *, increase_by: float,
                 decrease_fraction: float, patience: int):
        super().__init__(learning_rate)
        self.increase_by = _positive_finite(increase_by, "increase_by")
        self.decrease_fraction = _fraction(decrease_fraction, "decrease_fraction")
        if self.decrease_fraction == 0:
            raise ValueError("decrease_fraction debe ser positivo")
        if isinstance(patience, bool) or not isinstance(patience, (int, np.integer)) or patience < 1:
            raise ValueError("patience debe ser un entero positivo")
        self.patience = int(patience)
        self._previous_loss = None
        self._decrease_streak = 0
        self._increase_streak = 0

    def epoch_end(self, loss: float):
        super().epoch_end(loss)
        loss = float(loss)
        if self._previous_loss is not None:
            if loss < self._previous_loss:
                self._decrease_streak += 1
                self._increase_streak = 0
            elif loss > self._previous_loss:
                self._increase_streak += 1
                self._decrease_streak = 0
            else:
                self._decrease_streak = 0
                self._increase_streak = 0
            if self._decrease_streak == self.patience:
                self.learning_rate += self.increase_by
                self._decrease_streak = 0
            elif self._increase_streak == self.patience:
                self.learning_rate *= 1.0 - self.decrease_fraction
                self._increase_streak = 0
        self._previous_loss = loss

    def reset(self):
        self.learning_rate = self._initial_learning_rate
        self._previous_loss = None
        self._decrease_streak = 0
        self._increase_streak = 0


class RMSProp(Optimizer):
    """Dividir el gradiente por la raíz de su segundo momento móvil."""

    def __init__(self, learning_rate: float, *, gamma: float, epsilon: float):
        super().__init__(learning_rate)
        self.gamma = _fraction(gamma, "gamma")
        self.epsilon = _positive_finite(epsilon, "epsilon")
        self._squared_averages = None

    def step(self, parameters, gradients):
        pairs = self._pairs(parameters, gradients)
        if self._squared_averages is None:
            self._squared_averages = [np.zeros_like(parameter) for parameter, _ in pairs]
        if len(self._squared_averages) != len(pairs) or any(average.shape != parameter.shape
               for average, (parameter, _) in zip(self._squared_averages, pairs)):
            raise ValueError("el estado de RMSProp no coincide con los parámetros")
        for (parameter, gradient), average in zip(pairs, self._squared_averages):
            average *= self.gamma
            average += (1.0 - self.gamma) * gradient ** 2
            parameter -= self.learning_rate * gradient / np.sqrt(average + self.epsilon)

    def reset(self):
        self._squared_averages = None


class Adam(Optimizer):
    """Combinar Momentum y segundo momento con corrección de sesgo inicial."""

    def __init__(self, learning_rate: float = 0.001, *, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = _fraction(beta1, "beta1")
        self.beta2 = _fraction(beta2, "beta2")
        self.epsilon = _positive_finite(epsilon, "epsilon")
        self._first_moments = None
        self._second_moments = None
        self._step_count = 0

    def step(self, parameters, gradients):
        pairs = self._pairs(parameters, gradients)
        if self._first_moments is None:
            self._first_moments = [np.zeros_like(parameter) for parameter, _ in pairs]
            self._second_moments = [np.zeros_like(parameter) for parameter, _ in pairs]
        if (len(self._first_moments) != len(pairs)
                or len(self._second_moments) != len(pairs)
                or any(first.shape != parameter.shape or second.shape != parameter.shape
                       for first, second, (parameter, _) in
                       zip(self._first_moments, self._second_moments, pairs))):
            raise ValueError("el estado de Adam no coincide con los parámetros")
        self._step_count += 1
        correction1 = 1.0 - self.beta1 ** self._step_count
        correction2 = 1.0 - self.beta2 ** self._step_count
        for first, second, (parameter, gradient) in zip(
                self._first_moments, self._second_moments, pairs):
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * gradient ** 2
            corrected_first = first / correction1
            corrected_second = second / correction2
            parameter -= (self.learning_rate * corrected_first /
                          (np.sqrt(corrected_second) + self.epsilon))

    def reset(self):
        self._first_moments = None
        self._second_moments = None
        self._step_count = 0
