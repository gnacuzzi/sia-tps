"""Optimizadores reutilizados del TP3."""

import numpy as np


class GradientDescent:
    def __init__(self, learning_rate: float):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo y finito")
        self.learning_rate = float(learning_rate)

    def step(self, parameters, gradients):
        if len(parameters) == 0 or len(parameters) != len(gradients):
            raise ValueError("parameters y gradients deben coincidir")
        for parameter, gradient in zip(parameters, gradients):
            if parameter.shape != gradient.shape:
                raise ValueError("cada gradiente debe coincidir con su parámetro")
            parameter -= self.learning_rate * gradient


class Adam(GradientDescent):
    def __init__(self, learning_rate: float = 0.001, *, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 y beta2 deben pertenecer a [0, 1)")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon debe ser positivo y finito")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self._first_moments = None
        self._second_moments = None
        self._step = 0

    def step(self, parameters, gradients):
        pairs = list(zip(parameters, gradients))
        if not pairs or len(parameters) != len(gradients):
            raise ValueError("parameters y gradients deben coincidir")
        if self._first_moments is None:
            self._first_moments = [np.zeros_like(parameter)
                                   for parameter, _ in pairs]
            self._second_moments = [np.zeros_like(parameter)
                                    for parameter, _ in pairs]
        self._step += 1
        correction1 = 1.0 - self.beta1 ** self._step
        correction2 = 1.0 - self.beta2 ** self._step
        for parameter, gradient, first, second in zip(
                parameters, gradients, self._first_moments, self._second_moments):
            if parameter.shape != gradient.shape:
                raise ValueError("cada gradiente debe coincidir con su parámetro")
            first *= self.beta1
            first += (1.0 - self.beta1) * gradient
            second *= self.beta2
            second += (1.0 - self.beta2) * gradient ** 2
            parameter -= self.learning_rate * (first / correction1) / (
                np.sqrt(second / correction2) + self.epsilon
            )
