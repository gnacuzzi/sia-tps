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


class Momentum(GradientDescent):
    """Acumular una fraccion del cambio aplicado en el paso anterior."""

    def __init__(self, learning_rate: float, *, alpha: float = 0.9):
        super().__init__(learning_rate)
        if not np.isfinite(alpha) or not 0 <= alpha < 1:
            raise ValueError("alpha debe pertenecer a [0, 1)")
        self.alpha = float(alpha)
        self._previous_changes = None

    def step(self, parameters, gradients):
        if len(parameters) == 0 or len(parameters) != len(gradients):
            raise ValueError("parameters y gradients deben coincidir")
        if self._previous_changes is None:
            self._previous_changes = [np.zeros_like(parameter)
                                      for parameter in parameters]
        for parameter, gradient, previous in zip(
                parameters, gradients, self._previous_changes):
            if parameter.shape != gradient.shape:
                raise ValueError("cada gradiente debe coincidir con su parametro")
            change = -self.learning_rate * gradient + self.alpha * previous
            parameter += change
            previous[...] = change


class RMSProp(GradientDescent):
    """Adaptar cada paso usando el promedio movil del gradiente al cuadrado."""

    def __init__(self, learning_rate: float, *, gamma: float = 0.9,
                 epsilon: float = 1e-8):
        super().__init__(learning_rate)
        if not np.isfinite(gamma) or not 0 <= gamma < 1:
            raise ValueError("gamma debe pertenecer a [0, 1)")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon debe ser positivo y finito")
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self._squared_averages = None

    def step(self, parameters, gradients):
        if len(parameters) == 0 or len(parameters) != len(gradients):
            raise ValueError("parameters y gradients deben coincidir")
        if self._squared_averages is None:
            self._squared_averages = [np.zeros_like(parameter)
                                      for parameter in parameters]
        for parameter, gradient, average in zip(
                parameters, gradients, self._squared_averages):
            if parameter.shape != gradient.shape:
                raise ValueError("cada gradiente debe coincidir con su parametro")
            average *= self.gamma
            average += (1.0 - self.gamma) * gradient ** 2
            parameter -= (self.learning_rate * gradient
                          / np.sqrt(average + self.epsilon))


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


class ExponentialDecay:
    """Reducir exponencialmente la tasa de un optimizador entre dos extremos."""

    def __init__(self, optimizer, *, final_learning_rate: float,
                 total_steps: int):
        if (not np.isfinite(final_learning_rate)
                or final_learning_rate <= 0):
            raise ValueError("final_learning_rate debe ser positivo y finito")
        if (isinstance(total_steps, bool) or not isinstance(total_steps, int)
                or total_steps < 2):
            raise ValueError("total_steps debe ser un entero mayor o igual a 2")
        self.optimizer = optimizer
        self.initial_learning_rate = optimizer.learning_rate
        self.final_learning_rate = float(final_learning_rate)
        self.total_steps = total_steps
        self._step = 0

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    def step(self, parameters, gradients):
        bounded_step = min(self._step, self.total_steps - 1)
        fraction = bounded_step / (self.total_steps - 1)
        ratio = self.final_learning_rate / self.initial_learning_rate
        self.optimizer.learning_rate = (
            self.initial_learning_rate * ratio ** fraction
        )
        self.optimizer.step(parameters, gradients)
        self._step += 1


def optimizer_from_config(configuration):
    """Construir un optimizador a partir de su configuracion serializable."""
    if not isinstance(configuration, dict) or "name" not in configuration:
        raise ValueError("la configuracion del optimizador requiere name")
    name = configuration["name"]
    options = {
        key: value for key, value in configuration.items()
        if key not in {"name", "schedule"}
    }
    classes = {
        "gradient_descent": GradientDescent,
        "momentum": Momentum,
        "rmsprop": RMSProp,
        "adam": Adam,
    }
    if name not in classes:
        raise ValueError(f"optimizador desconocido: {name}")
    optimizer = classes[name](**options)
    schedule = configuration.get("schedule")
    if schedule is None:
        return optimizer
    if not isinstance(schedule, dict) or schedule.get("name") != "exponential":
        raise ValueError("schedule debe definir name='exponential'")
    schedule_options = {
        key: value for key, value in schedule.items() if key != "name"
    }
    return ExponentialDecay(optimizer, **schedule_options)
