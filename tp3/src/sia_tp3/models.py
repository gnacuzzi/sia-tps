"""Implementar las neuronas y backpropagation con NumPy."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class MultilayerPerceptron:
    """Construir una red densa con bias y una activación por capa."""

    def __init__(self, architecture: Sequence[int], *, activations: Sequence[str],
                 beta: float = 1.0, seed: int = 0, init_scale: float = 0.5,
                 loss: str = "mse", l2_lambda: float = 0.0):
        self.architecture = tuple(architecture)
        self.activations = tuple(activations)
        if (len(self.architecture) < 2 or any(
                isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0
                for n in self.architecture)):
            raise ValueError("architecture debe contener al menos dos tamaños enteros positivos")
        if len(self.activations) != len(self.architecture) - 1:
            raise ValueError("se necesita una activación por capa de pesos")
        if any(a not in {"step", "linear", "tanh", "logistic", "softmax"}
               for a in self.activations):
            raise ValueError("activación desconocida")
        if "softmax" in self.activations[:-1]:
            raise ValueError("softmax sólo puede utilizarse en la capa de salida")
        if "step" in self.activations and (len(self.architecture) != 2 or self.architecture[-1] != 1):
            raise ValueError("escalón solo está permitido en el perceptrón simple de una salida")
        if not np.isfinite(beta) or beta <= 0 or not np.isfinite(init_scale) or init_scale < 0:
            raise ValueError("beta debe ser positivo e init_scale no negativo, ambos finitos")
        if loss not in {"mse", "categorical_cross_entropy"}:
            raise ValueError("función de loss desconocida")
        if ((loss == "categorical_cross_entropy")
                != (self.activations[-1] == "softmax")):
            raise ValueError(
                "softmax y categorical_cross_entropy deben utilizarse juntos")
        if (not np.isfinite(l2_lambda) or l2_lambda < 0):
            raise ValueError("l2_lambda debe ser no negativo y finito")
        if self.activations[-1] == "step" and l2_lambda != 0:
            raise ValueError("L2 no está definido para Rosenblatt con escalón")
        self.beta = float(beta)
        self.loss_name = loss
        self.l2_lambda = float(l2_lambda)
        rng = np.random.default_rng(seed)
        self.weights = [rng.uniform(-init_scale, init_scale, (out_size, in_size))
                        for in_size, out_size in zip(self.architecture, self.architecture[1:])]
        self.biases = [np.zeros(out_size) for out_size in self.architecture[1:]]

    def _activate(self, h, name):
        if name == "step":
            return np.where(h >= 0, 1.0, -1.0)
        if name == "linear":
            return h
        if name == "tanh":
            return np.tanh(self.beta * h)
        if name == "softmax":
            shifted = h - np.max(h, axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            return exponentials / exponentials.sum(axis=1, keepdims=True)
        # Logística de clase: sigmoid(2 * beta * h), evaluada sin overflow.
        return np.exp(-np.logaddexp(0.0, -2.0 * self.beta * h))

    def _derivative(self, output, name):
        if name == "linear":
            return np.ones_like(output)
        if name == "tanh":
            return self.beta * (1.0 - output ** 2)
        if name == "logistic":
            return 2.0 * self.beta * output * (1.0 - output)
        raise ValueError("el escalón usa Rosenblatt, no una derivada")

    def _inputs(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != self.architecture[0]:
            raise ValueError("X debe tener forma (muestras, entradas) y no estar vacío")
        if not np.isfinite(X).all():
            raise ValueError("X debe contener valores finitos")
        return X

    def validate_data(self, X, y):
        """Validar matrices de entrada y objetivo sin broadcasting implícito."""
        X = self._inputs(X)
        y = np.asarray(y, dtype=np.float64)
        if y.shape != (len(X), self.architecture[-1]) or not np.isfinite(y).all():
            raise ValueError("y debe tener forma (muestras, salidas) y contener valores finitos")
        if self.activations[-1] == "step" and not np.isin(y, [-1, 1]).all():
            raise ValueError("el escalón requiere objetivos bipolares -1/+1")
        if self.loss_name == "categorical_cross_entropy" and (
                np.any(y < 0) or not np.allclose(y.sum(axis=1), 1.0)):
            raise ValueError(
                "categorical_cross_entropy requiere distribuciones objetivo que sumen 1")
        return X, y

    def _forward(self, X):
        outputs = [X]
        for w, b, name in zip(self.weights, self.biases, self.activations):
            outputs.append(self._activate(outputs[-1] @ w.T + b, name))
        return outputs

    def predict(self, X):
        """Predecir una matriz (muestras, salidas) sin umbralizar activaciones suaves."""
        return self._forward(self._inputs(X))[-1]

    def data_loss(self, X, y):
        """Calcular sólo la pérdida sobre los datos, sin regularización."""
        X, y = self.validate_data(X, y)
        predictions = self._forward(X)[-1]
        if self.loss_name == "categorical_cross_entropy":
            safe = np.clip(predictions, np.finfo(np.float64).tiny, 1.0)
            return float(-np.mean(np.sum(y * np.log(safe), axis=1)))
        return float(0.5 * np.mean(np.sum((predictions - y) ** 2, axis=1)))

    def regularization_loss(self):
        """Calcular 1/2 lambda por la suma cuadrática de los pesos, sin bias."""
        return float(0.5 * self.l2_lambda * sum(
            np.sum(weight ** 2) for weight in self.weights))

    def loss(self, X, y):
        """Calcular el objetivo de entrenamiento: datos más regularización L2."""
        return self.data_loss(X, y) + self.regularization_loss()

    def _gradients(self, X, y):
        outputs = self._forward(X)
        if self.loss_name == "categorical_cross_entropy":
            delta = outputs[-1] - y
        else:
            delta = ((outputs[-1] - y) *
                     self._derivative(outputs[-1], self.activations[-1]))
        dw, db = [None] * len(self.weights), [None] * len(self.weights)
        for layer in range(len(self.weights) - 1, -1, -1):
            dw[layer] = delta.T @ outputs[layer] / len(X)
            dw[layer] += self.l2_lambda * self.weights[layer]
            db[layer] = delta.mean(axis=0)
            if layer:
                delta = ((delta @ self.weights[layer]) *
                         self._derivative(outputs[layer], self.activations[layer - 1]))
        return dw, db

    def gradients(self, X, y):
        """Calcular gradientes de loss sin modificar ningún parámetro."""
        X, y = self.validate_data(X, y)
        return self._gradients(X, y)

    def _update(self, X, y, learning_rate):
        if self.activations[-1] == "step":
            error = y - self._forward(X)[-1]
            self.weights[0] += learning_rate * error.T @ X
            self.biases[0] += learning_rate * error[0]
        else:
            dw, db = self._gradients(X, y)
            for w, b, gw, gb in zip(self.weights, self.biases, dw, db):
                w -= learning_rate * gw
                b -= learning_rate * gb

    def train_sample(self, X, y, *, learning_rate: float):
        """Actualizar los parámetros usando exactamente una muestra."""
        X, y = self.validate_data(X, y)
        if len(X) != 1:
            raise ValueError("train_sample requiere exactamente una muestra")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo y finito")
        self._update(X, y, learning_rate)

    def save(self, path):
        """Guardar arquitectura, activaciones, beta, pesos y bias sin pickle."""
        metadata = json.dumps({"architecture": self.architecture,
                               "activations": self.activations, "beta": self.beta,
                               "loss": self.loss_name,
                               "l2_lambda": self.l2_lambda})
        arrays = {"metadata": np.array(metadata)}
        arrays.update({f"w{i}": w for i, w in enumerate(self.weights)})
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        with Path(path).open("wb") as file:
            np.savez_compressed(file, **arrays)

    @staticmethod
    def load(path):
        """Recuperar un modelo y permitir continuar entrenando sus parámetros."""
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            model = MultilayerPerceptron(**metadata, init_scale=0.0)
            for i in range(len(model.weights)):
                for key, target in [(f"w{i}", model.weights[i]), (f"b{i}", model.biases[i])]:
                    value = data[key]
                    if value.shape != target.shape or not np.isfinite(value).all():
                        raise ValueError("parámetros guardados inválidos")
                    target[...] = value
        return model


class Perceptron(MultilayerPerceptron):
    """Construir una única neurona escalón, lineal, tanh o logística."""

    def __init__(self, input_size: int, *, activation: str, beta: float = 1.0,
                 seed: int = 0, init_scale: float = 0.5):
        super().__init__([input_size, 1], activations=[activation], beta=beta,
                         seed=seed, init_scale=init_scale)
