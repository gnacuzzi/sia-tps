"""Autoencoder denso implementado con NumPy a partir del MLP del TP3."""

from pathlib import Path
from typing import Sequence
import json

import numpy as np


class Autoencoder:
    """Red densa con una capa latente accesible como encoder y decoder."""

    def __init__(self, architecture: Sequence[int], *, latent_layer: int,
                 activations: Sequence[str], beta: float = 1.0,
                 loss: str = "mse", seed: int = 0, init_scale: float = 0.5,
                 leaky_slope: float = 0.01):
        self.architecture = tuple(architecture)
        self.activations = tuple(activations)
        if (len(self.architecture) < 3
                or any(isinstance(size, bool) or not isinstance(size, (int, np.integer))
                       or size <= 0 for size in self.architecture)):
            raise ValueError("architecture debe contener al menos tres tamaños positivos")
        if self.architecture[0] != self.architecture[-1]:
            raise ValueError("entrada y salida deben tener la misma dimensión")
        if (isinstance(latent_layer, bool) or not isinstance(latent_layer, int)
                or not 0 < latent_layer < len(self.architecture) - 1):
            raise ValueError("latent_layer debe identificar una capa interna")
        if len(self.activations) != len(self.architecture) - 1:
            raise ValueError("se necesita una activación por capa de pesos")
        if any(name not in {"linear", "tanh", "logistic", "leaky_relu"}
               for name in self.activations):
            raise ValueError("activación desconocida")
        if loss not in {"mse", "bce"}:
            raise ValueError("loss debe ser 'mse' o 'bce'")
        if loss == "bce" and self.activations[-1] != "logistic":
            raise ValueError("BCE requiere activación logística en la salida")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta debe ser positivo y finito")
        if not np.isfinite(init_scale) or init_scale < 0:
            raise ValueError("init_scale debe ser no negativo y finito")
        if (not np.isfinite(leaky_slope)
                or not 0.0 < leaky_slope < 1.0):
            raise ValueError("leaky_slope debe pertenecer a (0, 1)")

        self.latent_layer = latent_layer
        self.beta = float(beta)
        self.loss_name = loss
        self.leaky_slope = float(leaky_slope)
        generator = np.random.default_rng(seed)
        self.weights = [
            generator.uniform(-init_scale, init_scale, (output_size, input_size))
            for input_size, output_size in zip(
                self.architecture, self.architecture[1:])
        ]
        self.biases = [np.zeros(size) for size in self.architecture[1:]]

    def _activate(self, values, name):
        if name == "linear":
            return values
        if name == "tanh":
            return np.tanh(self.beta * values)
        if name == "leaky_relu":
            return np.where(values >= 0.0, values, self.leaky_slope * values)
        # sigmoid(2 beta h), la logística utilizada en el TP3.
        return np.exp(-np.logaddexp(0.0, -2.0 * self.beta * values))

    def _derivative(self, output, name):
        if name == "linear":
            return np.ones_like(output)
        if name == "tanh":
            return self.beta * (1.0 - output ** 2)
        if name == "leaky_relu":
            return np.where(output >= 0.0, 1.0, self.leaky_slope)
        return 2.0 * self.beta * output * (1.0 - output)

    def _layer_output(self, inputs, weights, biases, activation):
        # Algunas versiones de NumPy/Accelerate en macOS dejan flags de punto
        # flotante espurios tras matmul. Validamos el resultado en vez de aceptar
        # silenciosamente una activacion no finita.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            output = self._activate(inputs @ weights.T + biases, activation)
        if not np.isfinite(output).all():
            raise FloatingPointError("la propagacion produjo activaciones no finitas")
        return output

    @staticmethod
    def _matrix(values, width, name):
        matrix = np.asarray(values, dtype=np.float64)
        if (matrix.ndim != 2 or len(matrix) == 0 or matrix.shape[1] != width
                or not np.isfinite(matrix).all()):
            raise ValueError(
                f"{name} debe tener forma (muestras, {width}) y valores finitos"
            )
        return matrix

    def _forward(self, inputs):
        outputs = [inputs]
        for weights, biases, activation in zip(
                self.weights, self.biases, self.activations):
            outputs.append(self._layer_output(
                outputs[-1], weights, biases, activation
            ))
        return outputs

    def encode(self, inputs):
        """Comprimir entradas hasta la capa latente."""
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        output = inputs
        for layer in range(self.latent_layer):
            output = self._layer_output(
                output, self.weights[layer], self.biases[layer],
                self.activations[layer],
            )
        return output

    def decode(self, latent_codes):
        """Reconstruir salidas a partir de códigos latentes."""
        output = self._matrix(
            latent_codes, self.architecture[self.latent_layer], "latent_codes"
        )
        for layer in range(self.latent_layer, len(self.weights)):
            output = self._layer_output(
                output, self.weights[layer], self.biases[layer],
                self.activations[layer],
            )
        return output

    def predict(self, inputs):
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        return self._forward(inputs)[-1]

    def loss(self, inputs, targets=None):
        """Pérdida media entre la predicción y los objetivos esperados.

        Si no se indican ``targets``, se usa la entrada como objetivo para
        conservar el comportamiento del autoencoder común.
        """
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        if targets is None:
            targets = inputs
        else:
            targets = self._matrix(targets, self.architecture[-1], "targets")
            if len(targets) != len(inputs):
                raise ValueError("inputs y targets deben tener igual cantidad de muestras")
        predictions = self.predict(inputs)
        if self.loss_name == "mse":
            difference = predictions - targets
            return float(0.5 * np.mean(np.sum(difference ** 2, axis=1)))
        if np.any((targets < 0.0) | (targets > 1.0)):
            raise ValueError("BCE requiere objetivos entre 0 y 1")
        epsilon = np.finfo(np.float64).eps
        probabilities = np.clip(predictions, epsilon, 1.0 - epsilon)
        per_pixel = -(targets * np.log(probabilities)
                      + (1.0 - targets) * np.log1p(-probabilities))
        return float(np.mean(np.sum(per_pixel, axis=1)))

    def gradients(self, inputs, targets=None):
        """Calcular gradientes respecto de los objetivos sin actualizar."""
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        if targets is None:
            targets = inputs
        else:
            targets = self._matrix(targets, self.architecture[-1], "targets")
            if len(targets) != len(inputs):
                raise ValueError("inputs y targets deben tener igual cantidad de muestras")
        if self.loss_name == "bce" and np.any((targets < 0.0) | (targets > 1.0)):
            raise ValueError("BCE requiere objetivos entre 0 y 1")
        outputs = self._forward(inputs)
        if self.loss_name == "bce":
            # d(BCE)/dh para y_hat = sigmoid(2 beta h).
            delta = 2.0 * self.beta * (outputs[-1] - targets)
        else:
            delta = ((outputs[-1] - targets)
                     * self._derivative(outputs[-1], self.activations[-1]))
        weight_gradients = [None] * len(self.weights)
        bias_gradients = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                weight_gradients[layer] = delta.T @ outputs[layer] / len(inputs)
            bias_gradients[layer] = delta.mean(axis=0)
            if (not np.isfinite(weight_gradients[layer]).all()
                    or not np.isfinite(bias_gradients[layer]).all()):
                raise FloatingPointError("backpropagation produjo gradientes no finitos")
            if layer:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    delta = ((delta @ self.weights[layer]) * self._derivative(
                        outputs[layer], self.activations[layer - 1]
                    ))
                if not np.isfinite(delta).all():
                    raise FloatingPointError("backpropagation produjo deltas no finitos")
        return weight_gradients, bias_gradients

    @property
    def parameters(self):
        return self.weights + self.biases

    @classmethod
    def load(cls, path):
        """Reconstruir un autoencoder desde un archivo creado por ``save``."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            model = cls(
                metadata["architecture"],
                latent_layer=metadata["latent_layer"],
                activations=metadata["activations"],
                beta=metadata["beta"],
                loss=metadata.get("loss", "mse"),
                leaky_slope=metadata.get("leaky_slope", 0.01),
            )
            for index, parameter in enumerate(model.weights):
                stored = archive[f"w{index}"]
                if stored.shape != parameter.shape:
                    raise ValueError("los pesos guardados no coinciden con la arquitectura")
                parameter[...] = stored
            for index, parameter in enumerate(model.biases):
                stored = archive[f"b{index}"]
                if stored.shape != parameter.shape:
                    raise ValueError("los sesgos guardados no coinciden con la arquitectura")
                parameter[...] = stored
        return model

    def save(self, path):
        metadata = json.dumps({
            "architecture": self.architecture,
            "latent_layer": self.latent_layer,
            "activations": self.activations,
            "beta": self.beta,
            "loss": self.loss_name,
            "leaky_slope": self.leaky_slope,
        })
        arrays = {"metadata": np.array(metadata)}
        arrays.update({f"w{index}": value
                       for index, value in enumerate(self.weights)})
        arrays.update({f"b{index}": value
                       for index, value in enumerate(self.biases)})
        with Path(path).open("wb") as output:
            np.savez_compressed(output, **arrays)
