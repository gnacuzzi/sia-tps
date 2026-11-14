"""VAE denso construido incrementalmente a partir del autoencoder del TP5."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .categorical import (categorical_cross_entropy, grouped_softmax,
                          validate_categorical_groups,
                          validate_categorical_targets)


class VariationalAutoencoder:
    """Encoder con ramas ``mu``/``log_var`` y muestreo reparametrizado."""

    def __init__(self, architecture: Sequence[int], *, latent_layer: int,
                 activations: Sequence[str], beta: float = 1.0,
                 loss: str = "bce", seed: int = 0, init_scale: float = 0.5,
                 leaky_slope: float = 0.01, sampling_seed=None,
                 kl_weight: float = 0.0, categorical_groups=None):
        self.architecture = tuple(architecture)
        self.activations = tuple(activations)
        if (len(self.architecture) < 3
                or any(isinstance(size, bool)
                       or not isinstance(size, (int, np.integer))
                       or size <= 0 for size in self.architecture)):
            raise ValueError(
                "architecture debe contener al menos tres tamanos positivos"
            )
        if self.architecture[0] != self.architecture[-1]:
            raise ValueError("entrada y salida deben tener la misma dimension")
        if (isinstance(latent_layer, bool)
                or not isinstance(latent_layer, int)
                or not 0 < latent_layer < len(self.architecture) - 1):
            raise ValueError("latent_layer debe identificar una capa interna")
        if len(self.activations) != len(self.architecture) - 1:
            raise ValueError("se necesita una activacion por capa conceptual")
        if any(name not in {
                "linear", "tanh", "logistic", "leaky_relu", "grouped_softmax"
        }
               for name in self.activations):
            raise ValueError("activacion desconocida")
        if "grouped_softmax" in self.activations[:-1]:
            raise ValueError("grouped_softmax solo puede usarse en la salida")
        if self.activations[latent_layer - 1] != "linear":
            raise ValueError("las ramas mu y log_var deben ser lineales")
        if loss not in {"mse", "bce", "categorical_cross_entropy"}:
            raise ValueError("loss desconocida")
        if loss == "bce" and self.activations[-1] != "logistic":
            raise ValueError("BCE requiere activacion logistica en la salida")
        categorical = loss == "categorical_cross_entropy"
        if categorical != (self.activations[-1] == "grouped_softmax"):
            raise ValueError(
                "grouped_softmax y categorical_cross_entropy se usan juntos"
            )
        if categorical:
            validate_categorical_groups(self.architecture[-1], categorical_groups)
        elif categorical_groups is not None:
            raise ValueError(
                "categorical_groups solo corresponde a la loss categorica"
            )
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta debe ser positivo y finito")
        if not np.isfinite(init_scale) or init_scale < 0:
            raise ValueError("init_scale debe ser no negativo y finito")
        if (not np.isfinite(leaky_slope)
                or not 0.0 < leaky_slope < 1.0):
            raise ValueError("leaky_slope debe pertenecer a (0, 1)")
        if (sampling_seed is not None
                and (isinstance(sampling_seed, bool)
                     or not isinstance(sampling_seed, (int, np.integer)))):
            raise ValueError("sampling_seed debe ser un entero o None")
        if not np.isfinite(kl_weight) or kl_weight < 0.0:
            raise ValueError("kl_weight debe ser no negativo y finito")

        self.latent_layer = latent_layer
        self.beta = float(beta)
        self.loss_name = loss
        self.leaky_slope = float(leaky_slope)
        self.kl_weight = float(kl_weight)
        self.categorical_groups = categorical_groups
        generator = np.random.default_rng(seed)
        self.sampling_generator = np.random.default_rng(
            seed if sampling_seed is None else sampling_seed
        )

        shared_sizes = self.architecture[:latent_layer]
        self.encoder_weights = [
            generator.uniform(-init_scale, init_scale, (output_size, input_size))
            for input_size, output_size in zip(shared_sizes, shared_sizes[1:])
        ]
        self.encoder_biases = [np.zeros(size) for size in shared_sizes[1:]]

        branch_input = self.architecture[latent_layer - 1]
        latent_size = self.architecture[latent_layer]
        self.mu_weight = generator.uniform(
            -init_scale, init_scale, (latent_size, branch_input)
        )
        self.mu_bias = np.zeros(latent_size)
        self.log_var_weight = generator.uniform(
            -init_scale, init_scale, (latent_size, branch_input)
        )
        self.log_var_bias = np.zeros(latent_size)

        decoder_sizes = self.architecture[latent_layer:]
        self.decoder_weights = [
            generator.uniform(-init_scale, init_scale, (output_size, input_size))
            for input_size, output_size in zip(decoder_sizes, decoder_sizes[1:])
        ]
        self.decoder_biases = [np.zeros(size) for size in decoder_sizes[1:]]

    def _activate(self, values, name):
        if name == "linear":
            return values
        if name == "tanh":
            return np.tanh(self.beta * values)
        if name == "leaky_relu":
            return np.where(values >= 0.0, values, self.leaky_slope * values)
        if name == "grouped_softmax":
            return grouped_softmax(values, self.categorical_groups)
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
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            output = self._activate(inputs @ weights.T + biases, activation)
        if not np.isfinite(output).all():
            raise FloatingPointError("la propagacion produjo valores no finitos")
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

    def _encode_forward(self, inputs):
        encoder_outputs = [inputs]
        encoder_activations = self.activations[:self.latent_layer - 1]
        for weights, biases, activation in zip(
                self.encoder_weights, self.encoder_biases,
                encoder_activations):
            encoder_outputs.append(self._layer_output(
                encoder_outputs[-1], weights, biases, activation
            ))

        shared = encoder_outputs[-1]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            mu = shared @ self.mu_weight.T + self.mu_bias
            log_var = shared @ self.log_var_weight.T + self.log_var_bias
        if not np.isfinite(mu).all() or not np.isfinite(log_var).all():
            raise FloatingPointError("mu o log_var contiene valores no finitos")
        return encoder_outputs, mu, log_var

    def reparameterize(self, mu, log_var, epsilon=None):
        """Construir ``z = mu + exp(0.5 log_var) * epsilon``."""
        latent_size = self.architecture[self.latent_layer]
        mu = self._matrix(mu, latent_size, "mu")
        log_var = self._matrix(log_var, latent_size, "log_var")
        if mu.shape != log_var.shape:
            raise ValueError("mu y log_var deben tener la misma forma")
        if epsilon is None:
            epsilon = self.sampling_generator.standard_normal(mu.shape)
        else:
            epsilon = np.asarray(epsilon, dtype=np.float64)
            if epsilon.shape != mu.shape or not np.isfinite(epsilon).all():
                raise ValueError("epsilon debe ser finito y tener la forma de mu")
        with np.errstate(over="ignore", invalid="ignore"):
            standard_deviation = np.exp(0.5 * log_var)
            z = mu + standard_deviation * epsilon
        if not np.isfinite(standard_deviation).all() or not np.isfinite(z).all():
            raise FloatingPointError("la reparametrizacion produjo valores no finitos")
        return z, standard_deviation, epsilon

    def _forward(self, inputs, epsilon=None):
        encoder_outputs, mu, log_var = self._encode_forward(inputs)
        z, standard_deviation, used_epsilon = self.reparameterize(
            mu, log_var, epsilon
        )

        decoder_outputs = [z]
        decoder_activations = self.activations[self.latent_layer:]
        for weights, biases, activation in zip(
                self.decoder_weights, self.decoder_biases,
                decoder_activations):
            decoder_outputs.append(self._layer_output(
                decoder_outputs[-1], weights, biases, activation
            ))
        return (encoder_outputs, mu, log_var, standard_deviation,
                used_epsilon, decoder_outputs)

    def encode_distribution(self, inputs):
        """Devolver los parametros de la distribucion diagonal q(z|x)."""
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        _, mu, log_var = self._encode_forward(inputs)
        return mu, log_var

    def encode(self, inputs, epsilon=None):
        """Muestrear un codigo mediante reparametrizacion."""
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        *_, decoder_outputs = self._forward(inputs, epsilon)
        return decoder_outputs[0]

    def decode(self, latent_codes):
        output = self._matrix(
            latent_codes, self.architecture[self.latent_layer], "latent_codes"
        )
        for weights, biases, activation in zip(
                self.decoder_weights, self.decoder_biases,
                self.activations[self.latent_layer:]):
            output = self._layer_output(output, weights, biases, activation)
        return output

    def predict(self, inputs, epsilon=None):
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        return self._forward(inputs, epsilon)[-1][-1]

    def loss_components(self, inputs, targets=None, epsilon=None):
        """Devolver reconstruccion, KL y su suma ponderada."""
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        if targets is None:
            targets = inputs
        else:
            targets = self._matrix(targets, self.architecture[-1], "targets")
            if len(targets) != len(inputs):
                raise ValueError("inputs y targets deben tener igual cantidad")
        _, mu, log_var, _, _, decoder_outputs = self._forward(inputs, epsilon)
        predictions = decoder_outputs[-1]
        if self.loss_name == "mse":
            difference = predictions - targets
            reconstruction = float(
                0.5 * np.mean(np.sum(difference ** 2, axis=1))
            )
        elif self.loss_name == "bce":
            if np.any((targets < 0.0) | (targets > 1.0)):
                raise ValueError("BCE requiere objetivos entre 0 y 1")
            clipping_epsilon = np.finfo(np.float64).eps
            probabilities = np.clip(
                predictions, clipping_epsilon, 1.0 - clipping_epsilon
            )
            per_value = -(targets * np.log(probabilities)
                          + (1.0 - targets) * np.log1p(-probabilities))
            reconstruction = float(np.mean(np.sum(per_value, axis=1)))
        else:
            reconstruction = categorical_cross_entropy(
                predictions, targets, self.categorical_groups
            )
        with np.errstate(over="ignore", invalid="ignore"):
            kl_per_sample = -0.5 * np.sum(
                1.0 + log_var - mu ** 2 - np.exp(log_var), axis=1
            )
        kl = max(0.0, float(np.mean(kl_per_sample)))
        total = reconstruction + self.kl_weight * kl
        if not np.isfinite(total):
            raise FloatingPointError("la perdida variacional no es finita")
        return {
            "total": total,
            "reconstruction": reconstruction,
            "kl": kl,
        }

    def loss(self, inputs, targets=None, epsilon=None):
        """Perdida total: reconstruccion mas ``kl_weight * KL``."""
        return self.loss_components(inputs, targets, epsilon)["total"]

    def gradients(self, inputs, targets=None, epsilon=None):
        """Backpropagar reconstruccion por ambas ramas reparametrizadas."""
        inputs = self._matrix(inputs, self.architecture[0], "inputs")
        if targets is None:
            targets = inputs
        else:
            targets = self._matrix(targets, self.architecture[-1], "targets")
            if len(targets) != len(inputs):
                raise ValueError("inputs y targets deben tener igual cantidad")
        if self.loss_name == "bce" and np.any(
                (targets < 0.0) | (targets > 1.0)):
            raise ValueError("BCE requiere objetivos entre 0 y 1")
        if self.loss_name == "categorical_cross_entropy":
            validate_categorical_targets(targets, self.categorical_groups)

        (encoder_outputs, mu, log_var, standard_deviation, used_epsilon,
         decoder_outputs) = self._forward(inputs, epsilon)
        if self.loss_name == "categorical_cross_entropy":
            delta = decoder_outputs[-1] - targets
        elif self.loss_name == "bce":
            delta = 2.0 * self.beta * (decoder_outputs[-1] - targets)
        else:
            delta = ((decoder_outputs[-1] - targets) * self._derivative(
                decoder_outputs[-1], self.activations[-1]
            ))

        decoder_weight_gradients = [None] * len(self.decoder_weights)
        decoder_bias_gradients = [None] * len(self.decoder_biases)
        decoder_activations = self.activations[self.latent_layer:]
        for layer in range(len(self.decoder_weights) - 1, -1, -1):
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                decoder_weight_gradients[layer] = (
                    delta.T @ decoder_outputs[layer] / len(inputs)
                )
            decoder_bias_gradients[layer] = delta.mean(axis=0)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                delta = ((delta @ self.decoder_weights[layer]) * self._derivative(
                    decoder_outputs[layer],
                    "linear" if layer == 0 else decoder_activations[layer - 1],
                ))

        shared = encoder_outputs[-1]
        mu_delta = delta + self.kl_weight * mu
        log_var_delta = (
            delta * used_epsilon * 0.5 * standard_deviation
            + self.kl_weight * 0.5 * (np.exp(log_var) - 1.0)
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            mu_weight_gradient = mu_delta.T @ shared / len(inputs)
        mu_bias_gradient = mu_delta.mean(axis=0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            log_var_weight_gradient = log_var_delta.T @ shared / len(inputs)
        log_var_bias_gradient = log_var_delta.mean(axis=0)

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            delta = (mu_delta @ self.mu_weight
                     + log_var_delta @ self.log_var_weight)
        encoder_weight_gradients = [None] * len(self.encoder_weights)
        encoder_bias_gradients = [None] * len(self.encoder_biases)
        encoder_activations = self.activations[:self.latent_layer - 1]
        for layer in range(len(self.encoder_weights) - 1, -1, -1):
            delta *= self._derivative(
                encoder_outputs[layer + 1], encoder_activations[layer]
            )
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                encoder_weight_gradients[layer] = (
                    delta.T @ encoder_outputs[layer] / len(inputs)
                )
            encoder_bias_gradients[layer] = delta.mean(axis=0)
            if layer:
                with np.errstate(
                        over="ignore", divide="ignore", invalid="ignore"):
                    delta = delta @ self.encoder_weights[layer]

        weight_gradients = (
            encoder_weight_gradients
            + [mu_weight_gradient, log_var_weight_gradient]
            + decoder_weight_gradients
        )
        bias_gradients = (
            encoder_bias_gradients
            + [mu_bias_gradient, log_var_bias_gradient]
            + decoder_bias_gradients
        )
        gradients = weight_gradients + bias_gradients
        if not all(np.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("backpropagation produjo gradientes no finitos")
        return weight_gradients, bias_gradients

    @property
    def parameters(self):
        return (
            self.encoder_weights
            + [self.mu_weight, self.log_var_weight]
            + self.decoder_weights
            + self.encoder_biases
            + [self.mu_bias, self.log_var_bias]
            + self.decoder_biases
        )

    def save(self, path):
        """Guardar arquitectura, hiperparametros y parametros entrenados."""
        metadata = json.dumps({
            "architecture": self.architecture,
            "latent_layer": self.latent_layer,
            "activations": self.activations,
            "beta": self.beta,
            "loss": self.loss_name,
            "leaky_slope": self.leaky_slope,
            "kl_weight": self.kl_weight,
            "categorical_groups": self.categorical_groups,
        })
        arrays = {"metadata": np.array(metadata)}
        arrays.update({
            f"parameter{index}": parameter
            for index, parameter in enumerate(self.parameters)
        })
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as file:
            np.savez_compressed(file, **arrays)

    @classmethod
    def load(cls, path):
        """Reconstruir un VAE guardado por :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            model = cls(
                metadata["architecture"],
                latent_layer=metadata["latent_layer"],
                activations=metadata["activations"],
                beta=metadata["beta"],
                loss=metadata["loss"],
                leaky_slope=metadata["leaky_slope"],
                kl_weight=metadata["kl_weight"],
                categorical_groups=metadata.get("categorical_groups"),
            )
            for index, parameter in enumerate(model.parameters):
                stored = archive[f"parameter{index}"]
                if stored.shape != parameter.shape:
                    raise ValueError(
                        "los parametros guardados no coinciden con la arquitectura"
                    )
                parameter[...] = stored
        return model

