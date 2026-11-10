"""Entrenamiento y métricas explícitas para el autoencoder."""

import numpy as np

from .noise import bit_flip


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


def fit_autoencoder(model, inputs, targets=None, *, optimizer, max_epochs: int,
                    target_max_pixel_errors=0, progress=None,
                    input_corruptor=None):
    """Entrenar full-batch contra objetivos explícitos o contra la entrada."""
    inputs = np.asarray(inputs, dtype=np.float64)
    if targets is None:
        targets = inputs
    else:
        targets = np.asarray(targets, dtype=np.float64)
    if inputs.shape != targets.shape or inputs.ndim != 2:
        raise ValueError("inputs y targets deben ser matrices de igual forma")
    if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 1:
        raise ValueError("max_epochs debe ser un entero positivo")
    if (target_max_pixel_errors is not None
            and (isinstance(target_max_pixel_errors, bool)
                 or not isinstance(target_max_pixel_errors, int)
                 or target_max_pixel_errors < 0)):
        raise ValueError("target_max_pixel_errors debe ser None o un entero no negativo")
    history = []
    for epoch in range(max_epochs + 1):
        epoch_inputs = inputs if input_corruptor is None else np.asarray(
            input_corruptor(inputs), dtype=np.float64
        )
        if epoch_inputs.shape != inputs.shape or not np.isfinite(epoch_inputs).all():
            raise ValueError(
                "input_corruptor debe devolver una matriz finita de igual forma"
            )
        predictions = model.predict(epoch_inputs)
        metrics = reconstruction_metrics(targets, predictions)
        row = {
            "epoch": epoch,
            "loss": model.loss(epoch_inputs, targets),
            "learning_rate": optimizer.learning_rate,
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
        weight_gradients, bias_gradients = model.gradients(epoch_inputs, targets)
        optimizer.step(model.parameters, weight_gradients + bias_gradients)
    return history


def fit_denoising_autoencoder(model, clean_inputs, *, optimizer, max_epochs: int,
                              noise_probability: float, generator,
                              progress=None, corruptions_per_sample: int = 1):
    """Entrenar con una corrupción bit-flip nueva y el objetivo limpio."""
    clean_inputs = np.asarray(clean_inputs, dtype=np.float64)
    if (isinstance(corruptions_per_sample, bool)
            or not isinstance(corruptions_per_sample, int)
            or corruptions_per_sample < 1):
        raise ValueError("corruptions_per_sample debe ser un entero positivo")
    repeated_clean = np.repeat(clean_inputs, corruptions_per_sample, axis=0)

    def corrupt(inputs):
        return bit_flip(inputs, noise_probability, generator=generator)

    return fit_autoencoder(
        model,
        repeated_clean,
        repeated_clean,
        optimizer=optimizer,
        max_epochs=max_epochs,
        target_max_pixel_errors=None,
        progress=progress,
        input_corruptor=corrupt,
    )


def fit_mixed_denoising_autoencoder(
        model, clean_inputs, *, optimizer, max_epochs: int,
        noise_probability: float, generator, progress=None):
    """Entrenar cada época con una copia limpia y otra ruidosa por muestra."""
    clean_inputs = np.asarray(clean_inputs, dtype=np.float64)
    mixed_inputs = np.concatenate((clean_inputs, clean_inputs), axis=0)
    mixed_targets = mixed_inputs.copy()
    clean_count = len(clean_inputs)

    def corrupt_noisy_half(inputs):
        return np.concatenate((
            inputs[:clean_count],
            bit_flip(
                inputs[clean_count:], noise_probability, generator=generator
            ),
        ), axis=0)

    return fit_autoencoder(
        model,
        mixed_inputs,
        mixed_targets,
        optimizer=optimizer,
        max_epochs=max_epochs,
        target_max_pixel_errors=None,
        progress=progress,
        input_corruptor=corrupt_noisy_half,
    )


def fit_multilevel_denoising_autoencoder(
        model, clean_inputs, *, optimizer, max_epochs: int,
        noise_probabilities, generator, progress=None):
    """Entrenar con una copia limpia y una corrupción por nivel y muestra."""
    clean_inputs = np.asarray(clean_inputs, dtype=np.float64)
    probabilities = tuple(noise_probabilities)
    if not probabilities or len(set(probabilities)) != len(probabilities):
        raise ValueError("noise_probabilities debe contener niveles únicos")

    block_count = 1 + len(probabilities)
    mixed_inputs = np.concatenate([clean_inputs] * block_count, axis=0)
    mixed_targets = mixed_inputs.copy()
    sample_count = len(clean_inputs)

    def corrupt_levels(inputs):
        blocks = [inputs[:sample_count]]
        for index, probability in enumerate(probabilities, start=1):
            start = index * sample_count
            blocks.append(bit_flip(
                inputs[start:start + sample_count],
                probability,
                generator=generator,
            ))
        return np.concatenate(blocks, axis=0)

    return fit_autoencoder(
        model,
        mixed_inputs,
        mixed_targets,
        optimizer=optimizer,
        max_epochs=max_epochs,
        target_max_pixel_errors=None,
        progress=progress,
        input_corruptor=corrupt_levels,
    )
