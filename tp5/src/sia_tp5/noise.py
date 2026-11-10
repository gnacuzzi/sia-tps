"""Corrupciones reproducibles para patrones binarios."""

import numpy as np


def bit_flip(inputs, probability, *, generator):
    """Invertir cada bit independientemente con la probabilidad indicada.

    El generador se recibe explícitamente para que una misma semilla reproduzca
    la secuencia completa de corrupciones sin repetir la misma muestra en cada
    llamada.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    if inputs.ndim != 2 or len(inputs) == 0 or not np.isfinite(inputs).all():
        raise ValueError("inputs debe ser una matriz no vacía de valores finitos")
    if not np.all((inputs == 0.0) | (inputs == 1.0)):
        raise ValueError("bit_flip requiere entradas binarias")
    if (isinstance(probability, (bool, np.bool_))
            or not np.isscalar(probability)
            or not np.isfinite(probability)
            or not 0.0 <= probability <= 1.0):
        raise ValueError("probability debe estar entre 0 y 1")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator debe ser un numpy.random.Generator")

    flips = generator.random(inputs.shape) < float(probability)
    return np.where(flips, 1.0 - inputs, inputs)
