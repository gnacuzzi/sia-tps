"""Datos binarios diminutos para aislar el mecanismo del VAE."""

import numpy as np


TOY_BAR_LABELS = tuple(
    [f"horizontal_{index}" for index in range(3)]
    + [f"vertical_{index}" for index in range(3)]
)


def toy_bars():
    """Devolver seis barras 3x3 aplanadas, sin relacion con el dataset final."""
    patterns = []
    for row in range(3):
        image = np.zeros((3, 3), dtype=np.float64)
        image[row, :] = 1.0
        patterns.append(image.ravel())
    for column in range(3):
        image = np.zeros((3, 3), dtype=np.float64)
        image[:, column] = 1.0
        patterns.append(image.ravel())
    return np.asarray(patterns)

