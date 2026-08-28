"""Pixel-based error and fitness functions."""

from typing import Tuple

import numpy as np
from PIL import Image

from .model import Individual, RGB
from .renderer import render_individual


def normalized_mse(target: Image.Image, generated: Image.Image) -> float:
    if target.size != generated.size:
        raise ValueError("target and generated images must have equal dimensions")
    target_array = np.asarray(target.convert("RGB"), dtype=np.float64)
    generated_array = np.asarray(generated.convert("RGB"), dtype=np.float64)
    value = float(np.mean(np.square(target_array - generated_array)) / (255.0**2))
    return min(1.0, max(0.0, value))


def fitness_from_error(error: float, epsilon: float) -> float:
    if not 0.0 <= error <= 1.0:
        raise ValueError("error must be in [0, 1]")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    return max(epsilon, 1.0 - error)


def evaluate_individual(
    individual: Individual,
    *,
    target: Image.Image,
    canvas_rgb: RGB,
    epsilon: float,
) -> Tuple[Individual, Image.Image]:
    generated = render_individual(
        individual, size=target.size, canvas_rgb=canvas_rgb
    )
    error = normalized_mse(target, generated)
    fitness = fitness_from_error(error, epsilon)
    return individual.evaluated(error=error, fitness=fitness), generated

