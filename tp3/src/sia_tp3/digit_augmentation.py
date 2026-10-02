"""Transformaciones conservadoras para imágenes de dígitos de 28 por 28."""

import numpy as np


IMAGE_SIDE = 28
PIXEL_COUNT = IMAGE_SIDE * IMAGE_SIDE


def _shift_slices(offset):
    if offset > 0:
        return slice(0, IMAGE_SIDE - offset), slice(offset, IMAGE_SIDE)
    if offset < 0:
        return slice(-offset, IMAGE_SIDE), slice(0, IMAGE_SIDE + offset)
    return slice(None), slice(None)


def random_translate_images(X, rng, *, max_shift, probability):
    """Trasladar una fracción de imágenes sin envolver píxeles en los bordes."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != PIXEL_COUNT:
        raise ValueError("X debe tener forma (muestras, 784)")
    if not np.isfinite(X).all():
        raise ValueError("X debe contener valores finitos")
    if (isinstance(max_shift, bool) or not isinstance(max_shift, (int, np.integer))
            or not 1 <= max_shift < IMAGE_SIDE):
        raise ValueError("max_shift debe ser un entero entre 1 y 27")
    if (not isinstance(probability, (int, float, np.integer, np.floating))
            or isinstance(probability, (bool, np.bool_))
            or not np.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("probability debe pertenecer a [0, 1]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng debe ser un Generator de NumPy")

    output = X.copy()
    selected = np.flatnonzero(rng.random(len(X)) < probability)
    if len(selected) == 0:
        return output

    displacements = np.array([
        (dy, dx)
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
        if (dy, dx) != (0, 0)
    ], dtype=np.int64)
    choices = rng.integers(0, len(displacements), size=len(selected))
    source_images = X.reshape(-1, IMAGE_SIDE, IMAGE_SIDE)
    output_images = output.reshape(-1, IMAGE_SIDE, IMAGE_SIDE)
    for choice in np.unique(choices):
        indices = selected[choices == choice]
        dy, dx = displacements[choice]
        source_y, destination_y = _shift_slices(int(dy))
        source_x, destination_x = _shift_slices(int(dx))
        translated = np.zeros(
            (len(indices), IMAGE_SIDE, IMAGE_SIDE), dtype=X.dtype)
        translated[:, destination_y, destination_x] = source_images[
            indices, source_y, source_x]
        output_images[indices] = translated
    return output


def random_rotate_images(X, rng, *, max_angle_degrees, probability):
    """Rotar imágenes alrededor del centro con interpolación bilineal y fondo cero."""
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != PIXEL_COUNT:
        raise ValueError("X debe tener forma (muestras, 784)")
    if not np.isfinite(X).all():
        raise ValueError("X debe contener valores finitos")
    if (not isinstance(max_angle_degrees,
                       (int, float, np.integer, np.floating))
            or isinstance(max_angle_degrees, (bool, np.bool_))
            or not np.isfinite(max_angle_degrees)
            or not 0 < max_angle_degrees <= 45):
        raise ValueError("max_angle_degrees debe pertenecer a (0, 45]")
    if (not isinstance(probability, (int, float, np.integer, np.floating))
            or isinstance(probability, (bool, np.bool_))
            or not np.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError("probability debe pertenecer a [0, 1]")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng debe ser un Generator de NumPy")

    output = X.copy()
    selected = np.flatnonzero(rng.random(len(X)) < probability)
    if len(selected) == 0:
        return output

    angles = np.deg2rad(rng.uniform(
        -max_angle_degrees, max_angle_degrees, size=len(selected)))
    center = (IMAGE_SIDE - 1) / 2
    rows, columns = np.indices((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float64)
    x = columns[None, :, :] - center
    y = rows[None, :, :] - center
    cosine = np.cos(angles)[:, None, None]
    sine = np.sin(angles)[:, None, None]

    # Mapeo inverso: para cada píxel de destino se interpola su origen.
    source_x = cosine * x + sine * y + center
    source_y = -sine * x + cosine * y + center
    x0 = np.floor(source_x).astype(np.int64)
    y0 = np.floor(source_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = source_x - x0
    wy = source_y - y0
    images = X.reshape(-1, IMAGE_SIDE, IMAGE_SIDE)[selected]

    def sample(sample_y, sample_x):
        valid = ((sample_y >= 0) & (sample_y < IMAGE_SIDE)
                 & (sample_x >= 0) & (sample_x < IMAGE_SIDE))
        clipped_y = np.clip(sample_y, 0, IMAGE_SIDE - 1)
        clipped_x = np.clip(sample_x, 0, IMAGE_SIDE - 1)
        batch = np.arange(len(selected))[:, None, None]
        return images[batch, clipped_y, clipped_x] * valid

    rotated = (
        sample(y0, x0) * (1 - wx) * (1 - wy)
        + sample(y0, x1) * wx * (1 - wy)
        + sample(y1, x0) * (1 - wx) * wy
        + sample(y1, x1) * wx * wy
    )
    output.reshape(-1, IMAGE_SIDE, IMAGE_SIDE)[selected] = rotated
    return output


def random_translate_rotate_images(
        X, rng, *, max_shift, translation_probability,
        max_angle_degrees, rotation_probability):
    """Combinar traslación y rotación con decisiones aleatorias independientes."""
    translated = random_translate_images(
        X, rng, max_shift=max_shift, probability=translation_probability)
    return random_rotate_images(
        translated, rng, max_angle_degrees=max_angle_degrees,
        probability=rotation_probability)
