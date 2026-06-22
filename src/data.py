"""Synthetic textured surface images, with and without defects.

A clean sample is a smooth low frequency texture field built from a few sine
gratings plus mild Gaussian noise. A defective sample takes a clean field and
stamps a localized anomaly onto it: a bright or dark blob, or a thin scratch.
The anomaly is local and high contrast, which is what a real surface defect
looks like against an otherwise regular material texture.

Everything is seeded so a given seed always yields the same image. That makes
the tests deterministic.
"""

from __future__ import annotations

import numpy as np

IMG_SIZE = 32


def _base_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    """A regular low frequency texture in the range [0, 1]."""
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, size),
        np.linspace(0.0, 1.0, size),
        indexing="ij",
    )
    field = np.zeros((size, size), dtype=np.float64)
    for _ in range(3):
        fx = rng.uniform(2.0, 6.0)
        fy = rng.uniform(2.0, 6.0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.2, 0.5)
        field += amp * np.sin(2.0 * np.pi * (fx * xx + fy * yy) + phase)
    field += rng.normal(0.0, 0.05, size=(size, size))
    field -= field.min()
    span = field.max() - field.min()
    if span > 0:
        field /= span
    return field


def _stamp_blob(
    img: np.ndarray, rng: np.random.Generator, size: int
) -> np.ndarray:
    """Add a localized bright or dark Gaussian blob."""
    cy = rng.integers(size // 4, 3 * size // 4)
    cx = rng.integers(size // 4, 3 * size // 4)
    radius = rng.uniform(size * 0.06, size * 0.15)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    strength = rng.uniform(0.6, 1.0) * sign
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * radius**2))
    return img + strength * blob


def _stamp_scratch(
    img: np.ndarray, rng: np.random.Generator, size: int
) -> np.ndarray:
    """Add a thin high contrast line, like a scratch."""
    angle = rng.uniform(0.0, np.pi)
    length = rng.uniform(size * 0.4, size * 0.8)
    cy = rng.uniform(size * 0.3, size * 0.7)
    cx = rng.uniform(size * 0.3, size * 0.7)
    dx, dy = np.cos(angle), np.sin(angle)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    strength = rng.uniform(0.7, 1.0) * sign
    out = img.copy()
    steps = int(length * 2)
    for t in np.linspace(-length / 2.0, length / 2.0, steps):
        y = int(round(cy + t * dy))
        x = int(round(cx + t * dx))
        if 0 <= y < size and 0 <= x < size:
            out[y, x] += strength
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if 0 <= ny < size and 0 <= nx < size:
                        out[ny, nx] += 0.4 * strength
    return out


def make_sample(
    defective: bool, seed: int, size: int = IMG_SIZE
) -> np.ndarray:
    """Create one textured image. Returns a float array in [0, 1]."""
    rng = np.random.default_rng(seed)
    img = _base_texture(rng, size)
    if defective:
        if rng.random() < 0.5:
            img = _stamp_blob(img, rng, size)
        else:
            img = _stamp_scratch(img, rng, size)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def generate_dataset(
    n_per_class: int, seed: int = 0, size: int = IMG_SIZE
):
    """Build a balanced dataset of clean and defective images.

    Returns images of shape (2 * n_per_class, size, size) and integer labels
    where 1 means defective and 0 means clean.
    """
    images = []
    labels = []
    counter = seed * 100_000
    for _ in range(n_per_class):
        images.append(make_sample(False, counter, size))
        labels.append(0)
        counter += 1
        images.append(make_sample(True, counter, size))
        labels.append(1)
        counter += 1
    return np.stack(images), np.asarray(labels, dtype=np.int64)
