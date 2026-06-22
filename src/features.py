"""Texture features for defect detection.

A defect is a local departure from an otherwise regular texture. So the most
informative signals are about the extremes and the local gradients rather than
the global mean. We compute a small fixed length feature vector per image:

  - mean and standard deviation of intensity
  - the min and max, and the full range
  - skewness and kurtosis of the intensity histogram
  - the 99th percentile of the absolute gradient magnitude, which spikes when
    a sharp blob edge or a scratch is present
  - the fraction of pixels that are strong outliers relative to the local
    median, which captures the localized nature of a defect

These are cheap, deterministic, and have no learned parameters, so the same
image always maps to the same vector.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

N_FEATURES = 9


def _single(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float64)
    flat = img.ravel()

    mean = flat.mean()
    std = flat.std()
    vmin = flat.min()
    vmax = flat.max()
    vrange = vmax - vmin

    if std > 1e-8:
        z = (flat - mean) / std
        skew = float(np.mean(z**3))
        kurt = float(np.mean(z**4))
    else:
        skew = 0.0
        kurt = 0.0

    gy, gx = np.gradient(img)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_p99 = float(np.percentile(grad_mag, 99))

    local_med = ndimage.median_filter(img, size=5)
    residual = np.abs(img - local_med)
    res_std = residual.std()
    if res_std > 1e-8:
        outlier_frac = float(np.mean(residual > (residual.mean() + 3.0 * res_std)))
    else:
        outlier_frac = 0.0

    return np.array(
        [mean, std, vmin, vmax, vrange, skew, kurt, grad_p99, outlier_frac],
        dtype=np.float64,
    )


def extract_features(img: np.ndarray) -> np.ndarray:
    """Feature vector of length N_FEATURES for a single 2D image."""
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {img.shape}")
    return _single(img)


def extract_features_batch(images: np.ndarray) -> np.ndarray:
    """Feature matrix of shape (n, N_FEATURES) for a stack of images."""
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError(
            f"expected a stack of 2D images, got shape {images.shape}"
        )
    return np.stack([_single(img) for img in images])
