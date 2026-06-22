import numpy as np
import pytest

from src.data import make_sample
from src.features import (
    N_FEATURES,
    extract_features,
    extract_features_batch,
)


def test_feature_vector_length():
    img = make_sample(defective=False, seed=1)
    feats = extract_features(img)
    assert feats.shape == (N_FEATURES,)
    assert np.all(np.isfinite(feats))


def test_batch_feature_shape():
    imgs = np.stack([make_sample(False, s) for s in range(6)])
    feats = extract_features_batch(imgs)
    assert feats.shape == (6, N_FEATURES)
    assert np.all(np.isfinite(feats))


def test_features_are_deterministic():
    img = make_sample(defective=True, seed=5)
    assert np.array_equal(extract_features(img), extract_features(img))


def test_defect_raises_gradient_and_outlier_signal():
    # Averaged across many seeds the defect features should be larger,
    # because a defect adds a sharp local edge and outlier pixels.
    clean = np.stack([make_sample(False, 1000 + s) for s in range(40)])
    defect = np.stack([make_sample(True, 1000 + s) for s in range(40)])
    fc = extract_features_batch(clean)
    fd = extract_features_batch(defect)
    # index 7 is grad_p99, index 8 is outlier_frac
    assert fd[:, 7].mean() > fc[:, 7].mean()
    assert fd[:, 8].mean() > fc[:, 8].mean()


def test_bad_input_dimensions():
    with pytest.raises(ValueError):
        extract_features(np.zeros((4, 4, 4)))
    with pytest.raises(ValueError):
        extract_features_batch(np.zeros((4, 4)))
