import numpy as np

from src.data import generate_dataset, make_sample, IMG_SIZE


def test_sample_shape_and_range():
    img = make_sample(defective=False, seed=1)
    assert img.shape == (IMG_SIZE, IMG_SIZE)
    assert img.min() >= 0.0
    assert img.max() <= 1.0
    assert img.dtype == np.float32


def test_sample_is_deterministic():
    a = make_sample(defective=True, seed=7)
    b = make_sample(defective=True, seed=7)
    assert np.array_equal(a, b)


def test_defect_changes_the_image():
    clean = make_sample(defective=False, seed=42)
    defective = make_sample(defective=True, seed=42)
    assert not np.array_equal(clean, defective)


def test_dataset_is_balanced_and_labeled():
    images, labels = generate_dataset(n_per_class=10, seed=3)
    assert images.shape == (20, IMG_SIZE, IMG_SIZE)
    assert labels.shape == (20,)
    assert int((labels == 0).sum()) == 10
    assert int((labels == 1).sum()) == 10


def test_dataset_seed_reproducible():
    i1, l1 = generate_dataset(n_per_class=5, seed=11)
    i2, l2 = generate_dataset(n_per_class=5, seed=11)
    assert np.array_equal(i1, i2)
    assert np.array_equal(l1, l2)
