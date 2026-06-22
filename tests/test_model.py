import numpy as np
import pytest

from src.data import generate_dataset
from src.model import DefectDetector


def _split():
    train_imgs, train_lbls = generate_dataset(n_per_class=80, seed=0)
    test_imgs, test_lbls = generate_dataset(n_per_class=40, seed=1)
    return train_imgs, train_lbls, test_imgs, test_lbls


def test_accuracy_beats_chance():
    tr_x, tr_y, te_x, te_y = _split()
    model = DefectDetector(seed=0).fit(tr_x, tr_y)
    acc = model.score(te_x, te_y)
    # Chance on a balanced two class problem is 0.5. Require a clear margin.
    assert acc > 0.8, f"accuracy {acc} did not clearly beat chance"


def test_proba_in_unit_interval():
    tr_x, tr_y, te_x, te_y = _split()
    model = DefectDetector(seed=0).fit(tr_x, tr_y)
    p = model.predict_proba(te_x)
    assert p.shape == (te_x.shape[0],)
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_confidence_bounds():
    tr_x, tr_y, te_x, te_y = _split()
    model = DefectDetector(seed=0).fit(tr_x, tr_y)
    conf = model.confidence(te_x)
    assert np.all(conf >= 0.5 - 1e-9) and np.all(conf <= 1.0 + 1e-9)


def test_predict_before_fit_raises():
    model = DefectDetector()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((1, 32, 32), dtype=np.float32))


def test_training_is_reproducible():
    tr_x, tr_y, te_x, _ = _split()
    p1 = DefectDetector(seed=0).fit(tr_x, tr_y).predict_proba(te_x)
    p2 = DefectDetector(seed=0).fit(tr_x, tr_y).predict_proba(te_x)
    assert np.allclose(p1, p2)
