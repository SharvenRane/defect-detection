import numpy as np
import pytest

from src.data import generate_dataset
from src.model import DefectDetector
from src.review import route_predictions


def test_low_confidence_goes_to_review():
    # 0.5 is maximally uncertain, 0.99 and 0.01 are confident.
    probs = np.array([0.5, 0.99, 0.01, 0.52, 0.97])
    result = route_predictions(probs, confidence_threshold=0.75)
    review_idx = set(result.review_queue.indices)
    # The two borderline cases (0.5 and 0.52) must be reviewed.
    assert 0 in review_idx
    assert 3 in review_idx
    # The confident ones must be auto decided.
    assert set(result.auto_indices.tolist()) == {1, 2, 4}


def test_every_prediction_is_routed_exactly_once():
    probs = np.linspace(0.0, 1.0, 50)
    result = route_predictions(probs, confidence_threshold=0.8)
    assert result.n_auto + result.n_review == len(probs)
    all_idx = set(result.auto_indices.tolist()) | set(result.review_queue.indices)
    assert all_idx == set(range(len(probs)))


def test_review_queue_sorted_most_uncertain_first():
    probs = np.array([0.5, 0.6, 0.55])
    result = route_predictions(probs, confidence_threshold=0.99)
    confs = [it.confidence for it in result.review_queue.items]
    assert confs == sorted(confs)
    # The first item is the closest to 0.5.
    assert result.review_queue.items[0].index == 0


def test_threshold_validation():
    with pytest.raises(ValueError):
        route_predictions(np.array([0.5]), confidence_threshold=0.4)
    with pytest.raises(ValueError):
        route_predictions(np.array([0.5]), confidence_threshold=1.5)


def test_higher_threshold_sends_more_to_review():
    probs = np.linspace(0.0, 1.0, 100)
    low = route_predictions(probs, confidence_threshold=0.6)
    high = route_predictions(probs, confidence_threshold=0.95)
    assert high.n_review >= low.n_review


def test_end_to_end_review_captures_uncertain_cases():
    # Train a real model, then confirm the review queue genuinely holds the
    # least confident predictions and the auto stream is more confident.
    tr_x, tr_y = generate_dataset(n_per_class=80, seed=0)
    te_x, te_y = generate_dataset(n_per_class=40, seed=2)
    model = DefectDetector(seed=0).fit(tr_x, tr_y)
    probs = model.predict_proba(te_x)

    result = route_predictions(probs, confidence_threshold=0.75)
    conf = np.maximum(probs, 1.0 - probs)

    if result.n_review > 0 and result.n_auto > 0:
        review_conf = np.array(
            [it.confidence for it in result.review_queue.items]
        )
        auto_conf = conf[result.auto_indices]
        assert review_conf.max() < 0.75
        assert auto_conf.min() >= 0.75
        assert review_conf.mean() < auto_conf.mean()
