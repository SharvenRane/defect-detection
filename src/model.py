"""The defect classifier.

We standardize the texture features and fit a logistic regression. Logistic
regression gives a calibrated probability for the positive class, which is
exactly what the confidence routing in review.py needs. The model is small,
trains in milliseconds on CPU, and is fully deterministic given a seed.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import extract_features_batch


class DefectDetector:
    """Fit on textured images, predict defective vs clean with confidence."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, random_state=seed)
        self._fitted = False

    def fit(self, images: np.ndarray, labels: np.ndarray) -> "DefectDetector":
        feats = extract_features_batch(images)
        feats = self.scaler.fit_transform(feats)
        self.clf.fit(feats, np.asarray(labels))
        self._fitted = True
        return self

    def _check(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before predicting")

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Probability of the defective class for each image."""
        self._check()
        feats = self.scaler.transform(extract_features_batch(images))
        return self.clf.predict_proba(feats)[:, 1]

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Hard label: 1 defective, 0 clean."""
        self._check()
        return (self.predict_proba(images) >= 0.5).astype(np.int64)

    def confidence(self, images: np.ndarray) -> np.ndarray:
        """Confidence in [0.5, 1.0]: distance of p from the 0.5 boundary.

        A probability near 0.5 means the model is unsure. We map that to a
        confidence near 0.5, and a probability near 0 or 1 to a confidence
        near 1.0.
        """
        p = self.predict_proba(images)
        return np.maximum(p, 1.0 - p)

    def score(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Accuracy on the given images."""
        preds = self.predict(images)
        return float(np.mean(preds == np.asarray(labels)))
