"""Confidence based human review handoff.

In a real inspection line you do not want to auto accept or auto reject a part
when the model is unsure. Those borderline cases go to a human. This module
takes the model confidences and splits predictions into two streams: the ones
the model is confident about, which are decided automatically, and the low
confidence ones, which land in a review queue for a person to look at.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ReviewItem:
    index: int
    probability: float
    confidence: float
    predicted_label: int


@dataclass
class ReviewQueue:
    """Holds the cases routed to a human, ordered most uncertain first."""

    items: List[ReviewItem] = field(default_factory=list)

    def add(self, item: ReviewItem) -> None:
        self.items.append(item)

    def sort_by_uncertainty(self) -> None:
        self.items.sort(key=lambda it: it.confidence)

    @property
    def indices(self) -> List[int]:
        return [it.index for it in self.items]

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class RoutingResult:
    auto_indices: np.ndarray
    auto_labels: np.ndarray
    review_queue: ReviewQueue

    @property
    def n_auto(self) -> int:
        return int(len(self.auto_indices))

    @property
    def n_review(self) -> int:
        return len(self.review_queue)


def route_predictions(
    probabilities: np.ndarray,
    confidence_threshold: float = 0.75,
) -> RoutingResult:
    """Split predictions into auto decided and human review streams.

    A prediction whose confidence (max of p and 1 - p) is below the threshold
    is sent to the review queue. Everything else is decided automatically.
    """
    if not 0.5 <= confidence_threshold <= 1.0:
        raise ValueError(
            "confidence_threshold must lie in [0.5, 1.0], "
            f"got {confidence_threshold}"
        )

    probabilities = np.asarray(probabilities, dtype=np.float64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    labels = (probabilities >= 0.5).astype(np.int64)

    auto_mask = confidence >= confidence_threshold
    review_mask = ~auto_mask

    queue = ReviewQueue()
    for idx in np.flatnonzero(review_mask):
        queue.add(
            ReviewItem(
                index=int(idx),
                probability=float(probabilities[idx]),
                confidence=float(confidence[idx]),
                predicted_label=int(labels[idx]),
            )
        )
    queue.sort_by_uncertainty()

    return RoutingResult(
        auto_indices=np.flatnonzero(auto_mask),
        auto_labels=labels[auto_mask],
        review_queue=queue,
    )
