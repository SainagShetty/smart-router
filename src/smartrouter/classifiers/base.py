"""Classifier protocol: turn request features into a difficulty score in [0, 1].

Higher = harder = wants a more capable tier. Implementations may optionally
expose the embedding they used so the training store can persist it.
"""
from __future__ import annotations

from typing import Optional

from ..features import RequestFeatures

try:  # numpy is a core dep, but keep the type import soft for clarity.
    import numpy as np

    NDArray = "np.ndarray"
except Exception:  # pragma: no cover
    np = None  # type: ignore
    NDArray = "object"


class Classifier:
    """Base class. Subclasses must implement ``score``."""

    #: Stable identifier of the embedding space, persisted with each logged row
    #: so a retrain can reuse stored vectors (e.g. "hashing:1024", model name).
    embedding_model_id: str = "none"

    def score(self, features: RequestFeatures) -> float:
        raise NotImplementedError

    def embed(self, text: str):  # -> Optional[np.ndarray]
        """Return the embedding vector used for this text, or None.

        Persisted by the training store so the classifier head can be refit
        without re-embedding. Classifiers without an embedding return None.
        """
        return None

    def score_and_embed(self, features: RequestFeatures):
        """Return (score, embedding-or-None) computing the embedding once.

        Subclasses that embed should override to avoid embedding twice.
        """
        return self.score(features), None

    @staticmethod
    def clamp(x: float) -> float:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x
