"""Classifier implementations and a config-driven factory."""
from __future__ import annotations

from ..config import ClassifierConfig
from .base import Classifier
from .embedding import EmbeddingClassifier
from .heuristic import HeuristicClassifier

__all__ = [
    "Classifier",
    "EmbeddingClassifier",
    "HeuristicClassifier",
    "build_classifier",
]


def build_classifier(config: ClassifierConfig) -> Classifier:
    if config.kind == "heuristic":
        return HeuristicClassifier()

    if config.kind == "embedding":
        if config.model_path:
            return EmbeddingClassifier.load(config.model_path)
        if config.embedding_backend == "hashing":
            # The bundled head is trained on the hashing space; load it directly.
            return EmbeddingClassifier.bundled()
        # Sentence-transformer backend with no explicit head: caller must fit().
        return EmbeddingClassifier(
            backend=config.embedding_backend,
            embedding_model=config.embedding_model,
        )

    if config.kind == "routellm":
        from .routellm import RouteLLMClassifier

        return RouteLLMClassifier(router=config.routellm_router, **config.params)

    raise ValueError(f"unknown classifier kind: {config.kind!r}")
