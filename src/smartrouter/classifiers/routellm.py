"""Adapter for lm-sys/RouteLLM pretrained routers (optional extra).

Install with ``pip install smartrouter[routellm]``. RouteLLM's routers output a
"strong model win rate" for a prompt; we use that directly as the difficulty
score (higher win rate for the strong model => harder query).
"""
from __future__ import annotations

from ..features import RequestFeatures
from .base import Classifier


class RouteLLMClassifier(Classifier):
    def __init__(self, router: str = "mf", **kwargs):
        try:
            from routellm.controller import Controller
        except Exception as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "RouteLLMClassifier requires `pip install smartrouter[routellm]`"
            ) from exc

        self.router = router
        self.embedding_model_id = f"routellm:{router}"
        # strong/weak are placeholders; we only use the router's scoring head.
        self._controller = Controller(
            routers=[router],
            strong_model=kwargs.get("strong_model", "gpt-4o"),
            weak_model=kwargs.get("weak_model", "mixtral-8x7b"),
        )

    def score(self, features: RequestFeatures) -> float:
        prompt = features.text
        for attr in ("calculate_strong_win_rate", "_calculate_win_rate"):
            fn = getattr(self._controller, attr, None)
            if callable(fn):
                try:
                    return self.clamp(float(fn(prompt, self.router)))
                except TypeError:
                    return self.clamp(float(fn(prompt)))
        raise RuntimeError(
            "Unable to obtain a win-rate from this RouteLLM version; please open "
            "an issue with your routellm version."
        )
