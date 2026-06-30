"""Deterministic capability gate.

Filters the model registry down to the models that *can* serve a request before
the difficulty classifier ever runs. Hard requirements always win: a request
with an image must go to a vision model regardless of how "easy" it looks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import ModelSpec
from .features import RequestFeatures


@dataclass
class GateResult:
    candidates: List[ModelSpec]
    rejected: List[str]  # human-readable "model_id: reason" for observability


def gate(models: List[ModelSpec], features: RequestFeatures) -> GateResult:
    candidates: List[ModelSpec] = []
    rejected: List[str] = []

    for m in models:
        reason = _reject_reason(m, features)
        if reason is None:
            candidates.append(m)
        else:
            rejected.append(f"{m.id}: {reason}")

    return GateResult(candidates=candidates, rejected=rejected)


def _reject_reason(m: ModelSpec, f: RequestFeatures):
    if f.has_images and not m.capabilities.vision:
        return "no vision support"
    if f.needs_tools and not m.capabilities.tools:
        return "no tool-calling support"
    if f.needs_json and not m.capabilities.json_mode:
        return "no JSON/structured-output mode"
    if f.estimated_tokens > m.context_window:
        return f"context window {m.context_window} < ~{f.estimated_tokens} tokens"
    return None
