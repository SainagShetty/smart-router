"""Zero-dependency difficulty heuristic.

Used as a fallback when no embedding backend is available, or as a cheap
baseline. It is intentionally simple and transparent: a monotonic blend of
length, code/math presence, and a few "this needs reasoning" keyword signals.
"""
from __future__ import annotations

import re

from ..features import RequestFeatures
from .base import Classifier

_HARD_KEYWORDS = re.compile(
    r"\b(prove|derive|optimi[sz]e|debug|refactor|algorithm|complexity|"
    r"step[- ]by[- ]step|reason|explain why|trade[- ]?off|architect|"
    r"theorem|integral|differential|recursion|concurren|distributed)\b",
    re.IGNORECASE,
)


class HeuristicClassifier(Classifier):
    embedding_model_id = "none"

    def score(self, features: RequestFeatures) -> float:
        # Length signal: saturates around ~600 tokens.
        length = min(features.estimated_tokens / 600.0, 1.0)

        # Keyword signal: each distinct hard cue adds weight, capped.
        hits = len(set(m.lower() for m in _HARD_KEYWORDS.findall(features.text)))
        keyword = min(hits / 3.0, 1.0)

        code = min(features.code_ratio * 2.0, 1.0)

        # Conversation depth: long threads tend to need more capable models.
        depth = min(max(features.num_turns - 2, 0) / 8.0, 1.0)

        raw = 0.4 * length + 0.35 * keyword + 0.15 * code + 0.10 * depth
        return self.clamp(raw)
