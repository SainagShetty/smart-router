"""Extract cheap, deterministic features from a request.

These features feed two things: the capability gate (hard requirements like
"this request contains an image") and the difficulty classifier (it scores the
concatenated user text). Nothing here calls a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

Message = Dict[str, Any]

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_CJK = re.compile(r"[一-鿿぀-ヿ가-힯]")


@dataclass
class RequestFeatures:
    text: str  # concatenated user-authored text (classifier input)
    estimated_tokens: int
    has_images: bool
    needs_tools: bool
    needs_json: bool
    num_turns: int
    code_ratio: float
    has_cjk: bool
    raw_chars: int
    extra: Dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token). Good enough for context-window gating."""
    return max(1, (len(text) + 3) // 4)


def _content_parts(content: Any) -> List[Any]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _part_is_image(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    t = part.get("type", "")
    return t in ("image_url", "image", "input_image") or "image" in t


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict) and part.get("type") in (None, "text", "input_text"):
        return part.get("text", "") or ""
    return ""


def extract(
    messages: Sequence[Message],
    *,
    tools: Optional[Sequence[Any]] = None,
    response_format: Optional[Any] = None,
) -> RequestFeatures:
    user_texts: List[str] = []
    all_text_chars = 0
    has_images = False

    for msg in messages:
        parts = _content_parts(msg.get("content"))
        for part in parts:
            if _part_is_image(part):
                has_images = True
                continue
            txt = _part_text(part)
            if txt:
                all_text_chars += len(txt)
                if msg.get("role") in ("user", "tool", None):
                    user_texts.append(txt)

    text = "\n".join(user_texts).strip()
    # Fall back to all message text if there were no user turns (e.g. system-only).
    if not text:
        text = " ".join(
            _part_text(p)
            for m in messages
            for p in _content_parts(m.get("content"))
        ).strip()

    code_chars = sum(len(m.group(0)) for m in _CODE_FENCE.finditer(text))
    code_ratio = (code_chars / len(text)) if text else 0.0

    needs_json = False
    if response_format is not None:
        if isinstance(response_format, dict):
            needs_json = response_format.get("type") in ("json_object", "json_schema")
        else:
            needs_json = bool(response_format)

    return RequestFeatures(
        text=text,
        estimated_tokens=estimate_tokens(text or " ".join(str(m) for m in messages)),
        has_images=has_images,
        needs_tools=bool(tools),
        needs_json=needs_json,
        num_turns=len(messages),
        code_ratio=round(code_ratio, 4),
        has_cjk=bool(_CJK.search(text)),
        raw_chars=all_text_chars,
    )
