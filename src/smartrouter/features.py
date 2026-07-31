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
    # Shape of the request, so a caller's agent loop can be told apart from one
    # long chat turn. Both look identical in `text` (tool results are folded into
    # it), but they are different populations: in a loop the transcript grows with
    # every observation, so length-derived features saturate and stop carrying
    # information. Recorded for analysis only — nothing routes on them yet.
    n_tool_msgs: int = 0
    n_assistant_msgs: int = 0
    head_chars: int = 0  # first user message — the task that sets the difficulty ceiling
    tail_chars: int = 0  # newest tool result — the observation the next step reacts to
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
    n_tool_msgs = 0
    n_assistant_msgs = 0
    head_chars = 0
    tail_chars = 0

    for msg in messages:
        role = msg.get("role")
        if role == "tool":
            n_tool_msgs += 1
        elif role == "assistant":
            n_assistant_msgs += 1
        msg_chars = 0
        parts = _content_parts(msg.get("content"))
        for part in parts:
            if _part_is_image(part):
                has_images = True
                continue
            txt = _part_text(part)
            if txt:
                all_text_chars += len(txt)
                msg_chars += len(txt)
                if role in ("user", "tool", None):
                    user_texts.append(txt)
        # First user turn and newest tool result, measured per message rather than
        # from the flattened text — once joined the boundaries are unrecoverable.
        if role == "user" and head_chars == 0:
            head_chars = msg_chars
        if role == "tool":
            tail_chars = msg_chars

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
        n_tool_msgs=n_tool_msgs,
        n_assistant_msgs=n_assistant_msgs,
        head_chars=head_chars,
        tail_chars=tail_chars,
    )
