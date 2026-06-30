"""RouterClient — an OpenAI-compatible drop-in.

    client = RouterClient(config)
    resp = client.chat.completions.create(messages=[...])
    print(resp.choices[0].message.content, resp.decision_id)

Same routing as RouterCore, but shaped like the OpenAI SDK so existing
OpenRouter/OpenAI code swaps in with one line. Per-request overrides
(force_tier / cheap_only / local_only) pass straight through.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence

from .config import RouterConfig
from .core import Decision, Message, RouterCore

_OVERRIDE_KEYS = ("force_tier", "cheap_only", "local_only", "tools", "response_format")


class _AttrDict(dict):
    """Dict with attribute access, recursively, for OpenAI-style ergonomics."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return _AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


class _Completions:
    def __init__(self, core: RouterCore):
        self._core = core

    def create(
        self,
        *,
        messages: Sequence[Message],
        stream: bool = False,
        model: Optional[str] = None,  # accepted & ignored: the router picks
        **kwargs,
    ):
        overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in _OVERRIDE_KEYS}

        if stream:
            chunks, decision = self._core.stream(messages, **overrides, **kwargs)
            return _stream_wrapper(chunks, decision)

        resp, decision = self._core.complete(messages, **overrides, **kwargs)
        wrapped = _wrap(resp)
        wrapped.decision_id = decision.decision_id
        wrapped.routing = _wrap({
            "tier": decision.tier, "model": decision.model,
            "score": decision.score, "reason": decision.reason,
        })
        return wrapped


def _stream_wrapper(chunks: Iterator[Dict[str, Any]], decision: Decision):
    for chunk in chunks:
        w = _wrap(chunk)
        w.decision_id = decision.decision_id
        yield w


class _Chat:
    def __init__(self, core: RouterCore):
        self.completions = _Completions(core)


class RouterClient:
    def __init__(self, config: RouterConfig):
        self._core = RouterCore(config)
        self.chat = _Chat(self._core)

    def route(self, messages: Sequence[Message], **kwargs) -> Decision:
        return self._core.route(messages, **kwargs)

    def feedback(self, decision_id: str, label: int, source: str = "manual") -> bool:
        return self._core.feedback(decision_id, label, source)

    def close(self) -> None:
        self._core.close()
