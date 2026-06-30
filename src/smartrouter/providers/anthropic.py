"""Anthropic provider: translates to/from the Messages API and re-shapes the
response into the OpenAI Chat Completions form so callers see one shape.

Kept intentionally minimal (text in / text out + usage). Most personal setups
reach Anthropic models through OpenRouter; this adapter exists for direct use.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List

import httpx

from ..config import ProviderConfig
from ..errors import ProviderError
from .base import Message, Provider

_DEFAULT_BASE = "https://api.anthropic.com"


class AnthropicProvider(Provider):
    def __init__(self, name: str, config: ProviderConfig):
        super().__init__(name, config)
        self.base_url = (config.base_url or _DEFAULT_BASE).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        key = self.config.resolve_api_key()
        if not key:
            raise ProviderError(f"{self.name}: missing Anthropic API key")
        return {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }

    @staticmethod
    def _split(messages: List[Message]):
        system_parts, turns = [], []
        for m in messages:
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                system_parts.append(content if isinstance(content, str) else str(content))
            else:
                turns.append({"role": "assistant" if role == "assistant" else "user",
                              "content": content})
        return "\n".join(system_parts), turns

    def complete(self, model: str, messages: List[Message], **params) -> Dict[str, Any]:
        system, turns = self._split(messages)
        body = {
            "model": model,
            "messages": turns,
            "max_tokens": params.pop("max_tokens", 1024),
        }
        if system:
            body["system"] = system
        for k in ("temperature", "top_p", "stop_sequences"):
            if k in params:
                body[k] = params[k]
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/messages", headers=self._headers(), json=body
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{self.name} HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc

        text = "".join(
            b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
        )
        usage = data.get("usage", {})
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": data.get("model", model),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": data.get("stop_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
        }

    def stream(self, model: str, messages: List[Message], **params) -> Iterator[Dict[str, Any]]:
        # Minimal: emit the full completion as a single OpenAI-style chunk.
        result = self.complete(model, messages, **params)
        content = result["choices"][0]["message"]["content"]
        yield {
            "id": result["id"],
            "object": "chat.completion.chunk",
            "model": result["model"],
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
        }
