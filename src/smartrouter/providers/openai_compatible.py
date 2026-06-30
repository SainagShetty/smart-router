"""OpenAI-compatible provider: covers OpenRouter, Ollama (/v1), and OpenAI direct.

All three speak POST /chat/completions with the same request/response shape, so
one adapter handles them; only base_url and auth differ.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List

import httpx

from ..config import ProviderConfig
from ..errors import ProviderError
from .base import Message, Provider

_DEFAULT_BASE = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
}


class OpenAICompatibleProvider(Provider):
    def __init__(self, name: str, config: ProviderConfig):
        super().__init__(name, config)
        self.base_url = (config.base_url or _DEFAULT_BASE[config.type]).rstrip("/")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self.config.resolve_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self.config.type == "openrouter":
            # Optional but recommended attribution headers.
            headers.setdefault("HTTP-Referer", "https://github.com/sainagshetty/smart-router")
            headers.setdefault("X-Title", "smartrouter")
        return headers

    def _payload(self, model, messages, params, stream):
        payload = {"model": model, "messages": list(messages)}
        payload.update(params or {})
        payload["stream"] = stream
        return payload

    def complete(self, model: str, messages: List[Message], **params) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                resp = client.post(
                    url, headers=self._headers(),
                    json=self._payload(model, messages, params, False),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{self.name} HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} request failed: {exc}") from exc

    def stream(self, model: str, messages: List[Message], **params) -> Iterator[Dict[str, Any]]:
        url = f"{self.base_url}/chat/completions"
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                with client.stream(
                    "POST", url, headers=self._headers(),
                    json=self._payload(model, messages, params, True),
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError:
                            continue
        except httpx.HTTPStatusError as exc:
            raise ProviderError(
                f"{self.name} HTTP {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} stream failed: {exc}") from exc
