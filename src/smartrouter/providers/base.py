"""Provider protocol + factory.

Providers normalize to the OpenAI Chat Completions response shape (a dict), so
the drop-in client can return something familiar regardless of backend. No
LangChain; just httpx.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, List

from ..config import ProviderConfig
from ..errors import ProviderError

Message = Dict[str, Any]


class Provider:
    def __init__(self, name: str, config: ProviderConfig):
        self.name = name
        self.config = config

    def complete(self, model: str, messages: List[Message], **params) -> Dict[str, Any]:
        raise NotImplementedError

    def stream(self, model: str, messages: List[Message], **params) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError


def build_provider(name: str, config: ProviderConfig) -> Provider:
    from .anthropic import AnthropicProvider
    from .openai_compatible import OpenAICompatibleProvider

    if config.type in ("openrouter", "ollama", "openai"):
        return OpenAICompatibleProvider(name, config)
    if config.type == "anthropic":
        return AnthropicProvider(name, config)
    raise ProviderError(f"unknown provider type: {config.type!r}")
