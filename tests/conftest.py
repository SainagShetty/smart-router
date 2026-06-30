import os
import tempfile

import pytest

from smartrouter import RouterConfig


def make_config(db_path=None, **policy):
    return RouterConfig.from_dict({
        "providers": {
            "ollama": {"type": "ollama", "base_url": "http://localhost:11434/v1",
                       "local": True},
            "openrouter": {"type": "openrouter", "api_key": "test"},
        },
        "tiers": [
            {"name": "local", "min_score": 0.0},
            {"name": "cheap", "min_score": 0.40},
            {"name": "frontier", "min_score": 0.75},
        ],
        "models": [
            {"id": "llama3.1:8b", "provider": "ollama", "tier": "local",
             "context_window": 8192, "capabilities": {"tools": True}},
            {"id": "gpt-4o-mini", "provider": "openrouter", "tier": "cheap",
             "cost_per_1k": 0.0006, "context_window": 128000,
             "capabilities": {"vision": True, "tools": True, "json": True}},
            {"id": "claude-3.5-sonnet", "provider": "openrouter", "tier": "frontier",
             "cost_per_1k": 0.009, "context_window": 200000,
             "capabilities": {"vision": True, "tools": True, "json": True}},
        ],
        "policy": policy or {},
        "logging": {"enabled": db_path is not None,
                    "db_path": db_path or "smartrouter.db"},
    })


@pytest.fixture
def config():
    return make_config()


@pytest.fixture
def tmp_db():
    path = os.path.join(tempfile.mkdtemp(), "sr.db")
    yield path


def stub_providers(core, content="ok", usage_tokens=100, fail_providers=()):
    """Replace each provider's complete()/stream() with a no-network stub.

    Providers whose name is in ``fail_providers`` raise ProviderError.
    """
    from smartrouter.errors import ProviderError

    def make(name):
        def complete(model, messages, **p):
            if name in fail_providers:
                raise ProviderError(f"stub failure: {name}")
            return {"id": "x", "model": model,
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {"total_tokens": usage_tokens}}

        def stream(model, messages, **p):
            if name in fail_providers:
                raise ProviderError(f"stub failure: {name}")
            yield {"choices": [{"delta": {"content": content}}]}

        return complete, stream

    for name, prov in core.providers.items():
        prov.complete, prov.stream = make(name)
