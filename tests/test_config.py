import pytest
from pydantic import ValidationError

from smartrouter import RouterConfig

from conftest import make_config


def test_valid_config_loads(config):
    assert config.tier_order() == ["local", "cheap", "frontier"]
    assert config.models_for_tier("local")[0].id == "llama3.1:8b"


def test_json_alias_accepted():
    cfg = make_config()
    m = [m for m in cfg.models if m.id == "gpt-4o-mini"][0]
    assert m.capabilities.json_mode is True


def test_unknown_tier_rejected():
    with pytest.raises(ValidationError):
        RouterConfig.from_dict({
            "providers": {"openrouter": {"type": "openrouter"}},
            "tiers": [{"name": "local"}],
            "models": [{"id": "m", "provider": "openrouter", "tier": "nope"}],
        })


def test_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        RouterConfig.from_dict({
            "providers": {"openrouter": {"type": "openrouter"}},
            "tiers": [{"name": "local"}],
            "models": [{"id": "m", "provider": "ghost", "tier": "local"}],
        })


def test_tiers_sorted_by_min_score():
    cfg = RouterConfig.from_dict({
        "providers": {"openrouter": {"type": "openrouter"}},
        "tiers": [
            {"name": "b", "min_score": 0.8},
            {"name": "a", "min_score": 0.0},
        ],
        "models": [{"id": "m", "provider": "openrouter", "tier": "a"}],
    })
    assert cfg.tier_order() == ["a", "b"]


def test_provider_is_local():
    cfg = make_config()
    assert cfg.providers["ollama"].is_local() is True
    assert cfg.providers["openrouter"].is_local() is False
