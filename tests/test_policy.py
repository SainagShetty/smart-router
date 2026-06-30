import pytest

from smartrouter.errors import NoEligibleModel
from smartrouter.policy import Overrides, decide

from conftest import make_config


@pytest.fixture
def cfg():
    return make_config()


def test_easy_routes_local(cfg):
    r = decide(cfg, cfg.models, 0.1)
    assert r.tier == "local"


def test_medium_routes_cheap_with_fallback_down(cfg):
    r = decide(cfg, cfg.models, 0.55)
    assert r.tier == "cheap"
    assert [m.id for m in r.fallback] == ["llama3.1:8b"]


def test_hard_routes_frontier(cfg):
    r = decide(cfg, cfg.models, 0.9)
    assert r.tier == "frontier"
    assert [m.id for m in r.fallback] == ["gpt-4o-mini", "llama3.1:8b"]


def test_local_only_overrides_high_score(cfg):
    r = decide(cfg, cfg.models, 0.95, Overrides(local_only=True))
    assert r.tier == "local"
    assert r.model.id == "llama3.1:8b"


def test_local_only_strict_raises_when_impossible(cfg):
    # Only cloud models survive (simulate an image request).
    cloud = [m for m in cfg.models if m.provider == "openrouter"]
    with pytest.raises(NoEligibleModel):
        decide(cfg, cloud, 0.2, Overrides(local_only=True))


def test_cheap_only_caps_below_frontier(cfg):
    r = decide(cfg, cfg.models, 0.95, Overrides(cheap_only=True))
    assert r.tier == "cheap"


def test_force_tier_ignores_score(cfg):
    r = decide(cfg, cfg.models, 0.05, Overrides(force_tier="frontier"))
    assert r.tier == "frontier"


def test_empty_candidates_raises(cfg):
    with pytest.raises(NoEligibleModel):
        decide(cfg, [], 0.5)


def test_overrides_disabled_when_policy_forbids():
    cfg = make_config(allow_overrides=False)
    r = decide(cfg, cfg.models, 0.95, Overrides(local_only=True))
    # override ignored -> high score still routes frontier
    assert r.tier == "frontier"


def test_max_cost_filters_expensive():
    cfg = make_config(max_cost_per_1k=0.001)
    r = decide(cfg, cfg.models, 0.95)
    # frontier (0.009) filtered out; cheap (0.0006) is the most capable left
    assert r.model.cost_per_1k <= 0.001
