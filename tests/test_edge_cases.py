"""Edge-case and boundary hardening for smartrouter.

Covers feature extraction corner cases, the capability gate's exact boundaries,
policy tier-band edges, classifier guarantees, and config validation.
"""
import pytest
from pydantic import ValidationError

from smartrouter import RouterConfig
from smartrouter import capabilities
from smartrouter.classifiers.base import Classifier
from smartrouter.classifiers.embedding import EmbeddingClassifier
from smartrouter.classifiers.heuristic import HeuristicClassifier
from smartrouter.errors import NoEligibleModel
from smartrouter.features import RequestFeatures, estimate_tokens, extract
from smartrouter.policy import Overrides, decide

from conftest import make_config


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _feat(**over):
    """Build a RequestFeatures with explicit, controllable values."""
    base = dict(
        text="x",
        estimated_tokens=1,
        has_images=False,
        needs_tools=False,
        needs_json=False,
        num_turns=1,
        code_ratio=0.0,
        has_cjk=False,
        raw_chars=1,
    )
    base.update(over)
    return RequestFeatures(**base)


def _cfg(tiers, models, **policy):
    return RouterConfig.from_dict({
        "providers": {
            "ollama": {"type": "ollama", "base_url": "http://x/v1", "local": True},
            "openrouter": {"type": "openrouter", "api_key": "test"},
        },
        "tiers": tiers,
        "models": models,
        "policy": policy,
        "logging": {"enabled": False},
    })


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

def test_extract_empty_messages():
    f = extract([])
    assert f.num_turns == 0
    assert f.text == ""
    assert f.estimated_tokens >= 1  # never zero
    assert not f.has_images and not f.needs_tools and not f.needs_json
    assert f.raw_chars == 0


def test_extract_system_only_falls_back_to_all_text():
    # No user/tool turns -> text falls back to concatenated message text.
    f = extract([{"role": "system", "content": "You are a helpful CJK-free bot."}])
    assert "helpful" in f.text
    assert f.num_turns == 1


def test_extract_mixed_text_and_image_parts():
    f = extract([{"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        {"type": "input_text", "text": "in detail"},
    ]}])
    assert f.has_images is True
    assert "describe this" in f.text
    assert "in detail" in f.text


def test_extract_content_none_does_not_crash():
    f = extract([{"role": "user", "content": None},
                 {"role": "assistant", "content": None}])
    assert f.text == ""
    assert f.num_turns == 2
    assert f.has_images is False


def test_extract_long_prompt_token_estimate():
    n = 40000
    f = extract([{"role": "user", "content": "a" * n}])
    # ~4 chars/token.
    assert f.estimated_tokens == estimate_tokens("a" * n)
    assert abs(f.estimated_tokens - n // 4) <= 1
    assert f.raw_chars == n


def test_extract_cjk_detection():
    cjk = extract([{"role": "user", "content": "请解释这个算法"}])
    assert cjk.has_cjk is True
    ascii_only = extract([{"role": "user", "content": "explain this algorithm"}])
    assert ascii_only.has_cjk is False


def test_extract_code_ratio_multiple_fences():
    text = "intro ```code one``` middle ```code two``` end"
    f = extract([{"role": "user", "content": text}])
    # Both fenced spans count toward code_chars.
    assert 0.0 < f.code_ratio < 1.0
    # Sanity: a no-code prompt is 0.
    assert extract([{"role": "user", "content": "no code here"}]).code_ratio == 0.0


def test_extract_tool_message_text_included():
    f = extract([
        {"role": "user", "content": "call the tool"},
        {"role": "tool", "content": "TOOL_RESULT_42"},
    ])
    assert "TOOL_RESULT_42" in f.text


def test_extract_response_format_variants():
    assert extract([{"role": "user", "content": "x"}],
                   response_format={"type": "json_object"}).needs_json is True
    assert extract([{"role": "user", "content": "x"}],
                   response_format={"type": "json_schema",
                                    "json_schema": {}}).needs_json is True
    assert extract([{"role": "user", "content": "x"}],
                   response_format={"type": "text"}).needs_json is False
    assert extract([{"role": "user", "content": "x"}],
                   response_format=None).needs_json is False


def test_estimate_tokens_floor_and_ratio():
    assert estimate_tokens("") == 1  # floor of 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 32768) == 8192


# --------------------------------------------------------------------------
# capability gate
# --------------------------------------------------------------------------

def test_gate_context_window_exact_boundary_passes(config):
    # estimated_tokens == window must pass (reject is strictly '>').
    f = _feat(estimated_tokens=8192)  # llama window is 8192
    res = capabilities.gate(config.models, f)
    assert "llama3.1:8b" in {m.id for m in res.candidates}


def test_gate_context_window_over_by_one_fails(config):
    f = _feat(estimated_tokens=8193)
    res = capabilities.gate(config.models, f)
    assert "llama3.1:8b" not in {m.id for m in res.candidates}
    assert any("context window" in r and "llama3.1:8b" in r for r in res.rejected)


def test_gate_image_plus_json_requires_both(config):
    f = _feat(has_images=True, needs_json=True)
    res = capabilities.gate(config.models, f)
    ids = {m.id for m in res.candidates}
    # local model lacks both vision and json -> rejected.
    assert "llama3.1:8b" not in ids
    assert ids == {"gpt-4o-mini", "claude-3.5-sonnet"}


def test_gate_all_rejected_returns_empty_with_reasons(config):
    # Exceed even the largest context window (frontier = 200000).
    f = _feat(estimated_tokens=500000)
    res = capabilities.gate(config.models, f)
    assert res.candidates == []
    assert len(res.rejected) == len(config.models)


# --------------------------------------------------------------------------
# policy bands
# --------------------------------------------------------------------------

def test_policy_min_score_boundary_inclusive(config):
    # cheap tier min_score is exactly 0.40; score == min_score selects it.
    assert decide(config, config.models, 0.40).tier == "cheap"
    assert decide(config, config.models, 0.75).tier == "frontier"


def test_policy_score_zero_and_one(config):
    assert decide(config, config.models, 0.0).tier == "local"
    assert decide(config, config.models, 1.0).tier == "frontier"


def test_policy_single_tier_config():
    cfg = _cfg(
        [{"name": "only", "min_score": 0.0}],
        [{"id": "m", "provider": "openrouter", "tier": "only",
          "cost_per_1k": 0.001, "context_window": 100000}],
    )
    assert decide(cfg, cfg.models, 0.0).tier == "only"
    assert decide(cfg, cfg.models, 0.99).tier == "only"


def test_policy_fallback_up_vs_none():
    up = make_config(fallback="up")
    r = decide(up, up.models, 0.05)  # routes local
    assert r.tier == "local"
    assert [m.id for m in r.fallback] == ["gpt-4o-mini", "claude-3.5-sonnet"]

    none = make_config(fallback="none")
    r2 = decide(none, none.models, 0.05)
    assert r2.fallback == []


def test_policy_cheap_only_two_tier_config():
    cfg = _cfg(
        [{"name": "local", "min_score": 0.0},
         {"name": "frontier", "min_score": 0.75}],
        [{"id": "llama", "provider": "ollama", "tier": "local",
          "context_window": 100000, "capabilities": {"tools": True}},
         {"id": "claude", "provider": "openrouter", "tier": "frontier",
          "cost_per_1k": 0.01, "context_window": 200000,
          "capabilities": {"vision": True, "tools": True, "json": True}}],
    )
    # High score would pick frontier; cheap_only caps below top tier -> local.
    r = decide(cfg, cfg.models, 0.95, Overrides(cheap_only=True))
    assert r.tier == "local"


def test_policy_force_tier_no_candidate_raises(config):
    local_only_pool = [m for m in config.models if m.tier == "local"]
    with pytest.raises(NoEligibleModel):
        decide(config, local_only_pool, 0.5, Overrides(force_tier="frontier"))


def test_policy_max_cost_filters_all_but_cheapest():
    cfg = make_config(max_cost_per_1k=0.0007)  # keeps llama(0) + gpt(0.0006)
    r = decide(cfg, cfg.models, 0.95)  # would be frontier(0.009) but it's filtered
    assert r.tier == "cheap"
    assert r.model.cost_per_1k <= 0.0007


def test_policy_max_cost_filters_everything_keeps_pool():
    # Cap below every model's cost: filter yields nothing, so pool is kept intact.
    cfg = make_config(max_cost_per_1k=-1.0)
    r = decide(cfg, cfg.models, 0.95)
    assert r.tier == "frontier"  # unfiltered routing still works


def test_policy_local_only_keeps_local(config):
    r = decide(config, config.models, 0.95, Overrides(local_only=True))
    assert r.tier == "local" and r.model.provider == "ollama"


def test_policy_local_only_no_local_candidate_raises(config):
    cloud = [m for m in config.models if m.provider == "openrouter"]
    with pytest.raises(NoEligibleModel):
        decide(config, cloud, 0.2, Overrides(local_only=True))


def test_policy_default_tier_used_when_score_below_all_bands():
    cfg = _cfg(
        [{"name": "cheap", "min_score": 0.30},
         {"name": "mid", "min_score": 0.60},
         {"name": "top", "min_score": 0.90}],
        [{"id": "c", "provider": "openrouter", "tier": "cheap",
          "cost_per_1k": 0.001, "context_window": 100000},
         {"id": "m", "provider": "openrouter", "tier": "mid",
          "cost_per_1k": 0.005, "context_window": 100000},
         {"id": "t", "provider": "openrouter", "tier": "top",
          "cost_per_1k": 0.01, "context_window": 100000}],
        default_tier="mid",
    )
    # 0.1 is below every tier's min_score -> spec says fall back to default_tier.
    assert decide(cfg, cfg.models, 0.1).tier == "mid"


# --------------------------------------------------------------------------
# classifier
# --------------------------------------------------------------------------

def test_classifier_clamp_bounds():
    assert Classifier.clamp(-5.0) == 0.0
    assert Classifier.clamp(5.0) == 1.0
    assert Classifier.clamp(0.42) == 0.42


def test_heuristic_score_in_range_and_deterministic():
    f = extract([{"role": "user",
                  "content": "prove the theorem and optimize this algorithm"}])
    h = HeuristicClassifier()
    s1, s2 = h.score(f), h.score(f)
    assert 0.0 <= s1 <= 1.0
    assert s1 == s2  # deterministic


def test_heuristic_embed_is_none():
    h = HeuristicClassifier()
    assert h.embed("anything") is None
    score, emb = h.score_and_embed(extract([{"role": "user", "content": "hi"}]))
    assert emb is None
    assert 0.0 <= score <= 1.0


def test_bundled_embedding_model_id_stable():
    a = EmbeddingClassifier.bundled()
    b = EmbeddingClassifier.bundled()
    assert a.embedding_model_id == b.embedding_model_id
    assert isinstance(a.embedding_model_id, str) and a.embedding_model_id


def test_bundled_classifier_deterministic_and_clamped():
    c = EmbeddingClassifier.bundled()
    f = extract([{"role": "user", "content": "design a distributed system"}])
    s1, e1 = c.score_and_embed(f)
    s2, _ = c.score_and_embed(f)
    assert s1 == s2
    assert 0.0 <= s1 <= 1.0
    assert e1 is not None


# --------------------------------------------------------------------------
# config validation
# --------------------------------------------------------------------------

def test_config_empty_models_rejected():
    with pytest.raises(ValidationError):
        _cfg([{"name": "t", "min_score": 0.0}], [])


def test_config_empty_tiers_rejected():
    with pytest.raises(ValidationError):
        _cfg([], [{"id": "m", "provider": "openrouter", "tier": "t"}])


def test_config_duplicate_tier_names_rejected():
    with pytest.raises(ValidationError):
        _cfg(
            [{"name": "dup", "min_score": 0.0}, {"name": "dup", "min_score": 0.5}],
            [{"id": "m", "provider": "openrouter", "tier": "dup"}],
        )


def test_config_unknown_tier_rejected():
    with pytest.raises(ValidationError):
        _cfg(
            [{"name": "real", "min_score": 0.0}],
            [{"id": "m", "provider": "openrouter", "tier": "ghost"}],
        )


def test_config_unknown_provider_rejected():
    with pytest.raises(ValidationError):
        _cfg(
            [{"name": "real", "min_score": 0.0}],
            [{"id": "m", "provider": "nope", "tier": "real"}],
        )


def test_config_default_tier_unknown_rejected():
    with pytest.raises(ValidationError):
        _cfg(
            [{"name": "real", "min_score": 0.0}],
            [{"id": "m", "provider": "openrouter", "tier": "real"}],
            default_tier="ghost",
        )


def test_config_tiers_autosorted_by_min_score():
    cfg = _cfg(
        [{"name": "frontier", "min_score": 0.75},
         {"name": "local", "min_score": 0.0},
         {"name": "cheap", "min_score": 0.40}],
        [{"id": "m", "provider": "openrouter", "tier": "cheap",
          "context_window": 100000}],
    )
    assert cfg.tier_order() == ["local", "cheap", "frontier"]
