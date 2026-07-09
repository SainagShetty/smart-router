"""Honest cost accounting: use the prompt/completion split the provider already
returns, not a single blended rate over total tokens."""
from smartrouter.config import ModelSpec
from smartrouter.core import _estimate_cost


def _resp(prompt=None, completion=None, total=None):
    usage = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if total is not None:
        usage["total_tokens"] = total
    return {"usage": usage}


def test_input_output_split_used_when_rates_set():
    spec = ModelSpec(id="m", provider="p", tier="t",
                     cost_per_1k_in=0.003, cost_per_1k_out=0.015)
    cost = _estimate_cost(_resp(prompt=1000, completion=200, total=1200), spec)
    # 1.0*0.003 + 0.2*0.015 = 0.006
    assert round(cost, 6) == 0.006


def test_blended_fallback_when_no_split_rates():
    spec = ModelSpec(id="m", provider="p", tier="t", cost_per_1k=0.0006)
    cost = _estimate_cost(_resp(total=1000), spec)
    assert round(cost, 6) == 0.0006


def test_split_rates_fall_back_to_blended_when_no_split_usage():
    # rates are per-direction but the provider only echoed total_tokens
    spec = ModelSpec(id="m", provider="p", tier="t",
                     cost_per_1k=0.01, cost_per_1k_in=0.003, cost_per_1k_out=0.015)
    cost = _estimate_cost(_resp(total=1000), spec)
    assert round(cost, 6) == 0.01


def test_none_when_no_pricing():
    spec = ModelSpec(id="m", provider="p", tier="t")
    assert _estimate_cost(_resp(total=1000), spec) is None


def test_none_when_no_usage():
    spec = ModelSpec(id="m", provider="p", tier="t", cost_per_1k=0.001)
    assert _estimate_cost({}, spec) is None
