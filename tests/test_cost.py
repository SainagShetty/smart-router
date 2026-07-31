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


# ---- streaming: cost must be recorded, not silently dropped ------------------

import sqlite3

from smartrouter import RouterCore
from smartrouter.config import ProviderConfig
from smartrouter.providers.openai_compatible import OpenAICompatibleProvider

from conftest import make_config, stub_providers


def _stub_stream_with_usage(core, usage, content="hi"):
    """Stream stub that emits a trailing usage-only chunk, as providers do when
    stream_options.include_usage is set."""
    def make(_name):
        def stream(model, messages, **p):
            yield {"choices": [{"delta": {"content": content}}]}
            yield {"choices": [], "usage": usage}
        return stream

    for name, prov in core.providers.items():
        prov.stream = make(name)


def _cost_of(db_path, decision_id):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT cost, chosen_tier FROM decisions WHERE decision_id=?",
            (decision_id,),
        ).fetchone()
    finally:
        conn.close()


def test_streaming_records_cost_from_trailing_usage_chunk(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    stub_providers(core)
    _stub_stream_with_usage(core, {"total_tokens": 1000})
    chunks, d = core.stream([{"role": "user", "content": "hi"}], force_tier="cheap")
    list(chunks)  # drain — cost is written when the generator finishes
    core.close()
    cost, tier = _cost_of(tmp_db, d.decision_id)
    assert cost is not None, "streamed call recorded no cost"
    assert round(cost, 6) == 0.0006  # gpt-4o-mini @ 0.0006/1k over 1000 tokens
    assert tier == "cheap"


def test_streaming_cost_is_none_when_provider_sends_no_usage(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    stub_providers(core)  # default stub yields no usage chunk
    chunks, d = core.stream([{"role": "user", "content": "hi"}], force_tier="cheap")
    list(chunks)
    core.close()
    cost, _ = _cost_of(tmp_db, d.decision_id)
    assert cost is None


def test_stream_options_requested_for_paid_provider():
    prov = OpenAICompatibleProvider(
        "openrouter", ProviderConfig(type="openrouter", api_key="x")
    )
    payload = prov._payload("m", [{"role": "user", "content": "hi"}], {}, True)
    assert payload["stream_options"] == {"include_usage": True}


def test_stream_options_not_sent_to_local_provider():
    # Ollama has no per-token rate, and not every local server accepts the field.
    prov = OpenAICompatibleProvider("ollama", ProviderConfig(type="ollama"))
    payload = prov._payload("m", [{"role": "user", "content": "hi"}], {}, True)
    assert "stream_options" not in payload


def test_stream_options_absent_on_non_streaming_calls():
    prov = OpenAICompatibleProvider(
        "openrouter", ProviderConfig(type="openrouter", api_key="x")
    )
    payload = prov._payload("m", [{"role": "user", "content": "hi"}], {}, False)
    assert "stream_options" not in payload


def test_caller_supplied_stream_options_wins():
    prov = OpenAICompatibleProvider(
        "openrouter", ProviderConfig(type="openrouter", api_key="x")
    )
    payload = prov._payload(
        "m", [{"role": "user", "content": "hi"}],
        {"stream_options": {"include_usage": False}}, True,
    )
    assert payload["stream_options"] == {"include_usage": False}
