"""Resilience and failure-injection tests for smartrouter.

Focus: provider fallback chains, malformed provider responses, classifier
degradation, streaming fallback, and that the training store records the right
post-fallback outcome. No network: everything is stubbed/monkeypatched.
"""
import json

import pytest

from smartrouter import RouterCore
from smartrouter.errors import ProviderError
from smartrouter.logging_ import TrainingStore

from conftest import make_config, stub_providers


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class FixedScore:
    """Classifier stub that returns a deterministic score (and no embedding).

    Lets a test pin the routed tier regardless of prompt text.
    """

    embedding_model_id = "fixed:test"

    def __init__(self, score: float):
        self._score = score

    def score_and_embed(self, feats):
        return self._score, None


def model_aware_stub(core, content="ok", usage_tokens=100, fail_ids=()):
    """Stub providers with *per-model-id* failure control.

    Unlike conftest.stub_providers (which fails by provider name), this fails
    only the specific model ids in ``fail_ids`` so a single provider can serve
    one model and fail another (cheap vs frontier both live on openrouter).
    """
    def make(_name):
        def complete(model, messages, **p):
            if model in fail_ids:
                raise ProviderError(f"stub failure: {model}")
            return {"id": "x", "model": model,
                    "choices": [{"message": {"role": "assistant",
                                             "content": content}}],
                    "usage": {"total_tokens": usage_tokens}}

        def stream(model, messages, **p):
            if model in fail_ids:
                raise ProviderError(f"stub failure: {model}")
            yield {"choices": [{"delta": {"content": content}}]}

        return complete, stream

    for _name, prov in core.providers.items():
        prov.complete, prov.stream = make(_name)


def install_complete(core, fn):
    """Point every provider's complete() at ``fn`` (model, messages, **p)->dict."""
    for prov in core.providers.values():
        prov.complete = fn


def read_row(store: TrainingStore, decision_id: str):
    """Read a decision row back (read-only) as a dict of the columns we assert on."""
    cur = store._conn.execute(
        "SELECT chosen_tier, chosen_model, cost, latency_ms, label, label_source "
        "FROM decisions WHERE decision_id=?",
        (decision_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    keys = ["chosen_tier", "chosen_model", "cost", "latency_ms",
            "label", "label_source"]
    return dict(zip(keys, row))


HARD = [{"role": "user", "content": "prove the theorem and analyze the algorithm"}]
EASY = [{"role": "user", "content": "hi"}]


# --------------------------------------------------------------------------
# fallback chain (complete)
# --------------------------------------------------------------------------

def test_chosen_tier_fails_falls_back_down_to_local():
    core = RouterCore(make_config())
    core.classifier = FixedScore(1.0)  # -> frontier (claude / openrouter)
    stub_providers(core, fail_providers=("openrouter",))
    resp, d = core.complete(HARD)
    # frontier+cheap are openrouter (fail); only local (ollama) survives.
    assert resp["model"] == "llama3.1:8b"
    assert d.tier == "frontier"  # the *decision* still reflects the routed tier


def test_first_fallback_in_chain_succeeds():
    core = RouterCore(make_config())
    core.classifier = FixedScore(1.0)  # frontier chosen
    # Fail only the chosen frontier model; the next link (cheap) answers.
    model_aware_stub(core, content="cheap-ans", fail_ids={"claude-3.5-sonnet"})
    resp, _ = core.complete(HARD)
    assert resp["model"] == "gpt-4o-mini"
    assert resp["choices"][0]["message"]["content"] == "cheap-ans"


def test_partial_chain_middle_fails_lower_succeeds():
    core = RouterCore(make_config())
    core.classifier = FixedScore(1.0)  # frontier chosen
    # chosen (frontier) and middle (cheap) both fail; lowest (local) succeeds.
    model_aware_stub(core, fail_ids={"claude-3.5-sonnet", "gpt-4o-mini"})
    resp, _ = core.complete(HARD)
    assert resp["model"] == "llama3.1:8b"


def test_all_providers_fail_raises_with_count():
    core = RouterCore(make_config())
    core.classifier = FixedScore(1.0)  # chain = frontier, cheap, local (len 3)
    stub_providers(core, fail_providers=("openrouter", "ollama"))
    with pytest.raises(ProviderError) as ei:
        core.complete(HARD)
    msg = str(ei.value)
    assert "3" in msg and "failed" in msg


def test_fallback_none_raises_immediately():
    calls = {"n": 0}
    core = RouterCore(make_config(fallback="none"))
    core.classifier = FixedScore(1.0)  # frontier

    def boom(model, messages, **p):
        calls["n"] += 1
        raise ProviderError("down")

    install_complete(core, boom)
    with pytest.raises(ProviderError) as ei:
        core.complete(HARD)
    # No fallback chain: exactly one provider attempt, message mentions count 1.
    assert calls["n"] == 1
    assert "1" in str(ei.value)


def test_fallback_up_escalates_from_local():
    core = RouterCore(make_config(fallback="up"))
    core.classifier = FixedScore(0.0)  # -> local (ollama)
    # local fails; escalate upward to the cheap tier.
    stub_providers(core, fail_providers=("ollama",), content="escalated")
    resp, d = core.complete(EASY)
    assert d.tier == "local"
    assert resp["model"] == "gpt-4o-mini"
    assert resp["choices"][0]["message"]["content"] == "escalated"


# --------------------------------------------------------------------------
# malformed provider responses
# --------------------------------------------------------------------------

def test_missing_usage_records_no_cost(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.5)  # -> cheap (gpt-4o-mini, a paying model)

    def no_usage(model, messages, **p):
        return {"id": "x", "model": model,
                "choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    install_complete(core, no_usage)
    resp, d = core.complete(EASY)
    assert resp["choices"][0]["message"]["content"] == "ok"
    row = read_row(core.store, d.decision_id)
    # paying model, but no usage -> cost must be None (not 0), and no crash.
    assert row["cost"] is None
    core.close()


def test_local_zero_cost_records_none(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.0)  # -> local (cost_per_1k == 0)
    stub_providers(core, usage_tokens=100)
    resp, d = core.complete(EASY)
    row = read_row(core.store, d.decision_id)
    assert row["chosen_model"] == "llama3.1:8b"
    # cost_per_1k is 0 for local -> _estimate_cost returns None.
    assert row["cost"] is None
    core.close()


def test_missing_choices_does_not_crash():
    core = RouterCore(make_config())
    core.classifier = FixedScore(0.0)
    payload = {"id": "x", "model": "llama3.1:8b", "usage": {"total_tokens": 3}}

    install_complete(core, lambda model, messages, **p: payload)
    resp, _ = core.complete(EASY)
    assert resp == payload  # returned verbatim, no choices needed


def test_missing_choices_json_first_content_none(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.5)  # json gates out local -> cheap

    # JSON requested, but response has no choices -> _first_content() is None.
    install_complete(
        core,
        lambda model, messages, **p: {"id": "x", "model": model,
                                      "usage": {"total_tokens": 3}},
    )
    resp, d = core.complete(EASY, response_format={"type": "json_object"})
    assert "choices" not in resp  # no crash, returned as-is
    row = read_row(core.store, d.decision_id)
    # No content to validate -> no implicit label applied.
    assert row["label"] is None
    core.close()


def test_empty_content_non_json_no_crash():
    core = RouterCore(make_config())
    core.classifier = FixedScore(0.0)
    stub_providers(core, content="")
    resp, _ = core.complete(EASY)
    assert resp["choices"][0]["message"]["content"] == ""


# --------------------------------------------------------------------------
# implicit JSON labeling
# --------------------------------------------------------------------------

def test_json_invalid_sets_implicit_label(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.5)  # cheap (has json capability)
    stub_providers(core, content="this is not json {")
    _, d = core.complete(EASY, response_format={"type": "json_object"})
    row = read_row(core.store, d.decision_id)
    assert row["label"] == 1
    assert row["label_source"] == "implicit:json_invalid"
    core.close()


def test_json_empty_content_sets_implicit_label(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.5)
    stub_providers(core, content="")  # "" is not valid JSON
    _, d = core.complete(EASY, response_format={"type": "json_object"})
    row = read_row(core.store, d.decision_id)
    assert row["label"] == 1
    assert row["label_source"] == "implicit:json_invalid"
    core.close()


def test_json_valid_no_implicit_label(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.5)
    stub_providers(core, content=json.dumps({"answer": 42}))
    _, d = core.complete(EASY, response_format={"type": "json_object"})
    row = read_row(core.store, d.decision_id)
    assert row["label"] is None
    assert row["label_source"] is None
    core.close()


# --------------------------------------------------------------------------
# classifier degradation
# --------------------------------------------------------------------------

def test_classifier_raises_degrades_to_cheapest_tier():
    core = RouterCore(make_config())

    class Boom:
        embedding_model_id = "boom"

        def score_and_embed(self, feats):
            raise RuntimeError("classifier down")

    core.classifier = Boom()
    stub_providers(core, content="served")
    resp, d = core.complete(HARD)  # even a "hard" prompt degrades to score 0
    assert d.tier == "local"
    assert d.score == 0.0
    assert resp["choices"][0]["message"]["content"] == "served"


def test_classifier_raises_then_serves_with_logging(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))

    class Boom:
        embedding_model_id = "boom"

        def score_and_embed(self, feats):
            raise ValueError("kaboom")

    core.classifier = Boom()
    stub_providers(core)
    _, d = core.complete(EASY)
    row = read_row(core.store, d.decision_id)
    assert row["chosen_tier"] == "local"
    core.close()


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------

def test_stream_fallback_before_first_chunk():
    core = RouterCore(make_config())
    core.classifier = FixedScore(1.0)  # frontier -> openrouter
    stub_providers(core, fail_providers=("openrouter",), content="from-local")
    gen, d = core.stream(HARD)
    chunks = list(gen)
    assert chunks
    assert chunks[0]["choices"][0]["delta"]["content"] == "from-local"
    assert d.tier == "frontier"


def test_stream_all_providers_fail_raises():
    core = RouterCore(make_config())
    core.classifier = FixedScore(1.0)
    stub_providers(core, fail_providers=("openrouter", "ollama"))
    with pytest.raises(ProviderError) as ei:
        core.stream(HARD)
    assert "3" in str(ei.value)


def test_stream_records_chosen_model_after_fallback(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(1.0)  # frontier chosen
    stub_providers(core, fail_providers=("openrouter",), content="x")
    gen, d = core.stream(HARD)
    list(gen)  # outcome is recorded only after the stream is fully consumed
    row = read_row(core.store, d.decision_id)
    assert row["chosen_model"] == "llama3.1:8b"  # post-fallback survivor
    assert row["latency_ms"] is not None
    core.close()


# --------------------------------------------------------------------------
# outcome recording (complete)
# --------------------------------------------------------------------------

def test_update_outcome_latency_and_model_after_fallback(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(1.0)  # frontier chosen
    stub_providers(core, fail_providers=("openrouter",))
    _, d = core.complete(HARD)
    row = read_row(core.store, d.decision_id)
    # Routed frontier, but local answered post-fallback: the persisted row
    # tracks the model AND tier that actually served, so stats stay consistent.
    assert row["chosen_tier"] == "local"
    assert row["chosen_model"] == "llama3.1:8b"
    assert row["latency_ms"] is not None
    core.close()


def test_cost_recorded_when_usage_present(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.5)  # cheap: gpt-4o-mini, cost_per_1k=0.0006
    stub_providers(core, usage_tokens=100)
    _, d = core.complete(EASY)
    row = read_row(core.store, d.decision_id)
    assert row["chosen_model"] == "gpt-4o-mini"
    assert row["cost"] == pytest.approx(100 / 1000.0 * 0.0006)
    core.close()


# --------------------------------------------------------------------------
# feedback
# --------------------------------------------------------------------------

def test_feedback_unknown_decision_returns_false(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    assert core.feedback("does-not-exist", 1) is False
    core.close()


def test_feedback_known_returns_true_and_overwrites(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.0)
    stub_providers(core)
    _, d = core.complete(EASY)

    assert core.feedback(d.decision_id, 0, source="manual") is True
    assert read_row(core.store, d.decision_id)["label"] == 0
    # overwrite with a different label
    assert core.feedback(d.decision_id, 1, source="manual2") is True
    row = read_row(core.store, d.decision_id)
    assert row["label"] == 1
    assert row["label_source"] == "manual2"
    core.close()


def test_feedback_overwrites_implicit_label(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    core.classifier = FixedScore(0.5)
    stub_providers(core, content="not json")
    _, d = core.complete(EASY, response_format={"type": "json_object"})
    # implicit label present...
    assert read_row(core.store, d.decision_id)["label"] == 1
    # explicit feedback clobbers it.
    assert core.feedback(d.decision_id, 0, source="human") is True
    row = read_row(core.store, d.decision_id)
    assert row["label"] == 0
    assert row["label_source"] == "human"
    core.close()


def test_feedback_without_store_returns_false():
    core = RouterCore(make_config())  # logging disabled (no db_path)
    assert core.store is None
    assert core.feedback("anything", 1) is False
