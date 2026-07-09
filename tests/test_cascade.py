"""Cascade mode: run the cheapest capable model first, validate cheaply, and
escalate up the tier ladder only when the answer is inadequate. Escalation
writes an implicit 'hard' label (reviving the training flywheel) and a
sensitive request must never escalate off-device."""
import sqlite3

import pytest

from smartrouter import RouterCore
from smartrouter.errors import ProviderError

from conftest import make_config


def stub_by_provider(core, mapping, usage_tokens=100):
    """Stub each provider to return mapping[name] as content; a missing/None
    entry raises ProviderError (simulates an unavailable provider)."""
    def make(name):
        def complete(model, messages, **p):
            content = mapping.get(name)
            if content is None:
                raise ProviderError(name)
            return {"model": model,
                    "choices": [{"message": {"role": "assistant", "content": content}}],
                    "usage": {"total_tokens": usage_tokens}}
        return complete
    for name, prov in core.providers.items():
        prov.complete = make(name)


def _label(db_path, decision_id):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT label, label_source FROM decisions WHERE decision_id=?",
            (decision_id,),
        ).fetchone()
    finally:
        conn.close()


def test_cascade_stays_cheap_when_answer_validates(config):
    core = RouterCore(config)
    stub_by_provider(core, {"ollama": "a good answer", "openrouter": "cloud"})
    # a hard-looking prompt would normally route to frontier; cascade tries local first
    resp, d = core.complete(
        [{"role": "user", "content":
          "prove the theorem and analyze the distributed algorithm"}],
        cascade=True,
    )
    assert resp["model"] == "llama3.1:8b"
    assert d.tier == "local"
    assert d.provider == "ollama"


def test_cascade_escalates_on_empty_answer(tmp_db):
    core = RouterCore(make_config(db_path=tmp_db))
    stub_by_provider(core, {"ollama": "  ", "openrouter": "a real cloud answer"})
    resp, d = core.complete([{"role": "user", "content": "2+2?"}], cascade=True)
    assert resp["choices"][0]["message"]["content"] == "a real cloud answer"
    assert resp["model"] == "gpt-4o-mini"  # cheapest capable above local
    assert _label(tmp_db, d.decision_id) == (1, "implicit:cascade_escalate")
    core.close()


def test_cascade_escalates_on_invalid_json(config):
    core = RouterCore(config)
    stub_by_provider(core, {"ollama": "not json at all", "openrouter": '{"x": 1}'})
    resp, d = core.complete(
        [{"role": "user", "content": "give me json"}],
        response_format={"type": "json_object"}, cascade=True,
    )
    assert resp["choices"][0]["message"]["content"] == '{"x": 1}'


def test_cascade_sensitive_never_escalates_off_device(config):
    core = RouterCore(config)
    # local answer is empty (would normally escalate), but cascade must NOT egress
    stub_by_provider(core, {"ollama": "", "openrouter": "cloud answer"})
    resp, d = core.complete(
        [{"role": "user", "content": "private data here"}],
        cascade=True, sensitive=True,
    )
    assert d.provider == "ollama"
    assert resp["model"] == "llama3.1:8b"  # best-effort local, never the cloud


def test_cascade_skips_failed_provider(config):
    core = RouterCore(config)
    # local provider down entirely -> cascade skips to cloud
    stub_by_provider(core, {"ollama": None, "openrouter": "cloud saved it"})
    resp, d = core.complete([{"role": "user", "content": "hi"}], cascade=True)
    assert resp["choices"][0]["message"]["content"] == "cloud saved it"


def test_cascade_off_uses_predicted_tier(config):
    core = RouterCore(config)
    stub_by_provider(core, {"ollama": "local", "openrouter": "cloud"})
    resp, d = core.complete(
        [{"role": "user", "content":
          "architect a fault-tolerant distributed system and prove correctness"}],
    )
    # no cascade: classifier prediction wins, escalates up front
    assert d.tier in ("cheap", "frontier")


def test_cascade_enabled_via_policy_config():
    cfg = make_config(cascade=True)
    core = RouterCore(cfg)
    stub_by_provider(core, {"ollama": "ok", "openrouter": "cloud"})
    resp, d = core.complete([{"role": "user", "content":
                             "prove and analyze the algorithm"}])
    assert d.tier == "local"  # policy default cascade -> tried local first
