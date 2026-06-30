"""Hardening tests for the smartrouter FastAPI server.

Covers the HTTP contract, request validation, top-level OpenAI param passthrough,
routing overrides over HTTP, auth, error mapping, streaming, feedback/stats, and
concurrency / thread-safety of the shared SQLite decision store.

Uses starlette's in-process TestClient only; never touches the live :4000 server.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smartrouter.server import create_app  # noqa: E402

from conftest import make_config, stub_providers  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _client(db_path=None, api_key=None, content="ok", fail_providers=()):
    app = create_app(make_config(db_path=db_path), api_key=api_key)
    stub_providers(app.state.core, content=content, fail_providers=fail_providers)
    return TestClient(app)


def _capturing_client(db_path=None, api_key=None, content="ok"):
    """A client whose every provider records the params it was called with.

    Returns (client, captured) where captured is a dict populated on each
    provider.complete call: {"model", "messages", "params"}.
    """
    app = create_app(make_config(db_path=db_path), api_key=api_key)
    core = app.state.core
    captured = {}

    def make(name):
        def complete(model, messages, **p):
            captured["provider"] = name
            captured["model"] = model
            captured["messages"] = messages
            captured["params"] = p
            return {
                "id": "x", "model": model,
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"total_tokens": 100},
            }
        return complete

    for name, prov in core.providers.items():
        prov.complete = make(name)
    return TestClient(app), captured


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# --------------------------------------------------------------------------- #
# request validation
# --------------------------------------------------------------------------- #
def test_missing_messages_returns_422():
    c = _client()
    r = c.post("/v1/chat/completions", json={"model": "gpt-4"})
    assert r.status_code == 422


def test_messages_not_a_list_returns_422():
    c = _client()
    r = c.post("/v1/chat/completions", json={"messages": "hello"})
    assert r.status_code == 422


def test_empty_messages_list_is_accepted():
    # An empty conversation is structurally valid; the router still decides a tier.
    c = _client()
    r = c.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 200
    assert r.json()["routing"]["tier"] == "local"


def test_unknown_extra_top_level_fields_accepted():
    # extra="allow" -> unknown top-level keys are captured, not rejected.
    c = _client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "user": "abc", "frequency_penalty": 0.2, "totally_made_up": True,
    })
    assert r.status_code == 200


def test_model_field_is_ignored():
    # The client-supplied model is accepted but the router picks its own.
    c = _client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "2+2?"}],
        "model": "gpt-4-turbo-please-use-this",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["routing"]["model"] == "llama3.1:8b"
    assert body["model"] != "gpt-4-turbo-please-use-this"


# --------------------------------------------------------------------------- #
# top-level OpenAI param passthrough
# --------------------------------------------------------------------------- #
def test_toplevel_params_reach_provider():
    c, captured = _capturing_client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3, "max_tokens": 256,
    })
    assert r.status_code == 200
    assert captured["params"].get("temperature") == 0.3
    assert captured["params"].get("max_tokens") == 256


def test_nested_params_reach_provider():
    c, captured = _capturing_client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "params": {"temperature": 0.7, "top_p": 0.9},
    })
    assert r.status_code == 200
    assert captured["params"].get("temperature") == 0.7
    assert captured["params"].get("top_p") == 0.9


def test_nested_params_win_on_conflict():
    # When the same key is sent both top-level and nested, nested `params` wins.
    c, captured = _capturing_client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.1,
        "params": {"temperature": 0.9},
    })
    assert r.status_code == 200
    assert captured["params"].get("temperature") == 0.9


# --------------------------------------------------------------------------- #
# routing overrides over HTTP
# --------------------------------------------------------------------------- #
_HARD = ("design a distributed lock-free rate limiter and prove linearizability "
         "and derive the complexity, optimize the algorithm step by step ") * 3


def test_hard_prompt_routes_frontier_by_default():
    c = _client()
    r = c.post("/route", json={"messages": [{"role": "user", "content": _HARD}]})
    assert r.status_code == 200
    assert r.json()["tier"] == "frontier"


def test_local_only_override_forces_local():
    c = _client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": _HARD}],
        "local_only": True,
    })
    assert r.status_code == 200
    assert r.json()["routing"]["tier"] == "local"


def test_force_tier_override():
    c = _client()
    r = c.post("/route", json={
        "messages": [{"role": "user", "content": "2+2?"}],
        "force_tier": "frontier",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "frontier"
    assert body["provider"] == "openrouter"


def test_cheap_only_caps_below_frontier():
    c = _client()
    r = c.post("/route", json={
        "messages": [{"role": "user", "content": _HARD}],
        "cheap_only": True,
    })
    assert r.status_code == 200
    # The same prompt routes frontier by default; cheap_only caps the ceiling.
    assert r.json()["tier"] == "cheap"


def test_local_only_with_image_returns_422():
    # No local model has vision -> NoEligibleModel -> 422.
    c = _client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}}]}],
        "local_only": True,
    })
    assert r.status_code == 422
    assert "local" in r.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# error mapping
# --------------------------------------------------------------------------- #
def test_provider_failure_returns_502():
    c = _client(fail_providers=("ollama", "openrouter"))
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 502


# --------------------------------------------------------------------------- #
# auth
# --------------------------------------------------------------------------- #
def test_health_open_when_key_set():
    c = _client(api_key="secret")
    assert c.get("/health").status_code == 200


def test_protected_endpoint_401_without_token():
    c = _client(api_key="secret")
    for path, payload in [
        ("/route", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/feedback", {"decision_id": "x", "label": 1}),
    ]:
        assert c.post(path, json=payload).status_code == 401
    assert c.get("/stats").status_code == 401


def test_protected_endpoint_200_with_correct_token():
    c = _client(api_key="secret")
    r = c.post("/route", headers=_auth("secret"),
               json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_protected_endpoint_401_with_wrong_token():
    c = _client(api_key="secret")
    r = c.post("/route", headers=_auth("wrong"),
               json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #
def _parse_sse(text):
    """Return (chunks, saw_done) from an SSE text body."""
    chunks, saw_done = [], False
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            saw_done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, saw_done


def test_streaming_returns_sse_with_content_and_done():
    c = _client(content="streamed-token")
    with c.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    chunks, saw_done = _parse_sse(text)
    assert saw_done
    assert any("streamed-token" in json.dumps(ch) for ch in chunks)


def test_streaming_chunks_carry_decision_id():
    c = _client(content="x")
    with c.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        text = "".join(r.iter_text())
    chunks, _ = _parse_sse(text)
    assert chunks, "expected at least one data chunk"
    ids = {ch.get("decision_id") for ch in chunks}
    assert None not in ids
    assert len(ids) == 1  # one decision, stamped on every chunk


# --------------------------------------------------------------------------- #
# /route makes no provider call
# --------------------------------------------------------------------------- #
def test_route_does_not_call_provider():
    app = create_app(make_config())
    called = {"complete": False, "stream": False}

    def boom_complete(*a, **k):
        called["complete"] = True
        raise AssertionError("provider.complete must not be called by /route")

    def boom_stream(*a, **k):
        called["stream"] = True
        raise AssertionError("provider.stream must not be called by /route")
        yield  # pragma: no cover

    for prov in app.state.core.providers.values():
        prov.complete, prov.stream = boom_complete, boom_stream

    c = TestClient(app)
    r = c.post("/route", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert called == {"complete": False, "stream": False}


# --------------------------------------------------------------------------- #
# feedback
# --------------------------------------------------------------------------- #
def test_feedback_known_id(tmp_db):
    c = _client(db_path=tmp_db)
    rid = c.post("/route", json={"messages": [{"role": "user", "content": "hi"}]}
                 ).json()["decision_id"]
    r = c.post("/feedback", json={"decision_id": rid, "label": 1})
    assert r.status_code == 200
    assert r.json() == {"updated": True}


def test_feedback_unknown_id_returns_404(tmp_db):
    c = _client(db_path=tmp_db)
    r = c.post("/feedback", json={"decision_id": "does-not-exist", "label": 1})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def test_stats_counts_increase_with_logging(tmp_db):
    c = _client(db_path=tmp_db)
    before = c.get("/stats").json()["total"]
    for _ in range(3):
        c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}]})
    after = c.get("/stats").json()
    assert after["total"] == before + 3
    assert "by_tier" in after


def test_stats_disabled_without_db():
    c = _client(db_path=None)  # logging.enabled is False
    r = c.get("/stats")
    assert r.status_code == 200
    assert r.json() == {"logging": "disabled"}


# --------------------------------------------------------------------------- #
# decision_id uniqueness
# --------------------------------------------------------------------------- #
def test_decision_ids_unique_across_requests():
    c = _client()
    ids = set()
    for _ in range(25):
        rid = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}]}).json()["decision_id"]
        ids.add(rid)
    assert len(ids) == 25


# --------------------------------------------------------------------------- #
# concurrency / thread-safety (shared SQLite decision store)
# --------------------------------------------------------------------------- #
def test_concurrent_requests_are_thread_safe(tmp_db):
    N, WORKERS = 100, 16
    c = _client(db_path=tmp_db)

    def fire(i):
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"hi {i}"}]})
        return r.status_code, r.json().get("decision_id")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(fire, range(N)))

    statuses = [s for s, _ in results]
    ids = [d for _, d in results]
    assert all(s == 200 for s in statuses), f"non-200 responses: {set(statuses)}"
    assert len(set(ids)) == N, "decision_ids collided under concurrency"

    # Every concurrent write must be durably and correctly persisted.
    stats = c.get("/stats").json()
    assert stats["total"] == N
