import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from smartrouter.server import create_app  # noqa: E402

from conftest import make_config, stub_providers  # noqa: E402


def _client(db_path=None, api_key=None, content="hi"):
    app = create_app(make_config(db_path=db_path), api_key=api_key)
    stub_providers(app.state.core, content=content)
    return TestClient(app)


def test_health():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["tiers"] == ["local", "cheap", "frontier"]


def test_route_endpoint():
    c = _client()
    r = c.post("/route", json={"messages": [{"role": "user", "content": "what is 2+2?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["tier"] == "local"
    assert body["decision_id"]


def test_chat_completions_openai_shape():
    c = _client(content="42")
    r = c.post("/v1/chat/completions",
               json={"messages": [{"role": "user", "content": "2+2?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "42"
    assert body["routing"]["tier"] == "local"
    assert body["decision_id"]


def test_local_only_override_via_api():
    c = _client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content":
                      "design a distributed lock-free rate limiter and prove linearizability"}],
        "local_only": True,
    })
    assert r.status_code == 200
    assert r.json()["routing"]["tier"] == "local"


def test_streaming_sse():
    c = _client(content="streamed")
    with c.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "stream": True}) as r:
        assert r.status_code == 200
        text = "".join(r.iter_text())
    assert "streamed" in text
    assert "[DONE]" in text


def test_feedback_endpoint(tmp_db):
    c = _client(db_path=tmp_db)
    rid = c.post("/route", json={"messages": [{"role": "user", "content": "hi"}]}).json()["decision_id"]
    r = c.post("/feedback", json={"decision_id": rid, "label": 1})
    assert r.status_code == 200 and r.json()["updated"] is True
    miss = c.post("/feedback", json={"decision_id": "nope", "label": 1})
    assert miss.status_code == 404


def test_stats_endpoint(tmp_db):
    c = _client(db_path=tmp_db)
    c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    r = c.get("/stats")
    assert r.status_code == 200
    assert r.json()["total"] >= 1
    assert "by_tier" in r.json()


def test_auth_required_when_key_set():
    c = _client(api_key="secret")
    # health is open
    assert c.get("/health").status_code == 200
    # protected without token
    assert c.post("/route", json={"messages": [{"role": "user", "content": "hi"}]}
                  ).status_code == 401
    # with token
    ok = c.post("/route", headers={"Authorization": "Bearer secret"},
                json={"messages": [{"role": "user", "content": "hi"}]})
    assert ok.status_code == 200


def test_no_eligible_model_returns_422():
    c = _client()
    # an image request with tools forcing nothing eligible? image needs vision;
    # local lacks it but cloud has it -> still eligible. Force impossible instead:
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x"}}]}],
        "local_only": True,  # no local vision model -> NoEligibleModel
    })
    assert r.status_code == 422
