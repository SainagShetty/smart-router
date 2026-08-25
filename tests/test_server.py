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


def _loopback_client(api_key="secret", trust_loopback=True):
    """TestClient that presents itself as a direct on-box caller."""
    app = create_app(
        make_config(), api_key=api_key, trust_loopback=trust_loopback
    )
    stub_providers(app.state.core)
    return TestClient(app, client=("127.0.0.1", 54321))


def test_loopback_exempt_from_auth():
    # Co-located services (resume-bot, jarvis, ...) send no Authorization header.
    c = _loopback_client()
    r = c.post("/route", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_proxied_loopback_still_requires_token():
    # tailscale serve / Caddy connect over loopback but set a forwarding header;
    # those requests came from off-box and must authenticate.
    c = _loopback_client()
    for headers in (
        {"X-Forwarded-For": "100.126.90.22"},
        {"Forwarded": "for=100.126.90.22"},
    ):
        assert c.post(
            "/route",
            headers=headers,
            json={"messages": [{"role": "user", "content": "hi"}]},
        ).status_code == 401

    ok = c.post(
        "/route",
        headers={"X-Forwarded-For": "100.126.90.22", "Authorization": "Bearer secret"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert ok.status_code == 200


def test_trust_loopback_disabled_requires_token_on_box():
    c = _loopback_client(trust_loopback=False)
    assert c.post(
        "/route", json={"messages": [{"role": "user", "content": "hi"}]}
    ).status_code == 401
    ok = c.post(
        "/route",
        headers={"Authorization": "Bearer secret"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert ok.status_code == 200


def test_loopback_exemption_is_moot_without_a_key():
    # No token configured -> everything is open, proxied or not (today's behavior).
    c = _loopback_client(api_key=None)
    assert c.post(
        "/route",
        headers={"X-Forwarded-For": "100.126.90.22"},
        json={"messages": [{"role": "user", "content": "hi"}]},
    ).status_code == 200


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
