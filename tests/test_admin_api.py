"""The admin surface: a second auth tier, and an ordering that cannot lie.

TWO TIERS, ONE PROCESS
    /v1/* exempts loopback callers, because ten co-located services depend on
    that and a token in front of them protects nothing they could not already
    read out of this process's environment.

    /admin/* does not, because these endpoints rewrite where every LLM call in
    the fleet goes. Under the same gate, any process on this box -- including an
    SSRF in any of the ten -- could repoint the router with no credential. The
    inverse test (loopback WITHOUT a token must be REFUSED) is the one that
    matters; it is the exact opposite of what test_server.py asserts for /v1.

THE ORDERING
    validate -> render -> reload -> commit.

    The router boots from the rendered file, so that file is the record of what
    runs. Changing it first means the database can only ever lag the running
    router, never lead it. A lagging database is visible in /health; a leading
    one is a UI showing a config that was never applied.
"""
import pathlib

import pytest
import yaml as pyyaml
from fastapi.testclient import TestClient

from smartrouter.config import RouterConfig, render_revision, write_rendered
from smartrouter.server import create_app

from conftest import make_config

TOKEN = "admin-secret"


# A production-SHAPED config: providers carry api_key_env, never a literal.
# conftest.make_config uses a literal for convenience, and the apply guard
# correctly refuses it -- which is how this fixture came to be written by hand.
BASE_YAML = """\
providers:
  ollama:
    type: ollama
    base_url: http://localhost:11434/v1
    local: true
  openrouter:
    type: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY

tiers:
  - {name: local, min_score: 0.0}
  - {name: cheap, min_score: 0.40}
  - {name: frontier, min_score: 0.75}

models:
  - id: "llama3.1:8b"
    provider: ollama
    tier: local
    context_window: 8192
    capabilities: {tools: true}
  - id: "gpt-4o-mini"
    provider: openrouter
    tier: cheap
    context_window: 128000
    cost_per_1k: 0.0006
    capabilities: {vision: true, tools: true, json: true}
  - id: "claude-3.5-sonnet"
    provider: openrouter
    tier: frontier
    context_window: 200000
    cost_per_1k: 0.009
    capabilities: {vision: true, tools: true, json: true}

policy:
  fallback: down
"""


@pytest.fixture
def env(tmp_path, tmp_db):
    """A server whose rendered config lives on disk, as in production."""
    import yaml as _y
    doc = _y.safe_load(BASE_YAML)
    doc["logging"] = {"enabled": True, "db_path": tmp_db}
    base_yaml = _y.safe_dump(doc)

    path = tmp_path / "router.yaml"
    write_rendered(str(path), base_yaml)
    app = create_app(RouterConfig.from_yaml(str(path)),
                     config_path=str(path), admin_token=TOKEN)
    client = TestClient(app)
    # Seed revision 1 so there is an active revision to diff and revert against.
    r = client.post("/admin/api/apply", json={"yaml": base_yaml, "note": "seed"},
                    headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 200, f"fixture seed failed: {r.text}"
    return client, path, base_yaml


def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def with_threshold(base_yaml, value):
    doc = pyyaml.safe_load(base_yaml)
    for t in doc["tiers"]:
        if t["name"] == "cheap":
            t["min_score"] = value
    return pyyaml.safe_dump(doc)


# --------------------------------------------------------------------------
# the auth split
# --------------------------------------------------------------------------

def test_admin_refuses_loopback_without_a_token(env):
    """The inverse of the /v1 exemption, and the whole point of the tier."""
    client, _, _ = env
    for path in ("/admin/api/config", "/admin/api/revisions"):
        assert client.get(path).status_code == 401, path


def test_admin_write_refuses_loopback_without_a_token(env):
    client, _, base = env
    r = client.post("/admin/api/apply", json={"yaml": base})
    assert r.status_code == 401, "an unauthenticated config write is the SSRF path"


def test_v1_still_exempts_loopback(env):
    """The other tier must be untouched: ten services depend on it."""
    client, _, _ = env
    r = client.post("/route", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200


def test_admin_disabled_when_no_token_configured(tmp_path, tmp_db):
    cfg = make_config(db_path=tmp_db)
    app = create_app(cfg, config_path=str(tmp_path / "r.yaml"))   # no admin_token
    client = TestClient(app)
    r = client.get("/admin/api/config", headers=auth())
    assert r.status_code == 503
    assert "SMARTROUTER_ADMIN_TOKEN" in r.json()["detail"]


# --------------------------------------------------------------------------
# apply ordering
# --------------------------------------------------------------------------

def test_apply_renders_reloads_and_commits(env):
    client, path, base = env
    r = client.post("/admin/api/apply",
                    json={"yaml": with_threshold(base, 0.25), "note": "faster"},
                    headers=auth())
    assert r.status_code == 200, r.text
    rid = r.json()["revision"]
    # all three landed, and agree
    assert RouterConfig.from_yaml(str(path)).revision == rid
    assert r.json()["running_revision"] == rid
    assert client.get("/admin/api/config", headers=auth()).json()["active_revision"] == rid


def test_health_reports_the_running_revision(env):
    client, _, base = env
    before = client.get("/health").json()["revision"]
    client.post("/admin/api/apply", json={"yaml": with_threshold(base, 0.3)},
                headers=auth())
    after = client.get("/health").json()["revision"]
    assert after != before, "/health must name what is actually being served"


def test_failed_render_changes_nothing(env, monkeypatch):
    client, path, base = env
    before_file = path.read_text()
    before_rev = client.get("/health").json()["revision"]

    import smartrouter.server as srv
    monkeypatch.setattr(srv, "write_rendered",
                        lambda *a, **k: (_ for _ in ()).throw(OSError(28, "full")))
    r = client.post("/admin/api/apply", json={"yaml": with_threshold(base, 0.25)},
                    headers=auth())
    assert r.status_code == 500
    assert path.read_text() == before_file, "the rendered config must be untouched"
    assert client.get("/health").json()["revision"] == before_rev


def test_unreloadable_change_is_saved_but_not_applied(env):
    """A provider change is valid history and an invalid hot swap."""
    client, path, base = env
    doc = pyyaml.safe_load(base)
    doc["providers"]["openrouter"]["base_url"] = "https://example.invalid/v1"
    before = path.read_text()

    r = client.post("/admin/api/apply", json={"yaml": pyyaml.safe_dump(doc)},
                    headers=auth())
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["saved_revision"], "the revision must survive as history"
    assert "restart" in detail["note"]
    assert path.read_text() == before, "nothing may be applied"


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_literal_api_key_is_refused(env):
    """The revision table is gzipped into ~/Backups nightly and kept 14 days."""
    client, _, base = env
    doc = pyyaml.safe_load(base)
    doc["providers"]["openrouter"]["api_key"] = "sk-should-never-persist"
    r = client.post("/admin/api/apply", json={"yaml": pyyaml.safe_dump(doc)},
                    headers=auth())
    assert r.status_code == 400
    assert "api_key_env" in r.json()["detail"]


def test_invalid_yaml_is_rejected_before_anything_is_written(env):
    client, path, _ = env
    before = path.read_text()
    r = client.post("/admin/api/apply", json={"yaml": "models: [oops"},
                    headers=auth())
    assert r.status_code == 400
    assert path.read_text() == before


# --------------------------------------------------------------------------
# preview + revert
# --------------------------------------------------------------------------

def test_preview_changes_nothing(env):
    client, path, base = env
    before_file, before_rev = path.read_text(), client.get("/health").json()["revision"]
    r = client.post("/admin/api/preview", json={"yaml": with_threshold(base, 0.25)},
                    headers=auth())
    assert r.status_code == 200
    body = r.json()
    assert "diff" in body and "replay" in body
    assert "coverage" in body["replay"], "a preview without coverage can read as safe"
    assert path.read_text() == before_file
    assert client.get("/health").json()["revision"] == before_rev


def test_preview_flags_a_capability_regression(env):
    client, _, base = env
    doc = pyyaml.safe_load(base)
    for m in doc["models"]:
        if m["tier"] == "cheap":
            m.setdefault("capabilities", {})["tools"] = False
    r = client.post("/admin/api/preview", json={"yaml": pyyaml.safe_dump(doc)},
                    headers=auth())
    assert any("tool calling" in g for g in r.json()["capability_regressions"])


def test_revert_restores_an_earlier_revision(env):
    client, path, base = env
    first = client.get("/health").json()["revision"]
    client.post("/admin/api/apply", json={"yaml": with_threshold(base, 0.25)},
                headers=auth())
    assert client.get("/health").json()["revision"] != first

    r = client.post(f"/admin/api/revert/{first}", headers=auth())
    assert r.status_code == 200
    restored = RouterConfig.from_yaml(str(path))
    cheap = [t for t in restored.tiers if t.name == "cheap"][0]
    assert cheap.min_score == 0.40, "the original threshold must be back"


def test_revert_to_unknown_revision_is_404(env):
    client, _, _ = env
    assert client.post("/admin/api/revert/999", headers=auth()).status_code == 404
