import pytest

pytest.importorskip("fastapi")
pytest.importorskip("ruamel.yaml")
from fastapi.testclient import TestClient  # noqa: E402

from smartrouter.logging_ import TrainingStore  # noqa: E402
from smartrouter.ui_server import create_app  # noqa: E402

CONFIG_YAML = """\
# a comment that must survive a save
providers:
  ollama: { type: ollama, base_url: http://localhost:11434/v1, local: true }
  openrouter: { type: openrouter, api_key_env: OPENROUTER_API_KEY }

tiers:
  - { name: local,    min_score: 0.0 }
  - { name: cheap,    min_score: 0.40 }  # inline comment
  - { name: frontier, min_score: 0.75 }

models:
  - id: "llama3.1:8b"
    provider: ollama
    tier: local
    context_window: 8192
    # a comment on this model's capabilities that must survive unrelated edits
    capabilities: {}
  - id: "gpt-4o-mini"
    provider: openrouter
    tier: cheap
    cost_per_1k: 0.0006
    context_window: 128000
    capabilities: { vision: true, tools: true, json: true }

policy:
  default_tier: local
"""


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "decisions.db"


@pytest.fixture
def config_file(tmp_path, db_path):
    path = tmp_path / "router.yaml"
    path.write_text(CONFIG_YAML + f"\nlogging:\n  db_path: {db_path}\n")
    return path


def _record(store, decision_id, chosen_model, chosen_tier, cost=None, label=None):
    store.record(
        decision_id=decision_id,
        prompt=f"prompt for {decision_id}",
        embedding=None,
        embedding_model_id="hashing:1024",
        features={},
        score=0.1,
        chosen_tier=chosen_tier,
        chosen_model=chosen_model,
        candidates=[chosen_model],
        cost=cost,
        label=label,
    )


def _client(config_path):
    return TestClient(create_app(str(config_path)))


def test_serves_index():
    c = _client("router.yaml")
    r = c.get("/")
    assert r.status_code == 200
    assert "Router Configuration" in r.text


def test_get_config_missing_file(tmp_path):
    c = _client(tmp_path / "nope.yaml")
    r = c.get("/api/config")
    assert r.status_code == 404


def test_get_config(config_file):
    c = _client(config_file)
    r = c.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert [t["name"] for t in body["tiers"]] == ["local", "cheap", "frontier"]


def test_update_config_preserves_comments_and_other_keys(config_file):
    c = _client(config_file)
    r = c.post("/api/config", json={"tiers": [{"name": "cheap", "min_score": 0.55}]})
    assert r.status_code == 200
    assert "restart" in r.json()["note"]

    text = config_file.read_text()
    assert "# a comment that must survive a save" in text
    assert "# inline comment" in text
    assert "default_tier: local" in text

    r2 = c.get("/api/config")
    tiers = {t["name"]: t["min_score"] for t in r2.json()["tiers"]}
    assert tiers == {"local": 0.0, "cheap": 0.55, "frontier": 0.75}


def test_update_config_unknown_tier_rejected(config_file):
    c = _client(config_file)
    r = c.post("/api/config", json={"tiers": [{"name": "nope", "min_score": 0.5}]})
    assert r.status_code == 400
    # file must be untouched
    assert "# a comment that must survive a save" in config_file.read_text()


def test_update_config_skips_write_for_unchanged_values(config_file):
    # Re-posting the same value must not rewrite the file (avoids reformatting
    # untouched flow-style entries and losing their original scalar styling).
    before = config_file.read_text()
    c = _client(config_file)
    r = c.post("/api/config", json={"tiers": [{"name": "local", "min_score": 0.0}]})
    assert r.status_code == 200
    assert config_file.read_text() == before


def test_update_config_out_of_range_rejected(config_file):
    c = _client(config_file)
    r = c.post("/api/config", json={"tiers": [{"name": "cheap", "min_score": 1.5}]})
    assert r.status_code == 422


def _base_models(config_file):
    return _client(config_file).get("/api/config").json()["models"]


def test_get_config_includes_models(config_file):
    models = _base_models(config_file)
    assert [m["id"] for m in models] == ["llama3.1:8b", "gpt-4o-mini"]


def test_update_models_edit_field_preserves_comment_and_siblings(config_file):
    c = _client(config_file)
    r = c.post("/api/config", json={"models": [
        {"id": "llama3.1:8b", "provider": "ollama", "tier": "local",
         "context_window": 16384, "orig_index": 0},
        {"id": "gpt-4o-mini", "provider": "openrouter", "tier": "cheap",
         "cost_per_1k": 0.0006, "context_window": 128000,
         "vision": True, "tools": True, "json_mode": True, "orig_index": 1},
    ]})
    assert r.status_code == 200, r.json()

    text = config_file.read_text()
    assert "# a comment on this model's capabilities that must survive unrelated edits" in text
    assert 'id: "gpt-4o-mini"' in text  # untouched sibling row's quoting style intact

    updated = _base_models(config_file)
    assert updated[0]["context_window"] == 16384
    assert updated[0]["id"] == "llama3.1:8b"


def test_update_models_omitted_field_leaves_models_untouched(config_file):
    before = config_file.read_text()
    c = _client(config_file)
    r = c.post("/api/config", json={"tiers": [{"name": "local", "min_score": 0.0}]})
    assert r.status_code == 200
    assert config_file.read_text() == before


def test_add_model(config_file):
    models = _base_models(config_file)
    payload = [
        {"id": m["id"], "provider": m["provider"], "tier": m["tier"],
         "cost_per_1k": m.get("cost_per_1k", 0.0), "context_window": m["context_window"],
         "vision": m.get("capabilities", {}).get("vision", False),
         "tools": m.get("capabilities", {}).get("tools", False),
         "json_mode": m.get("capabilities", {}).get("json", False),
         "orig_index": i}
        for i, m in enumerate(models)
    ]
    payload.append({"id": "new-model", "provider": "openrouter", "tier": "frontier",
                     "cost_per_1k": 0.01, "context_window": 200000,
                     "vision": True, "tools": True, "json_mode": True,
                     "orig_index": None})

    c = _client(config_file)
    r = c.post("/api/config", json={"models": payload})
    assert r.status_code == 200, r.json()

    updated = _base_models(config_file)
    assert [m["id"] for m in updated] == ["llama3.1:8b", "gpt-4o-mini", "new-model"]
    assert updated[2]["tier"] == "frontier"


def test_remove_model(config_file):
    c = _client(config_file)
    r = c.post("/api/config", json={"models": [
        {"id": "gpt-4o-mini", "provider": "openrouter", "tier": "cheap",
         "cost_per_1k": 0.0006, "context_window": 128000,
         "vision": True, "tools": True, "json_mode": True, "orig_index": 1},
    ]})
    assert r.status_code == 200, r.json()
    updated = _base_models(config_file)
    assert [m["id"] for m in updated] == ["gpt-4o-mini"]


def test_update_models_unknown_provider_rejected(config_file):
    c = _client(config_file)
    r = c.post("/api/config", json={"models": [
        {"id": "llama3.1:8b", "provider": "bogus-provider", "tier": "local",
         "context_window": 8192, "orig_index": 0},
    ]})
    assert r.status_code == 400
    assert "invalid" in r.json()["detail"]
    # rejected save must not touch the file
    assert "bogus-provider" not in config_file.read_text()


def test_update_models_unknown_tier_rejected(config_file):
    c = _client(config_file)
    r = c.post("/api/config", json={"models": [
        {"id": "llama3.1:8b", "provider": "ollama", "tier": "nonexistent-tier",
         "context_window": 8192, "orig_index": 0},
    ]})
    assert r.status_code == 400
    assert "invalid" in r.json()["detail"]


def test_update_models_skips_write_when_resubmitted_unchanged(config_file):
    models = _base_models(config_file)
    payload = [
        {"id": m["id"], "provider": m["provider"], "tier": m["tier"],
         "cost_per_1k": m.get("cost_per_1k", 0.0), "context_window": m["context_window"],
         "vision": m.get("capabilities", {}).get("vision", False),
         "tools": m.get("capabilities", {}).get("tools", False),
         "json_mode": m.get("capabilities", {}).get("json", False),
         "orig_index": i}
        for i, m in enumerate(models)
    ]
    before = config_file.read_text()
    c = _client(config_file)
    r = c.post("/api/config", json={"models": payload})
    assert r.status_code == 200, r.json()
    assert config_file.read_text() == before


def test_stats_missing_config(tmp_path):
    c = _client(tmp_path / "nope.yaml")
    r = c.get("/api/stats")
    assert r.status_code == 404


def test_stats_no_traffic_yet(config_file):
    # No TrainingStore has been created for this db_path yet.
    c = _client(config_file)
    r = c.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["labeled"] == 0
    assert body["savings"]["on_device_pct"] == 0.0


def test_stats_with_traffic(config_file, db_path):
    store = TrainingStore(str(db_path))
    _record(store, "d1", "llama3.1:8b", "local", cost=0.0, label=1)
    _record(store, "d2", "llama3.1:8b", "local", cost=0.0)
    _record(store, "d3", "gpt-4o-mini", "cheap", cost=0.002)
    store.close()

    c = _client(config_file)
    r = c.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert body["labeled"] == 1
    assert body["by_tier"] == {"local": 2, "cheap": 1}
    # llama3.1:8b runs on the local ollama provider -> counts as on-device.
    assert body["savings"]["on_device"] == 2
    assert body["savings"]["free"] == 2
    assert body["savings"]["on_device_pct"] == round(100 * 2 / 3, 1)
    assert body["savings"]["total_cost"] == 0.002


def test_stats_logging_disabled(tmp_path):
    path = tmp_path / "router.yaml"
    path.write_text(CONFIG_YAML + "\nlogging:\n  enabled: false\n")
    c = _client(path)
    r = c.get("/api/stats")
    assert r.status_code == 200
    assert r.json() == {"logging": "disabled"}


def test_static_and_config_paths_independent_of_cwd(config_file, tmp_path, monkeypatch):
    # regression: the old module hardcoded cwd-relative paths for both the
    # static dir and the config file, which broke when run from elsewhere.
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    c = _client(config_file)
    assert c.get("/").status_code == 200
    assert c.get("/api/config").status_code == 200
