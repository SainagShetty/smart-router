"""Savings summary: honest, computed only from persisted data. Leads with the
free-local floor (served-free %, kept-on-device), not an invented dollar brag,
and reads ~0 for traffic that never stayed local."""
from smartrouter.logging_ import TrainingStore


def _store(tmp_db, rows):
    s = TrainingStore(tmp_db)
    for i, (model, tier, cost) in enumerate(rows):
        s.record(decision_id=f"d{i}", prompt="p", embedding=None,
                 embedding_model_id="m", features={}, score=0.5,
                 chosen_tier=tier, chosen_model=model, candidates=[], cost=cost)
    return s


def test_empty_store_is_zero(tmp_db):
    s = TrainingStore(tmp_db)
    summary = s.savings_summary()
    s.close()
    assert summary["total"] == 0
    assert summary["on_device_pct"] == 0.0
    assert summary["total_cost"] == 0.0


def test_free_local_and_spend(tmp_db):
    s = _store(tmp_db, [
        ("llama3.1:8b", "local", None),     # free local
        ("llama3.1:8b", "local", 0.0),      # free local
        ("gpt-4o-mini", "cheap", 0.005),    # cloud, paid
    ])
    summary = s.savings_summary(local_models={"llama3.1:8b"})
    s.close()
    assert summary["total"] == 3
    assert summary["on_device"] == 2
    assert summary["on_device_pct"] == 66.7
    assert summary["free"] == 2
    assert round(summary["total_cost"], 6) == 0.005
    assert summary["by_tier"] == {"local": 2, "cheap": 1}


def test_all_cloud_reads_zero_on_device(tmp_db):
    s = _store(tmp_db, [
        ("gpt-4o-mini", "cheap", 0.005),
        ("claude-3.5-sonnet", "frontier", 0.02),
    ])
    summary = s.savings_summary(local_models={"llama3.1:8b"})
    s.close()
    assert summary["on_device"] == 0
    assert summary["on_device_pct"] == 0.0


def test_server_stats_includes_savings(tmp_db):
    from fastapi.testclient import TestClient
    from smartrouter.server import create_app
    from conftest import make_config

    cfg = make_config(db_path=tmp_db)
    app = create_app(cfg)
    for name, prov in app.state.core.providers.items():
        prov.complete = (lambda m, msgs, **p: {
            "model": m, "choices": [{"message": {"role": "assistant",
                                                 "content": "ok"}}],
            "usage": {"total_tokens": 10}})
    c = TestClient(app)
    c.post("/v1/chat/completions",
           json={"messages": [{"role": "user", "content": "2+2?"}]})
    r = c.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert "savings" in body
    assert body["savings"]["on_device"] == 1
