"""Decision receipt: reason / rejected / band are computed then must be persisted
so a routing decision is auditable after the fact."""
import json
import sqlite3

from smartrouter import RouterCore
from smartrouter.logging_ import TrainingStore

from conftest import make_config, stub_providers


def _row(db_path, decision_id, cols):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            f"SELECT {', '.join(cols)} FROM decisions WHERE decision_id=?",
            (decision_id,),
        ).fetchone()
    finally:
        conn.close()


def test_reason_and_rejected_persisted(tmp_db):
    cfg = make_config(db_path=tmp_db)
    core = RouterCore(cfg)
    stub_providers(core)
    # an image forces vision -> the local model is rejected, recorded as such
    _, d = core.complete([{"role": "user", "content": [
        {"type": "text", "text": "what is in this image"},
        {"type": "image_url", "image_url": {"url": "data:x"}}]}])
    core.close()
    reason, rejected = _row(tmp_db, d.decision_id, ["reason", "rejected"])
    assert reason == d.reason
    assert any("llama3.1:8b" in r for r in json.loads(rejected))


def test_reason_records_sensitive_pin(tmp_db):
    cfg = make_config(db_path=tmp_db)
    core = RouterCore(cfg)
    stub_providers(core)
    _, d = core.complete([{"role": "user", "content": "private note"}],
                         sensitive=True)
    core.close()
    (reason,) = _row(tmp_db, d.decision_id, ["reason"])
    assert "sensitive" in reason


def test_receipt_migration_adds_columns(tmp_db):
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        """CREATE TABLE decisions (
               decision_id TEXT PRIMARY KEY, ts REAL NOT NULL,
               prompt_sha256 TEXT NOT NULL, prompt_raw TEXT, embedding BLOB,
               embedding_model_id TEXT, features TEXT, score REAL,
               chosen_tier TEXT, chosen_model TEXT, candidates TEXT,
               cost REAL, latency_ms REAL, label INTEGER, label_source TEXT)"""
    )
    conn.execute(
        "INSERT INTO decisions (decision_id, ts, prompt_sha256) VALUES ('old', 1, 'x')"
    )
    conn.commit()
    conn.close()

    store = TrainingStore(tmp_db)
    store.record(
        decision_id="new", prompt="p", embedding=None, embedding_model_id="m",
        features={}, score=0.5, chosen_tier="local", chosen_model="x",
        candidates=[], reason="tier=local; score=0.500",
        rejected=["gpt-4o-mini: too pricey"],
    )
    store.close()

    conn = sqlite3.connect(tmp_db)
    old = conn.execute(
        "SELECT reason, rejected FROM decisions WHERE decision_id='old'"
    ).fetchone()
    new = conn.execute(
        "SELECT reason, rejected FROM decisions WHERE decision_id='new'"
    ).fetchone()
    conn.close()
    assert old == (None, None)
    assert new[0] == "tier=local; score=0.500"
    assert json.loads(new[1]) == ["gpt-4o-mini: too pricey"]
