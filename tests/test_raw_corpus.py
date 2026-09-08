"""Raw-corpus capture: per-request log_raw override, response capture (the
distillation target), source/sensitive tagging, and in-place DB migration."""
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


def _core(tmp_db, log_raw):
    cfg = make_config(db_path=tmp_db)
    cfg.logging.log_raw = log_raw
    core = RouterCore(cfg)
    stub_providers(core, content="the answer")
    return core


# ---- per-request log_raw override ------------------------------------------

def test_config_log_raw_true_stores_prompt_and_response(tmp_db):
    core = _core(tmp_db, log_raw=True)
    _, d = core.complete([{"role": "user", "content": "what is 2+2"}])
    core.close()
    prompt, resp = _row(tmp_db, d.decision_id, ["prompt_raw", "response_raw"])
    assert prompt == "what is 2+2"
    assert resp == "the answer"


def test_per_request_opt_out_overrides_config(tmp_db):
    core = _core(tmp_db, log_raw=True)
    _, d = core.complete(
        [{"role": "user", "content": "secret stuff"}], log_raw=False
    )
    core.close()
    prompt, resp = _row(tmp_db, d.decision_id, ["prompt_raw", "response_raw"])
    assert prompt is None
    assert resp is None


def test_per_request_opt_in_overrides_config(tmp_db):
    core = _core(tmp_db, log_raw=False)
    _, d = core.complete(
        [{"role": "user", "content": "log me"}], log_raw=True
    )
    core.close()
    prompt, resp = _row(tmp_db, d.decision_id, ["prompt_raw", "response_raw"])
    assert prompt == "log me"
    assert resp == "the answer"


def test_unset_falls_back_to_config_default_off(tmp_db):
    core = _core(tmp_db, log_raw=False)
    _, d = core.complete([{"role": "user", "content": "hello"}])
    core.close()
    prompt, resp = _row(tmp_db, d.decision_id, ["prompt_raw", "response_raw"])
    assert prompt is None
    assert resp is None


# ---- streaming response capture --------------------------------------------

def test_streaming_accumulates_response(tmp_db):
    core = _core(tmp_db, log_raw=True)
    chunks, d = core.stream([{"role": "user", "content": "hi"}])
    list(chunks)  # drain
    core.close()
    (resp,) = _row(tmp_db, d.decision_id, ["response_raw"])
    assert resp == "the answer"


def test_streaming_respects_log_raw_off(tmp_db):
    core = _core(tmp_db, log_raw=True)
    chunks, d = core.stream([{"role": "user", "content": "hi"}], log_raw=False)
    list(chunks)
    core.close()
    (resp,) = _row(tmp_db, d.decision_id, ["response_raw"])
    assert resp is None


# ---- source / sensitive tagging ---------------------------------------------

def test_source_and_sensitive_recorded(tmp_db):
    core = _core(tmp_db, log_raw=True)
    _, d = core.complete(
        [{"role": "user", "content": "how much did I spend"}],
        source="finance-auditor", sensitive=True,
    )
    _, d2 = core.complete(
        [{"role": "user", "content": "what does sai do"}], source="resume-bot"
    )
    core.close()
    src, sens = _row(tmp_db, d.decision_id, ["source", "sensitive"])
    assert (src, sens) == ("finance-auditor", 1)
    src2, sens2 = _row(tmp_db, d2.decision_id, ["source", "sensitive"])
    assert (src2, sens2) == ("resume-bot", 0)


def test_sft_rows_and_sensitive_filter(tmp_db):
    core = _core(tmp_db, log_raw=True)
    core.complete([{"role": "user", "content": "public q"}], source="resume-bot")
    core.complete([{"role": "user", "content": "private q"}],
                  source="finance-auditor", sensitive=True)
    core.close()

    store = TrainingStore(tmp_db)
    all_rows = store.sft_rows()
    assert len(all_rows) == 2
    assert {r["prompt"] for r in all_rows} == {"public q", "private q"}
    assert all(r["response"] == "the answer" for r in all_rows)
    safe = store.sft_rows(include_sensitive=False)
    assert [r["prompt"] for r in safe] == ["public q"]
    store.close()


def test_sft_rows_drops_synthetic_sources_by_default(tmp_db):
    core = _core(tmp_db, log_raw=True)
    core.complete([{"role": "user", "content": "real q"}], source="resume-bot")
    core.complete([{"role": "user", "content": "ok probe"}],
                  source="chat-healthcheck")
    core.complete([{"role": "user", "content": "ramp q"}], source="router-ramp")
    core.complete([{"role": "user", "content": "untagged q"}])  # source IS NULL
    core.close()

    store = TrainingStore(tmp_db)
    # Untagged rows predate source tagging. They are unattributed, not
    # synthetic, and a naive `source NOT IN (...)` would drop them: in SQL
    # `NULL NOT IN (...)` is NULL, not true.
    assert sorted(r["prompt"] for r in store.sft_rows()) == ["real q", "untagged q"]
    assert len(store.sft_rows(include_synthetic=True)) == 4
    store.close()


def test_sft_rows_synthetic_and_sensitive_filters_compose(tmp_db):
    core = _core(tmp_db, log_raw=True)
    core.complete([{"role": "user", "content": "real q"}], source="resume-bot")
    core.complete([{"role": "user", "content": "private q"}],
                  source="finance-auditor", sensitive=True)
    core.complete([{"role": "user", "content": "ok probe"}],
                  source="chat-healthcheck")
    core.close()

    store = TrainingStore(tmp_db)
    rows = store.sft_rows(include_sensitive=False)
    assert [r["prompt"] for r in rows] == ["real q"]
    store.close()


# ---- server layer -------------------------------------------------------------

def test_server_logging_fields_logged_not_forwarded(tmp_db):
    from fastapi.testclient import TestClient
    from smartrouter.server import create_app

    cfg = make_config(db_path=tmp_db)
    cfg.logging.log_raw = True
    app = create_app(cfg)
    captured = {}

    def make(name):
        def complete(model, messages, **p):
            captured["params"] = p
            return {"id": "x", "model": model,
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                    "usage": {"total_tokens": 10}}
        return complete

    for name, prov in app.state.core.providers.items():
        prov.complete = make(name)

    c = TestClient(app)
    r = c.post("/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "spend summary"}],
        "source": "finance-auditor", "sensitive": True, "max_tokens": 64,
    })
    assert r.status_code == 200
    # logging controls must never reach the provider; real params must.
    assert "source" not in captured["params"]
    assert "sensitive" not in captured["params"]
    assert "log_raw" not in captured["params"]
    assert captured["params"]["max_tokens"] == 64
    row = _row(tmp_db, r.json()["decision_id"],
               ["prompt_raw", "response_raw", "source", "sensitive"])
    assert row == ("spend summary", "ok", "finance-auditor", 1)


# ---- migration ---------------------------------------------------------------

def test_migration_adds_columns_to_v1_db(tmp_db):
    # Build a v1-shaped DB (no response_raw/source/sensitive) with one row.
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

    store = TrainingStore(tmp_db)  # triggers ALTER TABLE migration
    store.record(
        decision_id="new", prompt="p", embedding=None, embedding_model_id="m",
        features={}, score=0.5, chosen_tier="local", chosen_model="x",
        candidates=[], log_raw=True, source="svc", sensitive=True,
    )
    store.update_outcome("new", response_raw="r")
    store.close()

    conn = sqlite3.connect(tmp_db)
    old = conn.execute(
        "SELECT response_raw, source, sensitive FROM decisions "
        "WHERE decision_id='old'"
    ).fetchone()
    new = conn.execute(
        "SELECT prompt_raw, response_raw, source, sensitive FROM decisions "
        "WHERE decision_id='new'"
    ).fetchone()
    conn.close()
    assert old == (None, None, 0)
    assert new == ("p", "r", "svc", 1)
