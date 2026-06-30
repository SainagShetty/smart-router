"""SQLite training-data store.

Each routing decision is written as a *trainable* row: the embedding vector, the
extracted features, the score, the chosen model, observed cost/latency, and a
nullable label. Labels are what make the corpus trainable later — they arrive
implicitly (an escalation/validation failure) or explicitly via ``feedback()``.

Privacy: the raw prompt is stored only when ``log_raw=True``; otherwise just a
SHA-256. Never log secrets.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id        TEXT PRIMARY KEY,
    ts                 REAL NOT NULL,
    prompt_sha256      TEXT NOT NULL,
    prompt_raw         TEXT,
    embedding          BLOB,
    embedding_model_id TEXT,
    features           TEXT,
    score              REAL,
    chosen_tier        TEXT,
    chosen_model       TEXT,
    candidates         TEXT,
    cost               REAL,
    latency_ms         REAL,
    label              INTEGER,
    label_source       TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_label ON decisions(label);
CREATE INDEX IF NOT EXISTS idx_decisions_embmodel ON decisions(embedding_model_id);
"""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TrainingStore:
    def __init__(self, db_path: str = "smartrouter.db"):
        self.db_path = db_path
        # check_same_thread=False so a single store can be shared by a threaded
        # server (e.g. uvicorn's threadpool); a lock serializes access since one
        # sqlite3 connection is not safe for concurrent use.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA busy_timeout=5000;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---- writes -----------------------------------------------------------

    def record(
        self,
        *,
        decision_id: str,
        prompt: str,
        embedding: Optional[np.ndarray],
        embedding_model_id: str,
        features: Dict[str, Any],
        score: float,
        chosen_tier: str,
        chosen_model: str,
        candidates: List[str],
        cost: Optional[float] = None,
        latency_ms: Optional[float] = None,
        label: Optional[int] = None,
        label_source: Optional[str] = None,
        log_raw: bool = False,
    ) -> str:
        blob = None
        if embedding is not None:
            blob = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO decisions
                   (decision_id, ts, prompt_sha256, prompt_raw, embedding,
                    embedding_model_id, features, score, chosen_tier, chosen_model,
                    candidates, cost, latency_ms, label, label_source)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    decision_id, time.time(), _sha256(prompt),
                    prompt if log_raw else None, blob, embedding_model_id,
                    json.dumps(features), score, chosen_tier, chosen_model,
                    json.dumps(candidates), cost, latency_ms, label, label_source,
                ),
            )
            self._conn.commit()
        return decision_id

    def update_outcome(
        self,
        decision_id: str,
        *,
        cost: Optional[float] = None,
        latency_ms: Optional[float] = None,
        chosen_model: Optional[str] = None,
    ) -> None:
        """Record observed cost/latency and the model actually used (post-fallback)."""
        with self._lock:
            self._conn.execute(
                "UPDATE decisions SET cost=?, latency_ms=?, chosen_model=? "
                "WHERE decision_id=?",
                (cost, latency_ms, chosen_model, decision_id),
            )
            self._conn.commit()

    def feedback(self, decision_id: str, label: int, source: str = "manual") -> bool:
        """Attach/overwrite a label. Returns True if a row was updated."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE decisions SET label=?, label_source=? WHERE decision_id=?",
                (int(label), source, decision_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def set_label_if_unset(self, decision_id: str, label: int, source: str) -> bool:
        """Record a weak/implicit label without clobbering an explicit one."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE decisions SET label=?, label_source=? "
                "WHERE decision_id=? AND label IS NULL",
                (int(label), source, decision_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ---- reads ------------------------------------------------------------

    def count(self, labeled_only: bool = False) -> int:
        q = "SELECT COUNT(*) FROM decisions"
        if labeled_only:
            q += " WHERE label IS NOT NULL"
        with self._lock:
            return int(self._conn.execute(q).fetchone()[0])

    def tier_counts(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chosen_tier, COUNT(*) FROM decisions GROUP BY chosen_tier"
            ).fetchall()
        return {tier: n for tier, n in rows}

    def training_matrix(self, embedding_model_id: str) -> Tuple[np.ndarray, List[int]]:
        """Stored embeddings + labels for rows in a given embedding space.

        Lets the head be retrained without re-embedding any prompt.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT embedding, label FROM decisions "
                "WHERE label IS NOT NULL AND embedding IS NOT NULL "
                "AND embedding_model_id = ?",
                (embedding_model_id,),
            ).fetchall()
        vectors, labels = [], []
        for blob, label in rows:
            vectors.append(np.frombuffer(blob, dtype=np.float32))
            labels.append(int(label))
        if not vectors:
            return np.empty((0, 0), dtype=np.float32), []
        return np.vstack(vectors), labels

    def labeled_prompts(self) -> List[Tuple[str, int]]:
        """(raw_prompt, label) pairs where raw text was retained — for distillation."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT prompt_raw, label FROM decisions "
                "WHERE label IS NOT NULL AND prompt_raw IS NOT NULL"
            ).fetchall()
        return [(p, int(l)) for p, l in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
