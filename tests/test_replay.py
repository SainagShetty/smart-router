"""What a config change would have done to traffic that actually happened.

Replay turns adjusting a threshold from a guess into an observation. It re-asks
the routing question for each logged decision under a candidate config and
compares the answer to what was recorded.

The property these tests guard hardest is COVERAGE HONESTY. Most of the live log
cannot be replayed at all:

    2,729 decisions
    ├─ 1,877  logged before the decision receipt (bd7f5a5, 2026-07-11) -- we
    │         cannot tell whether an override was in play
    ├─   601  carried force_tier / sensitive -- never consulted a threshold, so
    │         replaying them against a new one is meaningless
    └─   251  actually replayable

A preview that reports "0 requests would change" without saying it looked at 251
of 2,729 is indistinguishable from "this change is safe". That is the same
failure the read plane is built around -- empty must never be confusable with
broken -- and it is the one that would quietly cause a bad routing change to
ship.
"""
import json
import time

import pytest

from smartrouter import RouterCore
from smartrouter.config import RouterConfig

from conftest import make_config

FEATURES = {"estimated_tokens": 50, "has_images": False, "needs_tools": False,
            "needs_json": False, "num_turns": 1, "code_ratio": 0.0,
            "has_cjk": False, "raw_chars": 200}


def seed(store, score, tier, model, reason="tier=x; score=0.5",
         cost=0.001, latency=1000.0):
    """Insert one decision directly. Explicit beats going through record()."""
    store._conn.execute(
        "INSERT INTO decisions (decision_id, ts, prompt_sha256, features, score, "
        "chosen_tier, chosen_model, cost, latency_ms, reason) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (f"d{time.time_ns()}", time.time(), "sha", json.dumps(FEATURES), score,
         tier, model, cost, latency, reason),
    )
    store._conn.commit()


@pytest.fixture
def core(tmp_db):
    c = RouterCore(make_config(db_path=tmp_db))
    yield c
    c.close()


def lowered(threshold=0.25, tier="cheap"):
    cfg = make_config()
    for t in cfg.tiers:
        if t.name == tier:
            t.min_score = threshold
    return cfg


# --------------------------------------------------------------------------
# the engine must not invent deltas
# --------------------------------------------------------------------------

def test_replaying_a_config_against_itself_moves_nothing(core):
    # Seed tiers that match the thresholds in make_config: local 0.0, cheap
    # 0.40, frontier 0.75. Seeding a score against the WRONG tier makes replay
    # report a move -- which is the engine working, but not what this asserts.
    for score, tier, model in ((0.10, "local", "llama3.1:8b"),
                               (0.30, "local", "llama3.1:8b"),
                               (0.50, "cheap", "gpt-4o-mini"),
                               (0.90, "frontier", "claude-3.5-sonnet")):
        seed(core.store, score, tier, model)
    out = core.replay(core.config)
    assert out["moves"] == {}, f"identity replay must be a no-op, got {out['moves']}"
    assert out["unchanged"] == 4


def test_lowering_a_threshold_moves_rows_in_the_expected_direction(core):
    for score in (0.30, 0.35, 0.10):
        seed(core.store, score, "local", "llama3.1:8b", cost=None, latency=28000.0)
    out = core.replay(lowered(0.25))
    assert out["moves"] == {"local -> cheap": 2}, out["moves"]
    assert out["unchanged"] == 1


# --------------------------------------------------------------------------
# coverage honesty -- the critical one
# --------------------------------------------------------------------------

def test_empty_corpus_reports_zero_coverage_not_zero_changes(core):
    """The failure this whole design exists to prevent."""
    out = core.replay(lowered())
    assert out["coverage"]["replayed"] == 0
    assert out["coverage"]["total"] == 0
    assert out["moves"] == {}
    # A caller CANNOT read this as "safe" without also seeing replayed == 0.
    assert "replayed" in out["coverage"] and "eligible" in out["coverage"]


def test_overridden_decisions_are_excluded_and_counted(core):
    seed(core.store, 0.30, "local", "llama3.1:8b")
    seed(core.store, 0.30, "frontier", "claude-3.5-sonnet",
         reason="tier=frontier; force_tier=frontier, score=0.30 (ignored: forced)")
    seed(core.store, 0.30, "local", "llama3.1:8b",
         reason="tier=local; pinned local (sensitive), score=0.30")
    out = core.replay(lowered())
    assert out["coverage"]["replayed"] == 1
    assert out["coverage"]["excluded_override"] == 2
    assert out["coverage"]["total"] == 3


def test_decisions_without_a_receipt_are_excluded_and_counted(core):
    seed(core.store, 0.30, "local", "llama3.1:8b")
    seed(core.store, 0.30, "local", "llama3.1:8b", reason=None)
    out = core.replay(lowered())
    assert out["coverage"]["replayed"] == 1
    assert out["coverage"]["excluded_no_receipt"] == 1


def test_logging_disabled_says_so_rather_than_reporting_a_clean_zero(core):
    c = RouterCore(make_config())          # no db_path -> logging disabled
    out = c.replay(lowered())
    assert out["coverage"]["replayed"] == 0
    assert "reason" in out["coverage"], "must explain why, not just report zero"
    assert "disabled" in out["coverage"]["reason"]


# --------------------------------------------------------------------------
# estimates
# --------------------------------------------------------------------------

def test_moving_off_a_free_tier_is_still_priced(core):
    """The local tier records cost NULL because it is FREE, not unknown.

    Treating that as unknown silently dropped every row that moved off local --
    exactly the rows the estimate exists to describe.
    """
    for score in (0.30, 0.35):
        seed(core.store, score, "local", "llama3.1:8b", cost=None, latency=28000.0)
    seed(core.store, 0.50, "cheap", "gpt-4o-mini", cost=0.002, latency=1500.0)
    out = core.replay(lowered(0.25))
    assert out["estimates"] is not None
    assert out["estimates"]["priced_rows"] == 3
    assert out["estimates"]["cost_delta"] > 0, "moving off free must cost more"
    assert out["estimates"]["latency_delta_ms_per_request"] < 0, "and be faster"


def test_estimates_are_labelled_as_estimates(core):
    # Both tiers need observations: pricing a move into a tier nothing was ever
    # observed in is exactly the guess this refuses to make.
    seed(core.store, 0.30, "local", "llama3.1:8b", cost=None, latency=28000.0)
    seed(core.store, 0.50, "cheap", "gpt-4o-mini", cost=0.002, latency=1500.0)
    out = core.replay(lowered())
    assert "basis" in out["estimates"]
    assert "median" in out["estimates"]["basis"]


def test_no_estimate_when_the_destination_tier_was_never_observed(core):
    """Refusing to guess is the correct answer, not a bug."""
    seed(core.store, 0.30, "local", "llama3.1:8b", cost=None, latency=28000.0)
    out = core.replay(lowered())
    assert out["moves"] == {"local -> cheap": 1}
    assert out["estimates"] is None, (
        "nothing was ever observed in the cheap tier, so there is no honest "
        "basis for pricing a move into it"
    )


# --------------------------------------------------------------------------
# bounds and edges
# --------------------------------------------------------------------------

def test_limit_is_respected(core):
    for _ in range(10):
        seed(core.store, 0.30, "local", "llama3.1:8b")
    out = core.replay(lowered(), limit=3)
    assert out["coverage"]["replayed"] == 3
    assert out["coverage"]["eligible"] == 10, "eligible reports the true total"


def test_a_candidate_that_cannot_serve_a_request_reports_it(core):
    """Not a silent skip: 'no model can serve this' is a real outcome."""
    seed(core.store, 0.30, "local", "llama3.1:8b")
    cfg = make_config()
    for m in cfg.models:
        m.context_window = 1          # nothing can hold a 50-token prompt
    out = core.replay(cfg)
    assert any("no eligible model" in k for k in out["moves"]), out["moves"]
