"""Routing config as append-only history.

The config used to live in `examples/router.yaml`, a git-tracked file that a
pull could silently revert under a running fleet. As rows it is append-only:
nothing is edited, nothing is deleted, and `active` names the one in force -- so
the audit trail and the revert mechanism are the same object.

These tests pin the two invariants everything above depends on:

    append-only     a revision is never mutated, so history is trustworthy
    one active      enforced by a partial unique index, not by convention

and the property that keeps the fleet bootable: the router never reads this
table at boot, so nothing here can stop it starting.
"""
import sqlite3

import pytest

from smartrouter.logging_ import TrainingStore

YAML_A = "models:\n  - id: a\n"
YAML_B = "models:\n  - id: b\n"


@pytest.fixture
def store(tmp_db):
    s = TrainingStore(tmp_db)
    yield s
    s.close()


# --------------------------------------------------------------------------
# append + activate
# --------------------------------------------------------------------------

def test_new_revision_is_not_active(store):
    """Creation and activation are separate steps on purpose: the caller must
    render and reload successfully before anything becomes current."""
    rid = store.add_revision(YAML_A, author="test", note="first")
    assert store.active_revision() is None
    assert store.revision(rid)["active"] is False


def test_activate_makes_exactly_one_active(store):
    a = store.add_revision(YAML_A)
    b = store.add_revision(YAML_B)
    assert store.activate_revision(a) is True
    assert store.active_revision()["id"] == a
    assert store.activate_revision(b) is True
    active = [r for r in store.revisions() if r["active"]]
    assert len(active) == 1 and active[0]["id"] == b


def test_activate_unknown_revision_returns_false(store):
    assert store.activate_revision(999) is False
    assert store.active_revision() is None


def test_two_active_rows_are_impossible(store):
    """The partial unique index is the enforcement, not application code."""
    a = store.add_revision(YAML_A)
    b = store.add_revision(YAML_B)
    store.activate_revision(a)
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            "UPDATE config_revisions SET active=1 WHERE id=?", (b,)
        )


def test_active_revision_carries_the_yaml(store):
    rid = store.add_revision(YAML_A, author="sai", note="swap cheap tier")
    store.activate_revision(rid)
    got = store.active_revision()
    assert got["yaml"] == YAML_A
    assert got["author"] == "sai"
    assert got["note"] == "swap cheap tier"


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------

def test_revisions_are_newest_first_and_omit_yaml(store):
    for i in range(3):
        store.add_revision(f"models: [{i}]")
    rows = store.revisions()
    assert [r["id"] for r in rows] == sorted((r["id"] for r in rows), reverse=True)
    assert "yaml" not in rows[0], "listing 50 configs' worth of yaml is wasteful"


def test_revisions_respects_limit(store):
    for i in range(5):
        store.add_revision(f"models: [{i}]")
    assert len(store.revisions(limit=2)) == 2


def test_history_is_append_only_across_activations(store):
    """Reverting must not destroy the revision it reverted away from."""
    a = store.add_revision(YAML_A)
    b = store.add_revision(YAML_B)
    store.activate_revision(a)
    store.activate_revision(b)
    store.activate_revision(a)          # revert
    assert len(store.revisions()) == 2
    assert store.revision(b)["yaml"] == YAML_B


# --------------------------------------------------------------------------
# the boot guarantee
# --------------------------------------------------------------------------

def test_existing_databases_migrate_in_place(tmp_db):
    """A db written before this table existed must gain it silently."""
    old = TrainingStore(tmp_db)
    old._conn.execute("DROP TABLE config_revisions")
    old._conn.commit()
    old.close()
    reopened = TrainingStore(tmp_db)          # must recreate, not raise
    assert reopened.revisions() == []
    reopened.close()
