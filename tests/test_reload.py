"""Putting a new config into force without restarting.

Three properties, and each exists because getting it wrong fails silently:

    ATOMIC      a request sees the whole old config or the whole new one, never
                a mix. Guaranteed by _Live being frozen and assigned in one
                statement -- see test_live_snapshot.py for the read-count half.

    NARROW      providers, logging and classifier are refused rather than
                half-applied. Each holds live state (httpx clients, an open
                SQLite connection, a joblib-loaded model) that an in-flight
                request may be using.

    PRESERVING  the classifier and the store survive a reload by identity. They
                are expensive and stateful respectively; rebuilding them would
                drop an open connection on the floor and reload a model nobody
                asked to change.
"""
import pytest

from smartrouter import RouterCore
from smartrouter.errors import ConfigNotHotReloadable

from conftest import make_config, stub_providers

PROMPT = [{"role": "user", "content": "hi"}]


def cfg(**policy):
    return make_config(**policy)


# --------------------------------------------------------------------------
# what swaps
# --------------------------------------------------------------------------

def test_policy_change_takes_effect_immediately():
    core = RouterCore(cfg())
    assert core.config.policy.fallback == "down"
    core.reload(cfg(fallback="up"))
    assert core.config.policy.fallback == "up"


def test_model_change_takes_effect_immediately():
    core = RouterCore(cfg())
    before = {m.id for m in core.config.models}
    new = cfg()
    new.models = [m for m in new.models if m.tier != "frontier"]
    core.reload(new)
    assert {m.id for m in core.config.models} < before


def test_reload_swaps_the_whole_pair_at_once():
    """config and providers must never be observable half-swapped."""
    core = RouterCore(cfg())
    old_pair = core._live
    core.reload(cfg(fallback="up"))
    assert core._live is not old_pair
    assert core._live.config is not old_pair.config
    # providers carried over by identity -- nothing was rebuilt
    assert core._live.providers is old_pair.providers


# --------------------------------------------------------------------------
# what is refused
# --------------------------------------------------------------------------

@pytest.mark.parametrize("section", ["providers", "logging", "classifier"])
def test_unsafe_sections_are_refused_not_half_applied(section):
    core = RouterCore(cfg())
    before = core._live
    new = cfg()
    if section == "providers":
        new.providers["openrouter"].base_url = "https://example.invalid/v1"
    elif section == "logging":
        new.logging.db_path = "/tmp/somewhere-else.db"
    else:
        new.classifier.kind = "heuristic"

    with pytest.raises(ConfigNotHotReloadable) as ei:
        core.reload(new)
    assert section in str(ei.value)
    assert "restart" in str(ei.value).lower(), "must say how to apply it instead"
    assert core._live is before, "a refused reload must change nothing at all"


def test_refusal_leaves_the_router_serving():
    """The old config stays wholly in force -- not partially, not not-at-all."""
    core = RouterCore(cfg())
    stub_providers(core, content="still working")
    new = cfg()
    new.logging.db_path = "/tmp/elsewhere.db"
    with pytest.raises(ConfigNotHotReloadable):
        core.reload(new)
    resp, _ = core.complete(PROMPT)
    assert resp["choices"][0]["message"]["content"] == "still working"


# --------------------------------------------------------------------------
# what survives
# --------------------------------------------------------------------------

def test_classifier_and_store_survive_by_identity(tmp_db):
    """Rebuilding either would drop an open SQLite connection or reload a
    joblib model for a change that did not touch them."""
    core = RouterCore(make_config(db_path=tmp_db))
    classifier, store = core.classifier, core.store
    core.reload(make_config(db_path=tmp_db, fallback="up"))
    assert core.classifier is classifier
    assert core.store is store
    core.close()


def test_requests_keep_working_across_a_reload():
    core = RouterCore(cfg())
    stub_providers(core, content="ok")
    core.complete(PROMPT)
    core.reload(cfg(fallback="up"))
    resp, _ = core.complete(PROMPT)
    assert resp["choices"][0]["message"]["content"] == "ok", (
        "provider stubs are patched onto the provider objects; carrying the "
        "dict over by identity is what keeps them attached"
    )
