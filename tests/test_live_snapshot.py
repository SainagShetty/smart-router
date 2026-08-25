"""One read of `_live` per request, and nothing more.

The config can be swapped at runtime, which turns "read config, then read
providers" into a race:

    route()     reads config     ──┐   a reload landing between these two
    complete()  reads providers  ──┘   reads routes a request against config A
                                       and executes it against config B's
                                       providers. Remove a model while a
                                       request is in flight and the lookup
                                       KeyErrors into a 500.

`_Live` makes that unrepresentable by binding both from one read. But nothing
about the type *enforces* the single read -- someone adding a codepath could
reintroduce `self._live` a second time and the race comes back, silently.

So these tests assert the property directly rather than trying to provoke the
race. A threaded test would pass whenever the interleaving does not happen,
which is most runs, and would give false confidence; a read counter cannot pass
by luck.
"""
import pytest

from smartrouter import RouterCore
from smartrouter.core import _Live

from conftest import make_config, stub_providers

PROMPT = [{"role": "user", "content": "hi"}]


class CountingLive:
    """A data descriptor that counts reads of `_live`.

    Must define __set__ as well as __get__: a non-data descriptor is shadowed by
    the instance __dict__, and RouterCore.__init__ assigns self._live there, so
    a __get__-only class would never fire.
    """

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.reads = 0

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self.reads += 1
        return self.snapshot

    def __set__(self, obj, value):
        self.snapshot = value


@pytest.fixture
def counted(monkeypatch):
    """A core whose `_live` reads are counted."""
    core = RouterCore(make_config())
    stub_providers(core, content="ok")
    counter = CountingLive(core.__dict__["_live"])
    monkeypatch.setattr(RouterCore, "_live", counter, raising=False)
    return core, counter


def test_complete_reads_live_exactly_once(counted):
    core, counter = counted
    core.complete(PROMPT)
    assert counter.reads == 1, (
        f"complete() read _live {counter.reads} times; each extra read is a "
        "window where a reload can tear config away from providers"
    )


def test_stream_reads_live_exactly_once(counted):
    core, counter = counted
    gen, _ = core.stream(PROMPT)
    list(gen)
    assert counter.reads == 1, (
        f"stream() read _live {counter.reads} times"
    )


def test_bare_route_takes_its_own_snapshot(counted):
    """route() called directly still works -- it snapshots for itself."""
    core, counter = counted
    core.route(PROMPT)
    assert counter.reads == 1


# --------------------------------------------------------------------------
# the compatibility surface
# --------------------------------------------------------------------------

def test_config_and_providers_remain_readable_attributes():
    """Eight test files mutate core.providers in place; keep that working."""
    core = RouterCore(make_config())
    assert core.config is core._live.config
    assert core.providers is core._live.providers
    # the mutation pattern every stub uses
    for prov in core.providers.values():
        prov.complete = lambda *a, **k: None
    assert all(p.complete.__name__ == "<lambda>" for p in core.providers.values())


def test_live_is_frozen():
    """A pair that could be mutated in place would defeat the whole point."""
    core = RouterCore(make_config())
    with pytest.raises(Exception):
        core._live.config = None
