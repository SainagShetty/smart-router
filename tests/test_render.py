"""Turning a stored revision into the file the router boots from.

Two properties carry the whole design:

    COMMENTS SURVIVE   examples/router.yaml carries load-bearing prose. The
                       twelve lines explaining why gemma's context_window is
                       4096 and must not be raised are worth more than the
                       setting; a safe_load round trip would delete all of it
                       silently, and the next person would raise it again.

    THE STAMP IS REAL  `revision: N` is a validated top-level key, not a
                       comment. That is what lets a running process name what
                       it is serving by reading the file it already reads --
                       and so what keeps the database off the boot path, where
                       a lock at 03:30 could otherwise stop the gateway.
"""
import os

import pytest
import yaml as pyyaml

from smartrouter.config import RouterConfig, render_revision, write_rendered

CONFIG_WITH_PROSE = """\
# Why this file looks the way it does.
#
# The local tier is deliberately small.
providers:
  ollama:
    type: ollama
    base_url: http://localhost:11434/v1
    local: true

tiers:
  - name: local
    min_score: 0.0
  - name: cheap
    min_score: 0.40

models:
  - id: "gemma3n:e4b"
    provider: ollama
    tier: local
    # 4096, not the architectural 32768: Ollama allocates num_ctx and defaults
    # to 4096 regardless. Raising it was measured and REJECTED -- prefill runs
    # ~6.25 ms/token, so a 10k transcript costs ~65s against 1-4s in the cloud.
    context_window: 4096
  - id: "cheap-model"
    provider: ollama
    tier: cheap
    context_window: 128000

policy:
  fallback: down
"""


# --------------------------------------------------------------------------
# comments
# --------------------------------------------------------------------------

def test_render_preserves_every_comment():
    out = render_revision(CONFIG_WITH_PROSE, 7)
    for fragment in ("Why this file looks the way it does",
                     "The local tier is deliberately small",
                     "Ollama allocates num_ctx",
                     "measured and REJECTED",
                     "~6.25 ms/token"):
        assert fragment in out, f"lost comment: {fragment!r}"


def test_render_preserves_quoting():
    out = render_revision(CONFIG_WITH_PROSE, 1)
    assert '"gemma3n:e4b"' in out, "unquoting an id that needs quotes breaks it"


def test_render_does_not_rewrap_long_comment_lines():
    long_line = "# " + "x" * 200
    out = render_revision(long_line + "\n" + CONFIG_WITH_PROSE, 1)
    assert long_line in out


# --------------------------------------------------------------------------
# the stamp
# --------------------------------------------------------------------------

def test_stamp_is_a_validated_key_not_a_comment():
    out = render_revision(CONFIG_WITH_PROSE, 12)
    assert pyyaml.safe_load(out)["revision"] == 12
    cfg = RouterConfig.from_dict(pyyaml.safe_load(out))
    assert cfg.revision == 12


def test_restamping_replaces_rather_than_duplicates():
    once = render_revision(CONFIG_WITH_PROSE, 1)
    twice = render_revision(once, 2)
    assert pyyaml.safe_load(twice)["revision"] == 2
    assert twice.count("revision:") == 1


def test_a_stored_config_has_no_revision():
    """The id exists once the row does; a config being composed has none."""
    cfg = RouterConfig.from_dict(pyyaml.safe_load(CONFIG_WITH_PROSE))
    assert cfg.revision is None


# --------------------------------------------------------------------------
# atomic write
# --------------------------------------------------------------------------

def test_write_is_atomic_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "nested" / "router.yaml"
    write_rendered(str(target), render_revision(CONFIG_WITH_PROSE, 3))
    assert RouterConfig.from_yaml(str(target)).revision == 3
    strays = [f for f in os.listdir(target.parent) if f.endswith(".tmp")]
    assert strays == [], f"left temp files behind: {strays}"


def test_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    """A half-written config would stop the gateway booting, and a boot failure
    on the gateway takes ten services with it."""
    target = tmp_path / "router.yaml"
    write_rendered(str(target), render_revision(CONFIG_WITH_PROSE, 1))
    before = target.read_text()

    def boom(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_rendered(str(target), render_revision(CONFIG_WITH_PROSE, 2))

    assert target.read_text() == before, "previous config must survive untouched"
    strays = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert strays == [], "a failed write must not leave debris"
