"""Re-timing a stream so it can be read as it arrives.

Providers batch tokens before flushing: measured against the frontier tier,
~101 characters per SSE event, one event every ~94ms. A whole sentence lands at
once, so the reader gets a series of dumps rather than a stream.

The two properties that matter are in tension, and both are pinned here:

    NOTHING IS LOST     re-timing must be invisible in the payload. The text
                        out equals the text in, chunk metadata survives, and
                        chunks carrying no text -- role headers, tool_calls,
                        finish_reason -- pass through untouched, or tool
                        calling breaks.

    NOTHING IS SLOWED   the schedule is absolute, not per-chunk. A model that
                        is already slower than the target rate must never be
                        made slower still; smoothing only ever spreads bursts.
"""
import time

import pytest

from smartrouter.server import _smooth, _delta_content, SMOOTH_DEFAULT_RATE

FAST = 100_000.0   # effectively no sleeping, for payload-shape assertions


def chunk(text=None, **extra):
    delta = {} if text is None else {"content": text}
    delta.update(extra.pop("delta", {}))
    return {"id": "c1", "model": "frontier", "choices": [{"index": 0, "delta": delta, **extra}]}


def text_of(chunks):
    return "".join(_delta_content(c) for c in chunks)


# --------------------------------------------------------------------------
# nothing is lost
# --------------------------------------------------------------------------

def test_text_survives_exactly():
    src = [chunk("Hello, "), chunk("world. "), chunk("Two revisions.")]
    assert text_of(_smooth(src, rate=FAST)) == "Hello, world. Two revisions."


def test_long_text_survives_exactly():
    body = "".join(f"line {i} of the answer. " for i in range(40))
    assert text_of(_smooth([chunk(body)], rate=FAST)) == body


def test_chunk_metadata_is_preserved_on_every_fragment():
    out = list(_smooth([chunk("abcdefghij")], rate=FAST))
    assert len(out) > 1, "a ten-character chunk should be split"
    for frag in out:
        assert frag["model"] == "frontier"
        assert frag["id"] == "c1"
        assert frag["choices"][0]["index"] == 0


def test_source_chunk_is_not_mutated():
    """Fragments share structure with the original; sharing must be read-only."""
    src = chunk("abcdef")
    list(_smooth([src], rate=FAST))
    assert src["choices"][0]["delta"]["content"] == "abcdef"


@pytest.mark.parametrize("passthrough", [
    chunk(None, delta={"role": "assistant"}),
    chunk(None, finish_reason="stop"),
    chunk(None, delta={"tool_calls": [{"index": 0, "id": "t1"}]}),
])
def test_chunks_without_text_pass_through_identically(passthrough):
    """Tool calling and stop reasons must not be re-shaped."""
    out = list(_smooth([passthrough], rate=FAST))
    assert out == [passthrough]


def test_ordering_is_preserved_around_control_chunks():
    src = [chunk(None, delta={"role": "assistant"}), chunk("hi"), chunk(None, finish_reason="stop")]
    out = list(_smooth(src, rate=FAST))
    assert out[0] == src[0]
    assert out[-1] == src[-1]
    assert text_of(out) == "hi"


def test_rate_zero_is_an_exact_passthrough():
    """The off switch must not even copy chunks -- callers compare identity."""
    src = [chunk("a"), chunk(None, finish_reason="stop")]
    assert list(_smooth(src, rate=0)) == src


def test_empty_stream():
    assert list(_smooth([], rate=FAST)) == []


# --------------------------------------------------------------------------
# nothing is slowed
# --------------------------------------------------------------------------

def test_a_burst_is_spread_over_roughly_the_target_rate():
    body = "x" * 600
    t0 = time.monotonic()
    list(_smooth([chunk(body)], rate=2000.0))
    elapsed = time.monotonic() - t0
    assert 0.15 < elapsed < 0.60, f"600 chars at 2000/s should take ~0.3s, took {elapsed:.2f}s"


def test_a_model_slower_than_the_rate_is_not_slowed_further():
    """The schedule is absolute. A generator that already lags it must pass
    straight through -- otherwise smoothing would compound the local tier's
    ~28s p50 rather than leave it alone."""
    # 120 chars produced over ~0.36s = ~330 c/s, genuinely slower than the
    # 2000 c/s target. The schedule wants 0.06s and is met long before the
    # source delivers, so no sleep should ever fire.
    def slow():
        for _ in range(4):
            time.sleep(0.09)
            yield chunk("x" * 30)

    t0 = time.monotonic()
    list(_smooth(slow(), rate=2000.0))
    elapsed = time.monotonic() - t0
    assert elapsed < 0.50, f"smoothing added delay to an already-slow stream ({elapsed:.2f}s)"


def test_default_rate_is_readable_not_glacial():
    """A regression guard on the constant itself: 300 c/s renders a 2000-char
    answer in under ten seconds."""
    assert 150 <= SMOOTH_DEFAULT_RATE <= 600
    assert 2000 / SMOOTH_DEFAULT_RATE < 10
