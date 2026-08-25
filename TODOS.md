# TODOs

Known gaps, with enough context to pick each one up cold. Ordered by priority
within each section, not by discovery date.

## P1 — correctness

### `estimated_tokens` omits system and assistant turns

`features.extract` builds `feats.text` by joining only `user`, `tool` and
role-less messages (`src/smartrouter/features.py:111`), and `estimated_tokens`
is derived from that text. `capabilities.gate` then compares it against
`context_window` (`src/smartrouter/capabilities.py:43`).

So the context gate systematically under-counts. A ten-turn conversation
dominated by long assistant replies can measure ~2k estimated tokens against a
real ~8k prompt, sail through gemma's 4096 gate, and get silently truncated by
Ollama — the exact failure the 4096 declaration was lowered to prevent
(`examples/router.yaml`, commit `40e0fd8`).

Two parts to the fix: count all message roles, and reserve headroom for the
completion, since Ollama must fit prompt *plus* output inside `num_ctx`. A
4000-token prompt passes a 4096 gate today and then clips whatever it generates.

### `cascade` is forwarded to the provider on streamed calls

`server.py::_overrides()` always includes `"cascade"`, but `RouterCore.stream()`
has no `cascade` parameter, so it lands in `**params` → `call_params` →
`_payload` and ships to OpenRouter/Ollama as `"cascade": null` in the request
body. Two effects: an unknown field is sent upstream, and a streamed request can
never actually use cascade mode even though the API accepts the flag.

### Escalation costs are dropped when every spec fails

`complete()`'s cascade escalation `continue`s past `_finalize`, so tokens billed
by a discarded attempt never reach the decision row. `/stats` under-reports
spend in exactly the case where the most money was burned for nothing.

## P1 — security

### The loopback auth exemption fails open

`server.py::require_auth` exempts a caller when the peer address is loopback
**and** no `X-Forwarded-For`/`Forwarded` header is present. The header is the
only discriminator, so any loopback ingress that does not add one is fully
unauthenticated against a process holding provider keys:

- `ssh -L 4000:127.0.0.1:4000` from any device with SSH access
- `tailscale serve --tcp` / `--tls-terminated-tcp` (unlike the HTTP mode in use)
- a Caddy vhost configured with `header_up -X-Forwarded-For`
- **SSRF in any co-located service on this box** — the residual path the
  docstring's threat model does not currently mention. It covers only "shared or
  multi-tenant machine".

Each silently disables auth with no error and no log line, while
`docs/CONNECTING.md` tells the reader off-box access is token-protected.
Minimum: a startup warning when a token is configured but the exemption is
active, and treating a missing `SMARTROUTER_API_KEY` as a hard error when the
bind is not loopback.

### `SMARTROUTER_TRUST_LOOPBACK=` (empty) silently means "trust"

`trust_loopback_from_env()` tests `not in ("0", "false", "no")`, and `""` is not
in that set. Clearing the variable in a shell or setting `SMARTROUTER_TRUST_LOOPBACK: ""`
in the PM2 `env` block keeps the exemption **on** for someone who meant to turn
it off. `off`, `n` and `disabled` are unhandled too. Invert to an explicit
truthy allowlist.

## P2 — robustness

### A non-dict stream chunk crashes `stream()`

Providers yield whatever `json.loads` returns from an SSE `data:` line. `_collect`
(`core.py`, in `_emit`) calls `chunk.get(...)` unguarded, so a scalar or array
payload raises `AttributeError` out of `stream()`. `server.py` does not catch it,
so it surfaces as a 500 with no fallback. Guard with `isinstance(chunk, dict)`.

### `sensitive` / `local_only` prompts between ~4k and ~8k tokens now 422

Consequence of lowering gemma's declared `context_window` to 4096 (`40e0fd8`).
`policy.decide` filters the pool to local providers for sensitive requests and
raises `NoEligibleModel` when nothing survives, and there is no cloud fallback on
that path by design. Prompts in that band used to be served; they now fail with a
422. Correct in the sense that they would have been truncated before — but it is
a user-visible regression for sensitive callers, not graceful degradation.

## Abandoned — empty-answer guard (2026-08-24)

**Do not restart this without reading the whole entry.** Two full implementations
were written and both were dropped after review.

**The bug it targeted is real.** The cheap tier (`xiaomi/mimo-v2.5`) is a
reasoning model. With a small `max_tokens` it spends the whole budget on
reasoning tokens and stops with `content:null` and `finish_reason:"length"`
(observed live 2026-07-05). Nothing raises, so the empty answer returns as a
success and callers render it blank — a chat UI shows an empty assistant bubble
with no error and no log line.

**Why it was abandoned — the guard cannot reach the traffic that needs it.**
Retrying on another model requires the fallback chain to have another model in
it, and under the shipped configuration it usually does not:

```
  force_tier: <tier>   pins the pool to one tier  -> one-element chain
  sensitive: true      pins the pool to local     -> one-element chain
  fallback: down       local is the bottom tier   -> one-element chain
```

Measured against `examples/router.yaml`: a `force_tier: frontier` request, a
`sensitive` request, and any easy prompt routed local all get a chain of length
one. Without a retry target the guard must either do nothing, or convert a blank
200 into a hard 502 — and the 502 is worse than the bug for the roughly 85% of
traffic that routes local.

**Three further problems found in review, all real:**

1. Under `fallback: down` a retry is a *de-escalation* — the same exhausted
   `max_tokens` is handed to a weaker model, so a chain containing more than one
   reasoning model pays for every member and still returns a blank.
2. The streaming form cannot be caught without buffering, and buffering fights
   time-to-first-token: reasoning models stream `{"content": "", "reasoning": ...}`
   deltas for tens of seconds. A chunk/time cap bounds the damage but re-opens
   the hole it was added to close, and a wall-clock bound cannot be enforced from
   inside `for chunk in gen` — it only fires when a chunk *arrives*, never during
   the silence between chunks. Bounding the wait needs timeout-aware iteration.
3. Keying on emptiness alone false-positives on legitimately empty answers
   (moderation stop, refusal, a prompt that asks for nothing). Keying on
   `finish_reason == "length"` fixes that for `complete()` but has no clean
   streaming equivalent, since the terminal chunk arrives last.

**What actually mitigates it, and works today:** send a generous `max_tokens`
(the exhaustion never happens) and keep latency-budgeted interactive calls off
the cheap tier. Both are per-request, cost nothing, and need no router change.
This is the documented consumer guidance already — see `docs/CONNECTING.md`.

**If revisited,** the honest framings are (a) fix it at the provider adapter, by
treating a `length` finish with no content as a `ProviderError` where the
fallback chain already handles errors, or (b) make `force_tier` and the local
tier have real retry targets first — but that is a routing change for all nine
consumers and a much larger decision than this guard.

Abandoned commits are recoverable: `git cherry-pick 43ab94f 126c176`.

## Corrected record

`fallback` has been `down` in `examples/router.yaml` since the initial commit
`158dd6a`. It has never been `up` — `git log -S'fallback: up' --all` returns
nothing, on any branch. Notes elsewhere claiming the deployed policy was switched
to `up` on 2026-07-08 were wrong; that change was proven on an ephemeral instance
and never applied to the config PM2 loads. Anything reasoning about escalation
*from* the local tier must not assume an upward chain exists.
