# smartrouter

**Pick the cheapest model that can actually handle each request — and escalate only when the query needs it.**

`smartrouter` is a small, dependency-light routing layer that sits in front of your LLM
providers. For every request it:

1. **Gates by capability** (deterministic) — a request with an image must go to a vision
   model; one that needs tools, JSON mode, or a large context window is filtered to models
   that support it. The difficulty classifier never overrides a hard requirement.
2. **Scores difficulty** with a pluggable classifier (default: a fast embedding + a
   pre-fit logistic-regression head, ships working with no training data).
3. **Routes to a tier** — `local` (e.g. Ollama) → `cheap` cloud → `frontier` — using
   configurable thresholds, with per-request overrides and a fallback chain on errors.
4. **Logs every decision** to a SQLite training store so you can later retune the
   classifier — or distill a smaller, faster one — on your own traffic.

Two interfaces, same core:

```python
from smartrouter import RouterCore, RouterClient, RouterConfig

config = RouterConfig.from_yaml("router.yaml")

# Explicit: see and adjust the decision.
decision = RouterCore(config).route(messages=[{"role": "user", "content": "2+2?"}])
print(decision.model, decision.tier, decision.score, decision.reason)

# Drop-in: one-line swap for an OpenAI/OpenRouter client.
client = RouterClient(config)
resp = client.chat.completions.create(messages=[{"role": "user", "content": "2+2?"}])
```

### Forcing where a request goes

The classifier picks a tier by default, but any single call can override it. Overrides
apply *after* the capability gate, so they never select a model that can't serve the
request:

```python
# Keep this call on-device regardless of difficulty.
core.route(messages, local_only=True)

# Pin an exact tier, or cap at the cheapest cloud tier.
core.route(messages, force_tier="frontier")
core.route(messages, cheap_only=True)

# Same overrides work on the drop-in client.
client.chat.completions.create(messages=[...], local_only=True)
```

`local_only` is strict: if no local model can satisfy the request's hard requirements
(e.g. an image with no local vision model), it raises `NoEligibleModel` instead of
silently falling back to the cloud. "Local" means any provider marked `local: true`
(default for Ollama).

## How it decides "too complex for the local model"

There's no perfect a-priori signal, so `smartrouter` makes the decision **up front with a
classifier** (fast + precise) and keeps cheap heuristics only as hard capability gates:

- **Default classifier** — embed the prompt and run a small logistic-regression head that
  outputs a difficulty score in `[0, 1]`. Sub-100ms, no extra LLM call, bundled pre-fit so
  it works on day 0. The default embedding backend is a lightweight scikit-learn
  `HashingVectorizer` (no torch); install `smartrouter[local-embed]` to use a
  sentence-transformer instead.
- **Pluggable** — swap in `RouteLLMClassifier` (wraps
  [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) checkpoints), retrain the embedding
  head on your own logs, or fall back to a zero-dependency heuristic classifier.

The thresholds map score bands to tiers; you tune them in config.

## Getting better over time

Every routed request is written to a SQLite store as a *trainable* row: the embedding
vector, extracted features, the score, the chosen model, cost/latency, and a nullable
**label**. Labels come from implicit signals (an escalation happened, JSON validation
failed, the user regenerated) or explicitly via `router.feedback(decision_id, label=...)`.
The `train/` scripts turn that corpus into a refreshed classifier head, or distill a
smaller/faster embedder that approximates your routing decisions.

## Run it as a shared server

For more than one service, run `smartrouter` once and point everything at it — so you get
one classifier, one central decision log (every service feeds the same training corpus),
and provider keys in a single process. The endpoint is OpenAI-compatible, so services swap
in by changing a base URL.

```bash
pip install "smartrouter[server]"
export OPENROUTER_API_KEY=...           # provider keys live here, in one place
export SMARTROUTER_API_KEY=...          # optional bearer token clients must send
smartrouter serve --config router.yaml --host 127.0.0.1 --port 4000
```

Endpoints: `POST /v1/chat/completions` (OpenAI-compatible, supports `stream`),
`POST /route` (decision only), `POST /feedback`, `GET /stats`, `GET /health`.

Any service then points its existing OpenAI/OpenRouter client at the router:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="<SMARTROUTER_API_KEY>")
resp = client.chat.completions.create(model="auto", messages=[...])  # router picks the model
```

Per-request overrides ride along in the JSON body (`local_only`, `force_tier`,
`cheap_only`). Deploy it as a daemon with the PM2 config in
[`examples/ecosystem.config.js`](examples/ecosystem.config.js). For one-off scripts you can
still embed the library directly (`RouterCore` / `RouterClient`) instead of running a server.

## Where it fits

- **vs [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)** — RouteLLM is a research
  framework for *training* routers on preference data. `smartrouter` is batteries-included
  and capability-aware, with first-class local↔cloud escalation, both a drop-in client and
  an explicit API, and decision logging. You can use a RouteLLM checkpoint as one backend.
- **vs [semantic-router](https://github.com/aurelio-labs/semantic-router)** — that does
  *intent* routing (which tool/route to take). `smartrouter` does *difficulty-tier*
  escalation. Different problem; they compose.

## License

MIT.
