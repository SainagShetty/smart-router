# smartrouter — Specification

**Status:** draft v0.1 · **Owner:** Sainag Shetty · **License:** MIT · **Lang:** Python ≥3.9

---

## 1. Purpose

A small, dependency-light **routing layer** that sits in front of LLM providers and, per
request, sends it to the cheapest model that can actually handle it — escalating from a
**local** model (Ollama) → **cheap** cloud → **frontier** only when the query needs it.

Shipped as an open-source library (configurable provider, like `revive`) that also drops
into the owner's own services (Jarvis agents, finance auditor, etc.).

### The core question it answers
*"How does it know a query is too complex for the local model?"* → A **classifier** scores
difficulty up front (fast + precise, no extra LLM call). Cheap heuristics are kept **only**
as hard capability gates, not as the complexity signal. A bundled pre-fit head means it
works on day 0 with no training data; logged traffic later retrains/distills a better one.

## 2. Goals / Non-goals

**Goals**
- One decision per request: capability gate → difficulty score → tier → model.
- Two interfaces: explicit `route()` and an OpenAI-compatible drop-in client.
- Multi-provider, local↔cloud first-class. No LangChain; thin `httpx` adapters.
- Every decision logged to a **trainable** SQLite store (embedding + label).
- Pluggable classifier (embedding default, RouteLLM adapter, heuristic fallback).

**Non-goals (v1)**
- Not intent/tool routing (that's semantic-router's job).
- Not a training framework for routers from preference data (that's RouteLLM).
- No multi-step cascade/verify in v1 (classifier-first decision was chosen); the log is
  designed so cascade-style labels *can* be added later.
- No TS port in v1 (Python-first).

## 3. Routing pipeline

```
request (messages, tools?, response_format?, overrides?)
      │
      ▼
[1] features.extract        → RequestFeatures (tokens, has_images, needs_tools,
      │                         needs_json, num_turns, code_ratio, text, ...)
      ▼
[2] capabilities.gate       → candidate ModelSpecs that satisfy HARD requirements
      │                         (vision / tools / json / context window). Deterministic.
      ▼
[3] classifier.score        → difficulty ∈ [0,1]   (embedding + logreg head)
      ▼
[4] policy.decide           → pick tier by score band, then a model among gated
      │                         candidates; apply force_tier / cheap_only / max_cost.
      ▼
[5] provider.complete       → call model; on error, fallback chain (down/up/none).
      ▼
[6] logging_.record         → trainable row to SQLite (+ optional tracing).
      ▼
Decision / OpenAI-shaped response
```

**Invariant:** the classifier only chooses a *tier among capable models*. A hard
requirement (e.g. an image) always wins over the difficulty score.

## 4. Module map

| Module | Responsibility |
|---|---|
| `config.py` | Pydantic `RouterConfig`, `ModelSpec`, tiers, providers, classifier/policy/logging config; YAML/dict loaders; cross-reference validation. |
| `features.py` | Deterministic feature extraction from OpenAI-style messages. |
| `capabilities.py` | Hard-requirement gate → candidate models + rejection reasons. |
| `classifiers/base.py` | `Classifier` protocol: `score(features)->float`, optional `embed(text)`. |
| `classifiers/embedding.py` | **Default.** Embedding backend (`hashing` / `sentence-transformer`) + logreg head; `fit()`, `save/load`, `bundled()`. |
| `classifiers/heuristic.py` | Zero-dep fallback scorer. |
| `classifiers/routellm.py` | Optional adapter over lm-sys/RouteLLM checkpoints. |
| `policy.py` | Score + budget/overrides + candidates → `Decision`; tier bands; fallback order. |
| `providers/*` | `complete()`/`stream()` over OpenRouter, Ollama, OpenAI, Anthropic (httpx). |
| `core.py` | `RouterCore.route()` (explicit) + `feedback()`. |
| `client.py` | `RouterClient.chat.completions.create()` drop-in + streaming. |
| `logging_.py` | SQLite training store, implicit + explicit label capture, tracing hooks. |
| `train/` | `export.py`, `train_head.py`, `distill_embedder.py` (optional). |
| `eval/run_eval.py` | Offline tier-distribution + cost/quality harness. |

## 5. Public API

```python
from smartrouter import RouterConfig, RouterCore, RouterClient, Decision

config = RouterConfig.from_yaml("router.yaml")        # or .from_dict({...})

# Explicit
core = RouterCore(config)
d: Decision = core.route(messages, tools=None, response_format=None,
                         force_tier=None, cheap_only=False, local_only=False)
# d.model, d.tier, d.score, d.reason, d.candidates, d.decision_id
core.feedback(d.decision_id, label=1, source="manual")

# Drop-in (OpenAI-shaped) — overrides accepted as kwargs
client = RouterClient(config)
resp = client.chat.completions.create(messages=[...], stream=False,
                                      local_only=True)   # keep this call on-device
client.feedback(resp.decision_id, label=0)
```

### Per-request overrides

All optional; honored only when `policy.allow_overrides` is true. They restrict the
candidate set *after* the capability gate, so they never produce a model that can't serve
the request:

| Override | Effect |
|---|---|
| `force_tier="frontier"` | Pin to one exact tier; skip the classifier. |
| `cheap_only=True` | Cap at the cheapest cloud tier (no frontier). |
| `local_only=True` | Restrict to **local** providers (`provider.local`, defaults true for Ollama). Strict: if no local model survives the capability gate, raises `NoEligibleModel` rather than going to the cloud. |

**`Decision`** (dataclass): `decision_id, model, provider, tier, score, reason,
candidates, rejected, features, embedding(optional)`.

## 6. Config schema (YAML)

```yaml
providers:
  openrouter: { type: openrouter, base_url: https://openrouter.ai/api/v1, api_key_env: OPENROUTER_API_KEY }
  ollama:     { type: ollama, base_url: http://localhost:11434/v1, local: true }  # local: defaults true for ollama

tiers:                      # ordered cheapest -> most capable, by min_score
  - { name: local,    min_score: 0.0 }
  - { name: cheap,    min_score: 0.40 }
  - { name: frontier, min_score: 0.75 }

models:
  - { id: "llama3.1:8b", provider: ollama, tier: local, context_window: 8192,
      capabilities: { vision: false, tools: true, json: true } }
  - { id: "openai/gpt-4o-mini", provider: openrouter, tier: cheap, cost_per_1k: 0.0006,
      context_window: 128000, capabilities: { vision: true, tools: true, json: true } }
  - { id: "anthropic/claude-3.5-sonnet", provider: openrouter, tier: frontier,
      cost_per_1k: 0.009, context_window: 200000,
      capabilities: { vision: true, tools: true, json: true } }

classifier:
  kind: embedding           # embedding | heuristic | routellm
  embedding_backend: hashing  # hashing (no torch) | sentence-transformer
  # model_path: ./my_head.joblib   # optional: bring your own head

policy:
  fallback: down            # down | up | none
  allow_overrides: true
  default_tier: cheap
  # max_cost_per_1k: 0.01

logging:
  enabled: true
  db_path: smartrouter.db
  log_raw: false            # opt-in raw-prompt storage; never log secrets
  # tracing: langfuse
```

**Tier band rule:** choose the most-capable tier whose `min_score <= score`, restricted to
tiers that still have a gated candidate; otherwise fall to `policy.default_tier`.

## 7. Classifier detail

- **Default = embedding + logistic-regression head.** Output `P(hard) ∈ [0,1]`.
- **Embedding backends:**
  - `hashing` — scikit-learn `HashingVectorizer` (1024-d, L2). No torch. Ships the bundled
    `data/default_classifier.joblib`, trained on `data/difficulty_exemplars.jsonl`.
  - `sentence-transformer` — local MiniLM via `smartrouter[local-embed]` (higher quality).
- **`embedding_model_id`** (e.g. `hashing:1024`, model name) is persisted with each row so a
  retrain can reuse stored vectors.
- **Pluggable:** `routellm` adapter; user-fit heads; `heuristic` fallback.
- **Graceful failure:** classifier error → `policy.default_tier`, request still served.

## 8. Training-data store (the reason logs exist)

SQLite (WAL, `busy_timeout=5000`, `foreign_keys=ON`). One row per decision:

| Column | Notes |
|---|---|
| `decision_id` (PK) | uuid |
| `ts` | iso8601 |
| `prompt_sha256` | always stored |
| `prompt_raw` | only when `log_raw=true` (needed to re-embed for distillation) |
| `embedding` (blob) + `embedding_model_id` | reuse vectors on head-retrain |
| `features` (json) | extracted features |
| `score`, `chosen_tier`, `chosen_model`, `candidates` (json) | the decision |
| `cost`, `latency_ms` | observed |
| `label` (nullable int) + `label_source` | the trainable signal |

**Label capture**
- *Implicit (auto):* provider error/fallback fired, JSON/Pydantic validation failed, an
  escalation/regeneration happened → weak label.
- *Explicit:* `feedback(decision_id, label, source)`.

**`train/` pipeline**
- `export.py` — labeled rows → versioned dataset.
- `train_head.py` — refit logreg on stored embeddings → new `default_classifier.joblib`.
- `distill_embedder.py` — (optional, "faster model") distill a smaller embedder from
  `(prompt → label)` pairs, then swap as the default backend.

## 9. Providers

Unified `complete(model, messages, **params)` and `stream(...)` over `httpx`. OpenRouter &
Ollama both speak the OpenAI Chat Completions shape (Ollama via `/v1`), so one OpenAI-style
adapter covers them; OpenAI direct reuses it; Anthropic gets a thin shape-translation
adapter. API keys via direct value or `api_key_env`. No secrets logged.

## 10. Dependencies

Core: `pydantic≥2`, `httpx`, `scikit-learn`, `numpy`, `pyyaml`, `joblib`.
Extras: `[local-embed]` (sentence-transformers), `[routellm]`, `[langfuse]`, `[dev]` (pytest).

## 11. Verification

- **Unit:** capability gate (image→vision, tools, json, context overflow), policy bands +
  overrides, fallback chain, config validation, graceful classifier-failure default.
- **Offline eval:** `eval/run_eval.py` on `labeled_prompts.jsonl` → tier distribution +
  cost/quality vs a gold model (target: most quality retained at large cost cut).
- **Live smoke:** local=Ollama, frontier=OpenRouter; easy + hard + image prompt land in the
  expected tiers and return valid responses.
- **Training loop:** run requests → `feedback()`/implicit labels → `export` + `train_head`
  → asserts a fresh head is produced and loads. Proves the log is genuinely trainable.

## 12. Milestones

1. config ✓ · 2. features + gate ✓ · 3. classifiers + bundled head ·
4. policy · 5. providers · 6. core + client · 7. logging store + feedback ·
8. train pipeline · 9. routellm adapter · 10. eval + tests + README/examples + packaging.

## 13. Open items

- Package/repo name (working title `smartrouter`).
- Default embedding: ship hashing as default; promote sentence-transformer when `[local-embed]` present?
- Curating `difficulty_exemplars.jsonl` beyond the seed set (e.g. MT-Bench / Arena-style).
- Cost accounting source (static `cost_per_1k` vs provider usage echo).
