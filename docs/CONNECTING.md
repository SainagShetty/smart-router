# Connecting to the smartrouter service

A shared model router runs on this machine. Instead of calling OpenRouter/Ollama
directly, send chat requests here and it picks the cheapest model that can handle each
request (local Gemma → cheap cloud → frontier), escalating only when needed. It speaks the
**OpenAI Chat Completions API**, so you connect by pointing an OpenAI-style client at it.

## Endpoint

- **Base URL:** `http://127.0.0.1:4000/v1`
- **Auth:** none right now (bound to localhost). Any non-empty `api_key` string is accepted.
  (If a bearer token is later enabled, send `Authorization: Bearer <token>`.)
- **Model field:** required by the OpenAI schema but **ignored** — the router decides. Pass
  `"auto"`.
- Standard sampling params work normally: `temperature`, `max_tokens`, `top_p`, `stop`,
  `seed`, `stream`, `tools`, `response_format`, etc.

## Quickest integration (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:4000/v1", api_key="smartrouter")

resp = client.chat.completions.create(
    model="auto",                       # ignored; router chooses the model
    messages=[{"role": "user", "content": "What's the capital of France?"}],
    max_tokens=50,
)
print(resp.choices[0].message.content)
# Which model was chosen + the id you can give feedback on:
print(resp.model_extra.get("routing"))      # {"tier","model","score","reason"}
print(resp.model_extra.get("decision_id"))
```

If your code already calls OpenRouter with the OpenAI SDK, the **only change** is
`base_url` (and you can drop the per-call `model=`, or leave any string — it's ignored).

## Raw HTTP (httpx / requests / curl)

```python
import httpx

r = httpx.post("http://127.0.0.1:4000/v1/chat/completions", json={
    "model": "auto",
    "messages": [{"role": "user", "content": "Prove sqrt(2) is irrational."}],
    "max_tokens": 200,
})
data = r.json()
text        = data["choices"][0]["message"]["content"]
decision_id = data["decision_id"]
routing     = data["routing"]   # {"tier","model","score","reason"}
```

## Forcing where a request goes (overrides)

Add these to the request body (raw HTTP) or via `extra_body={...}` (OpenAI SDK). They apply
*after* the capability check, so they never pick a model that can't serve the request.

| Field | Effect |
|---|---|
| `local_only: true` | Keep the call on the local model. Strict: returns HTTP 422 if no local model can serve it (e.g. an image with no local vision model). |
| `force_tier: "frontier"` | Pin to one exact tier (`local` \| `cheap` \| `frontier`); skips the classifier. |
| `cheap_only: true` | Cap at the cheapest cloud tier (never frontier). |

```python
# OpenAI SDK: keep a cheap intent-classification step on-device
client.chat.completions.create(model="auto", messages=msgs,
                               extra_body={"local_only": True})

# raw HTTP: force the best model for a user-facing final answer
httpx.post(URL, json={"model": "auto", "messages": msgs, "force_tier": "frontier"})
```

## Closing the loop (optional but valuable)

Every request is logged centrally with its routing decision. If you can tell that a chosen
model under-served a request (your validation failed, you had to retry, the user thumbed it
down), report it — the router uses these labels to improve over time:

```python
httpx.post("http://127.0.0.1:4000/feedback",
           json={"decision_id": decision_id, "label": 1, "source": "myservice"})
# label: 1 = was actually hard (should have escalated), 0 = was fine
```

## Streaming

Set `"stream": true`. The response is Server-Sent Events (`data: {json}\n\n`, terminated by
`data: [DONE]`), each chunk in OpenAI delta shape. The OpenAI SDK's `stream=True` works
directly.

## All endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible completion (routes + calls the model) |
| POST | `/route` | Return the routing decision **without** calling a model |
| POST | `/feedback` | Attach a label to a `decision_id` |
| GET | `/stats` | Decision-log totals + tier distribution |
| GET | `/health` | Liveness; lists loaded tiers + models |

## Current tiers / models (from the running config)

- `local` → `gemma3n:e4b` (Ollama, on-device)
- `cheap` → `xiaomi/mimo-v2.5` (OpenRouter)
- `frontier` → `google/gemini-3.1-flash-lite` (OpenRouter)

Check `GET /health` for the live list.

## Failure modes

- **422** — no model can serve the request under the active overrides (e.g. `local_only`
  with an image and no local vision model). The body's `detail` says why.
- **502** — every candidate provider failed (after the fallback chain). Body has the last error.

## Notes

- Don't put secrets in prompts — requests are logged (the prompt is hashed by default, not
  stored in plaintext, but treat the log as sensitive anyway).
- The router already has the provider API keys; your service does **not** need an
  OpenRouter key to use it.
