"""FastAPI server: run smartrouter once, let every service point at it.

Endpoints
  GET  /health                 liveness + the loaded models/tiers
  POST /v1/chat/completions    OpenAI-compatible; routes + calls the model
  POST /route                  decision only (no provider call)
  POST /feedback               attach a label to a decision_id
  GET  /stats                  decision-log totals + tier distribution

One shared RouterCore (one classifier, one central decision log, provider keys in
one place). Sync endpoints run in the server's threadpool; the decision store is
thread-safe. Optional bearer auth via the SMARTROUTER_API_KEY env var — recommended,
since this process holds your provider keys.

Run it:
    smartrouter serve --config router.yaml --host 127.0.0.1 --port 4000
or:
    SMARTROUTER_CONFIG=router.yaml uvicorn smartrouter.server:app
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from . import __version__
from .config import RouterConfig
from .core import RouterCore
from .errors import NoEligibleModel, ProviderError


class ChatRequest(BaseModel):
    # extra="allow" so standard OpenAI top-level params (temperature, max_tokens,
    # top_p, stop, seed, ...) sent by the OpenAI SDK or any OpenAI-shaped client
    # are captured and forwarded to the provider instead of being dropped.
    model_config = {"extra": "allow"}

    messages: List[Dict[str, Any]]
    model: Optional[str] = None  # accepted & ignored; the router decides
    stream: bool = False
    tools: Optional[List[Any]] = None
    response_format: Optional[Any] = None
    force_tier: Optional[str] = None
    cheap_only: bool = False
    local_only: bool = False
    cascade: Optional[bool] = None
    # logging controls (declared so they never leak into model_extra -> provider):
    # log_raw overrides the config default per request; source tags which service
    # sent it; sensitive marks rows for one-filter exclusion from training exports.
    log_raw: Optional[bool] = None
    source: Optional[str] = None
    sensitive: bool = False
    # explicit passthrough params, for clients that prefer to nest them
    params: Dict[str, Any] = Field(default_factory=dict)


class RouteRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tools: Optional[List[Any]] = None
    response_format: Optional[Any] = None
    force_tier: Optional[str] = None
    cheap_only: bool = False
    local_only: bool = False
    cascade: Optional[bool] = None
    log_raw: Optional[bool] = None
    source: Optional[str] = None
    sensitive: bool = False


class FeedbackRequest(BaseModel):
    decision_id: str
    label: int
    source: str = "api"


def create_app(config: RouterConfig, api_key: Optional[str] = None):
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
    except Exception as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "the server requires `pip install smartrouter[server]`"
        ) from exc

    core = RouterCore(config)
    app = FastAPI(title="smartrouter", version=__version__)
    app.state.core = core

    def require_auth(authorization: Optional[str] = Header(default=None)):
        if not api_key:
            return
        expected = f"Bearer {api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def _overrides(req) -> Dict[str, Any]:
        return {
            "tools": req.tools,
            "response_format": req.response_format,
            "force_tier": req.force_tier,
            "cheap_only": req.cheap_only,
            "local_only": req.local_only,
            "cascade": req.cascade,
            "log_raw": req.log_raw,
            "source": req.source,
            "sensitive": req.sensitive,
        }

    def _passthrough(req) -> Dict[str, Any]:
        # Standard OpenAI top-level params (temperature, max_tokens, ...) land in
        # model_extra; the explicit `params` dict wins on conflict.
        merged = {k: v for k, v in (req.model_extra or {}).items() if v is not None}
        merged.update(req.params or {})
        return merged

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "tiers": config.tier_order(),
            "models": [m.id for m in config.models],
        }

    @app.post("/route", dependencies=[Depends(require_auth)])
    def route(req: RouteRequest):
        try:
            d = core.route(req.messages, **_overrides(req))
        except NoEligibleModel as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {
            "decision_id": d.decision_id, "model": d.model, "provider": d.provider,
            "tier": d.tier, "score": d.score, "reason": d.reason,
            "candidates": d.candidates, "rejected": d.rejected,
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(require_auth)])
    def chat_completions(req: ChatRequest):
        ov = _overrides(req)
        passthrough = _passthrough(req)
        try:
            if req.stream:
                chunks, decision = core.stream(req.messages, **ov, **passthrough)

                def event_stream():
                    for chunk in chunks:
                        chunk.setdefault("decision_id", decision.decision_id)
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")

            resp, decision = core.complete(req.messages, **ov, **passthrough)
            resp["decision_id"] = decision.decision_id
            resp["routing"] = {"tier": decision.tier, "model": decision.model,
                               "score": decision.score, "reason": decision.reason}
            return resp
        except NoEligibleModel as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ProviderError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/feedback", dependencies=[Depends(require_auth)])
    def feedback(req: FeedbackRequest):
        updated = core.feedback(req.decision_id, req.label, req.source)
        if not updated:
            raise HTTPException(status_code=404, detail="unknown decision_id")
        return {"updated": True}

    @app.get("/stats", dependencies=[Depends(require_auth)])
    def stats():
        if not core.store:
            return {"logging": "disabled"}
        local_providers = {n for n, p in config.providers.items() if p.is_local()}
        local_models = {m.id for m in config.models if m.provider in local_providers}
        return {
            "total": core.store.count(),
            "labeled": core.store.count(labeled_only=True),
            "by_tier": core.store.tier_counts(),
            "savings": core.store.savings_summary(local_models=local_models),
        }

    return app


def _app_from_env():
    path = os.environ.get("SMARTROUTER_CONFIG", "router.yaml")
    config = RouterConfig.from_yaml(path)
    return create_app(config, api_key=os.environ.get("SMARTROUTER_API_KEY"))


def __getattr__(name):
    # Lazily build `app` so `uvicorn smartrouter.server:app` works, without
    # constructing a RouterCore just by importing this module.
    if name == "app":
        return _app_from_env()
    raise AttributeError(name)
