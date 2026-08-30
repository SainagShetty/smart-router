"""FastAPI server: run smartrouter once, let every service point at it.

Endpoints
  GET  /health                 liveness + the loaded models/tiers
  POST /v1/chat/completions    OpenAI-compatible; routes + calls the model
  POST /route                  decision only (no provider call)
  POST /feedback               attach a label to a decision_id
  GET  /stats                  decision-log totals + tier distribution

One shared RouterCore (one classifier, one central decision log, provider keys in
one place). Sync endpoints run in the server's threadpool; the decision store is
thread-safe.

Auth: optional bearer token via the SMARTROUTER_API_KEY env var — recommended,
since this process holds your provider keys. When the token is set, direct
loopback callers are exempt by default: they already run on this host, where the
provider keys sit in this process's environment, so a token in front of them adds
no protection while breaking every co-located service. Requests arriving through a
reverse proxy (tailscale serve, Caddy) also connect over loopback, so they are
identified by their forwarding header (X-Forwarded-For / Forwarded) and must
present the token. Set SMARTROUTER_TRUST_LOOPBACK=0 to require it from everyone.

That default assumes anything able to open a loopback socket on this host is
already trusted. On a shared or multi-tenant machine that is not true — set
SMARTROUTER_TRUST_LOOPBACK=0 there and give each client the token.

Run it:
    smartrouter serve --config router.yaml --host 127.0.0.1 --port 4000
or:
    SMARTROUTER_CONFIG=router.yaml uvicorn smartrouter.server:app
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from . import __version__
from .config import (RouterConfig, capability_regressions, render_revision,
                     write_rendered)
from .core import RouterCore
from .errors import ConfigNotHotReloadable, NoEligibleModel, ProviderError

# Imported at module scope on purpose: `from __future__ import annotations` makes
# every annotation a string, and FastAPI resolves the `request: Request` hint in
# require_auth against these globals -- a create_app-local import would leave it
# unresolvable and get silently treated as a body field (422 on every call).
# Guarded so importing this module without the [server] extra still works;
# create_app raises the actionable ImportError in that case.
try:
    from starlette.requests import Request
except ImportError:  # pragma: no cover - optional extra
    Request = Any  # type: ignore[misc,assignment]



# --------------------------------------------------------------------------
# stream smoothing
# --------------------------------------------------------------------------

SMOOTH_DEFAULT_RATE = 300.0   # characters per second
SMOOTH_PIECE = 3              # characters per emitted fragment


def _delta_content(chunk: Dict[str, Any]) -> str:
    """The assistant text carried by an OpenAI-shaped stream chunk, or ''."""
    try:
        return chunk["choices"][0]["delta"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


def _rechunk(chunk: Dict[str, Any], text: str) -> Dict[str, Any]:
    """`chunk` with its delta content replaced. Copies only the four nodes on
    the path to `content`; the rest is shared, which is safe because nothing
    downstream mutates a chunk it has already yielded."""
    choice = dict(chunk["choices"][0])
    choice["delta"] = {**choice["delta"], "content": text}
    return {**chunk, "choices": [choice] + list(chunk["choices"][1:])}


def _smooth(
    chunks: Iterable[Dict[str, Any]],
    rate: float = SMOOTH_DEFAULT_RATE,
    piece: int = SMOOTH_PIECE,
) -> Iterator[Dict[str, Any]]:
    """Re-time a stream so text arrives at a readable pace.

    WHY THIS EXISTS. Providers batch tokens before flushing. Measured against
    the frontier tier: ~101 characters per SSE event, one event every ~94ms --
    so a whole sentence materialises at once and the reader gets a series of
    dumps rather than a stream. Splitting is not enough on its own: fragments
    written back-to-back leave in one TCP segment and land together. The pacing
    is the point.

    Fragments are released against a schedule of `rate` characters per second,
    anchored at the first fragment. Because the schedule is absolute rather
    than per-chunk, a slow model is never slowed further -- if generation is
    already behind the schedule the sleep is skipped and output is passed
    through as fast as it arrives. It only ever spreads bursts.

    Chunks carrying no text -- role headers, tool_calls, finish_reason -- pass
    through untouched and immediately, so tool calling and stop reasons are
    unaffected.

    Cost: a response longer than rate x its natural duration takes longer to
    finish rendering. That is the requested behaviour, and `rate` tunes it.
    """
    if rate <= 0:
        yield from chunks
        return

    start: Optional[float] = None
    emitted = 0

    for chunk in chunks:
        text = _delta_content(chunk)
        if not text:
            yield chunk
            continue
        for i in range(0, len(text), piece):
            fragment = text[i:i + piece]
            if start is None:
                start = time.monotonic()
            else:
                due = start + emitted / rate
                ahead = due - time.monotonic()
                if ahead > 0:
                    time.sleep(ahead)
            emitted += len(fragment)
            yield _rechunk(chunk, fragment)


class ChatRequest(BaseModel):
    # extra="allow" so standard OpenAI top-level params (temperature, max_tokens,
    # top_p, stop, seed, ...) sent by the OpenAI SDK or any OpenAI-shaped client
    # are captured and forwarded to the provider instead of being dropped.
    model_config = {"extra": "allow"}

    messages: List[Dict[str, Any]]
    # Historically accepted and ignored. Now: an exact TIER NAME pins that tier,
    # anything else -- including "auto" and any real model id -- is ignored as
    # before. That keeps the nine existing consumers (all of which send "auto")
    # working untouched, while letting a client whose UI has a model dropdown
    # use it to choose a tier and label the result honestly.
    model: Optional[str] = None
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
    # Presentation, not routing. Declared for the same reason as the logging
    # controls above: an undeclared field lands in model_extra and would be
    # forwarded to the provider, which rejects unknown params.
    #
    # bool BEFORE float in the union is load-bearing: as a bare Optional[float]
    # pydantic coerces `true` to 1.0, which is a rate of one character per
    # second rather than "on".
    smooth_stream: Optional[Union[bool, float]] = None
    # explicit passthrough params, for clients that prefer to nest them
    params: Dict[str, Any] = Field(default_factory=dict)


class RouteRequest(BaseModel):
    messages: List[Dict[str, Any]]
    # Present so /route produces the SAME decision /v1/chat/completions would.
    # /route exists to preview routing without paying for it; a field one
    # honours and the other ignores makes it lie.
    model: Optional[str] = None
    tools: Optional[List[Any]] = None
    response_format: Optional[Any] = None
    force_tier: Optional[str] = None
    cheap_only: bool = False
    local_only: bool = False
    cascade: Optional[bool] = None
    log_raw: Optional[bool] = None
    source: Optional[str] = None
    sensitive: bool = False


class ConfigCandidate(BaseModel):
    yaml: str
    author: Optional[str] = None
    note: Optional[str] = None
    replay_limit: int = 1000


class FeedbackRequest(BaseModel):
    decision_id: str
    label: int
    source: str = "api"


# Loopback source addresses that skip bearer auth when `trust_loopback` is on.
# IPv6-mapped IPv4 shows up when the socket is dual-stack.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})


def trust_loopback_from_env() -> bool:
    """Whether on-box callers skip bearer auth. Both entry points read this."""
    return os.environ.get("SMARTROUTER_TRUST_LOOPBACK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def create_app(
    config: RouterConfig,
    api_key: Optional[str] = None,
    trust_loopback: bool = True,
    config_path: Optional[str] = None,
    admin_token: Optional[str] = None,
):
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

    def require_auth(
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        if not api_key:
            return
        # Direct loopback callers are already on this host, where the provider
        # keys sit in this process's environment -- a bearer token in front of
        # them buys nothing, and requiring one would break every co-located
        # service. Reverse proxies (tailscale serve, Caddy) also connect over
        # loopback, so the exemption additionally requires that no forwarding
        # header is present: those requests originated off-box and must
        # authenticate. Set SMARTROUTER_TRUST_LOOPBACK=0 to require the token
        # from everyone, loopback included.
        client = request.client.host if request.client else None
        forwarded = (
            "x-forwarded-for" in request.headers
            or "forwarded" in request.headers
        )
        if trust_loopback and client in _LOOPBACK_HOSTS and not forwarded:
            return
        expected = f"Bearer {api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    def _tier_from_model(req) -> Optional[str]:
        """A `model` that names a tier means "pin this tier".

        Exact match only, and only against tiers this config actually declares.
        A model id, "auto", or anything unrecognised falls through to None and
        the router decides -- which is what every existing consumer relies on.
        An explicit force_tier in the body always wins over this.
        """
        name = (getattr(req, "model", None) or "").strip()
        if not name or name == "auto":
            return None
        return name if name in core.config.tier_order() else None

    def _overrides(req) -> Dict[str, Any]:
        return {
            "tools": req.tools,
            "response_format": req.response_format,
            "force_tier": req.force_tier or _tier_from_model(req),
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
            # core.config, not the closure variable: the config can be
            # swapped at runtime, and a /health that reports the config this
            # process STARTED with rather than the one it is serving is worse
            # than no /health at all.
            "tiers": core.config.tier_order(),
            "models": [m.id for m in core.config.models],
            "revision": core.config.revision,
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

                if req.smooth_stream:
                    rate = (SMOOTH_DEFAULT_RATE
                            if isinstance(req.smooth_stream, bool)
                            else float(req.smooth_stream))
                    chunks = _smooth(chunks, rate=rate)

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
        live = core.config          # not the closure variable -- see /health
        local_providers = {n for n, p in live.providers.items() if p.is_local()}
        local_models = {m.id for m in live.models if m.provider in local_providers}
        return {
            "total": core.store.count(),
            "labeled": core.store.count(labeled_only=True),
            "by_tier": core.store.tier_counts(),
            "savings": core.store.savings_summary(local_models=local_models),
        }

    # ---- admin ------------------------------------------------------------
    #
    # A SECOND auth tier, deliberately not the one above.
    #
    # require_auth exempts loopback callers, and that is right for /v1/*: ten
    # co-located services depend on it, and a token in front of them protects
    # nothing they could not already read out of this process's environment.
    #
    # It is wrong here. These endpoints rewrite where every LLM call in the
    # fleet goes. Under require_auth, any process on this box -- including an
    # SSRF in any of the ten, one of which is a 3,000-package Node app on the
    # public internet -- could repoint the router with no credential at all.
    # Reading is not writing, so they do not share a gate.
    #
    # The browser never holds this token: Caddy injects it on the way through,
    # so Cloudflare Access SSO is the first gate and the token is the second.
    # An on-box forger has neither.

    def require_admin(authorization: Optional[str] = Header(default=None)):
        if not admin_token:
            raise HTTPException(
                status_code=503,
                detail="admin API is disabled: set SMARTROUTER_ADMIN_TOKEN",
            )
        if authorization != f"Bearer {admin_token}":
            raise HTTPException(
                status_code=401, detail="invalid or missing admin token"
            )

    ADMIN = [Depends(require_admin)]

    def _candidate(yaml_text: str) -> RouterConfig:
        """Parse and validate a candidate config, or 400 with the reason."""
        import yaml as _yaml
        try:
            data = _yaml.safe_load(yaml_text)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"not valid YAML: {exc}")
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="config must be a mapping")
        for name, prov in (data.get("providers") or {}).items():
            if isinstance(prov, dict) and prov.get("api_key"):
                # A literal key here would be written to the revision table --
                # which is gzipped into ~/Backups nightly and kept 14 days.
                raise HTTPException(
                    status_code=400,
                    detail=f"provider {name!r} carries a literal api_key; use "
                           "api_key_env so the credential stays out of the "
                           "revision history",
                )
        try:
            return RouterConfig.from_dict(data)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid config: {exc}")

    @app.get("/admin/", dependencies=ADMIN)
    @app.get("/admin", dependencies=ADMIN)
    def admin_index():
        from fastapi.responses import FileResponse
        from pathlib import Path as _P
        return FileResponse(str(_P(__file__).parent / "ui" / "index.html"))

    @app.get("/admin/api/config", dependencies=ADMIN)
    def admin_get_config():
        """Everything the UI needs to render, in one call.

        Deliberately self-contained. The Caddy front publishes /admin* and 404s
        everything else, so /v1 and the provider keys behind it can never leak
        onto that hostname -- which also means a page served there CANNOT reach
        /health or /stats. An earlier version of the UI called both and rendered
        nothing at all in production, because one 404 rejected the Promise.all.
        Anything /admin needs lives under /admin.
        """
        active = core.store.active_revision() if core.store else None
        stats = None
        if core.store:
            by_tier = core.store.tier_counts()
            stats = {"by_tier": by_tier, "total": sum(by_tier.values())}
        return {
            "running_revision": core.config.revision,
            "active_revision": active["id"] if active else None,
            "yaml": active["yaml"] if active else None,
            "config": core.config.model_dump(mode="json"),
            "stats": stats,
        }

    @app.get("/admin/api/decisions", dependencies=ADMIN)
    def admin_decisions(limit: int = 25, source: Optional[str] = None):
        if not core.store:
            return {"decisions": [], "note": "logging is disabled"}
        return {"decisions": core.store.recent_decisions(limit=limit, source=source)}

    @app.get("/admin/api/revisions", dependencies=ADMIN)
    def admin_revisions(limit: int = 50):
        if not core.store:
            return {"revisions": [], "note": "logging is disabled"}
        return {"revisions": core.store.revisions(limit=limit),
                "running_revision": core.config.revision}

    @app.post("/admin/api/preview", dependencies=ADMIN)
    def admin_preview(req: ConfigCandidate):
        """What this candidate would do -- without doing any of it."""
        candidate = _candidate(req.yaml)
        active = core.store.active_revision() if core.store else None
        current_yaml = active["yaml"] if active else None
        diff = []
        if current_yaml is not None:
            import difflib
            diff = list(difflib.unified_diff(
                current_yaml.splitlines(), req.yaml.splitlines(),
                fromfile=f"revision {active['id']}", tofile="candidate",
                lineterm="", n=3))
        return {
            "diff": diff,
            "capability_regressions": capability_regressions(core.config, candidate),
            "replay": core.replay(candidate, limit=req.replay_limit),
        }

    @app.post("/admin/api/apply", dependencies=ADMIN)
    def admin_apply(req: ConfigCandidate):
        """Put a candidate into force.

            validate ──▶ render ──▶ reload ──▶ commit
                             │         │          │
                         fails: nothing changed at all
                                       │          │
                                   fails: re-render the previous revision
                                          and reload it back
                                                  │
                                              fails: file and router lead the
                                              DB by one, which /health reports

        The order is the design. The router boots from the RENDERED FILE, so
        that file is the record of what runs -- changing it first means the
        database can only ever LAG the running router, never lead it. A lagging
        database is visible and harmless; a leading one is a UI that shows you a
        config which was never applied, and this repo has already lost days to a
        setting believed live that was not.
        """
        if not core.store:
            raise HTTPException(status_code=503,
                                detail="logging is disabled; revisions need it")
        if not config_path:
            raise HTTPException(status_code=503,
                                detail="no config_path; cannot render")

        candidate = _candidate(req.yaml)
        try:
            # ASK, do not do. An earlier draft called reload() here as its dry
            # run; a successful reload is not dry -- it swaps the live config,
            # so a later render failure left the router serving an un-rendered,
            # uncommitted candidate reporting revision: null.
            core.check_reloadable(candidate)
        except ConfigNotHotReloadable as exc:
            rid = core.store.add_revision(req.yaml, author=req.author, note=req.note)
            raise HTTPException(
                status_code=409,
                detail={"error": str(exc), "saved_revision": rid,
                        "note": "saved as history; restart smart-router to apply"},
            )

        previous_yaml = None
        active = core.store.active_revision()
        if active:
            previous_yaml = active["yaml"]

        revision_id = core.store.add_revision(req.yaml, author=req.author,
                                              note=req.note)
        try:
            write_rendered(config_path, render_revision(req.yaml, revision_id))
        except OSError as exc:
            raise HTTPException(status_code=500,
                                detail=f"could not write {config_path}: {exc}")
        try:
            core.reload(RouterConfig.from_yaml(config_path))
        except Exception as exc:
            if previous_yaml is not None and active:
                write_rendered(config_path,
                               render_revision(previous_yaml, active["id"]))
                core.reload(RouterConfig.from_yaml(config_path))
            raise HTTPException(status_code=500,
                                detail=f"reload failed, previous config restored: {exc}")
        core.store.activate_revision(revision_id)
        return {"revision": revision_id, "running_revision": core.config.revision}

    @app.post("/admin/api/revert/{revision_id}", dependencies=ADMIN)
    def admin_revert(revision_id: int):
        if not core.store:
            raise HTTPException(status_code=503, detail="logging is disabled")
        row = core.store.revision(revision_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown revision")
        return admin_apply(ConfigCandidate(
            yaml=row["yaml"], author="revert",
            note=f"revert to revision {revision_id}"))

    return app


def _app_from_env():
    path = os.environ.get("SMARTROUTER_CONFIG", "router.yaml")
    config = RouterConfig.from_yaml(path)
    return create_app(
        config,
        api_key=os.environ.get("SMARTROUTER_API_KEY"),
        trust_loopback=trust_loopback_from_env(),
        config_path=path,
        admin_token=os.environ.get("SMARTROUTER_ADMIN_TOKEN"),
    )


def __getattr__(name):
    # Lazily build `app` so `uvicorn smartrouter.server:app` works, without
    # constructing a RouterCore just by importing this module.
    if name == "app":
        return _app_from_env()
    raise AttributeError(name)
