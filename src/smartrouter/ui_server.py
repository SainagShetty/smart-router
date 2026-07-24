"""Minimal HTTP UI for editing a router.yaml config's tiers and models, plus a
read-only view of the decision log.

    smartrouter ui --config router.yaml --host 127.0.0.1 --port 4001

Reads/writes the same config file the shared `smartrouter serve` process loads.
The write path round-trips the YAML through ruamel and only touches fields that
actually changed, so comments and unrelated structure in the file survive a
save. Every save is validated against the full `RouterConfig` schema before
it's written, so a bad edit (unknown provider, duplicate tier, ...) is rejected
with a 400 instead of corrupting the file the live router loads.

Note: the running `serve` process loads its config once at startup, so a save
here does not take effect until that process is restarted.

`/api/stats` reads the same SQLite decision log `serve` writes to (a fresh,
short-lived connection per request — WAL mode makes this a safe concurrent
read) and reports the same totals/savings shape as `serve`'s own `/stats`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .config import RouterConfig
from .logging_ import TrainingStore

_UI_DIR = Path(__file__).resolve().parent / "ui"


class TierUpdate(BaseModel):
    name: str
    min_score: float = Field(ge=0.0, le=1.0)


class ModelUpdate(BaseModel):
    id: str
    provider: str
    tier: str
    cost_per_1k: float = Field(default=0.0, ge=0.0)
    context_window: int = Field(default=8192, gt=0)
    vision: bool = False
    tools: bool = False
    json_mode: bool = False
    # position of this row in the file *before* this save (None = new row).
    # Lets a save rename `id` in place instead of it looking like a
    # delete+add, and lets untouched rows keep their original formatting.
    orig_index: Optional[int] = None


class ConfigUpdate(BaseModel):
    # None = leave this section untouched (a client that only edits tiers
    # shouldn't have to resend every model, and omitting the field must never
    # be read as "delete everything"). An explicit [] means "make it empty".
    tiers: Optional[List[TierUpdate]] = None
    models: Optional[List[ModelUpdate]] = None


def _apply_tiers(doc, tiers: List[TierUpdate]) -> bool:
    existing = {tier["name"]: tier for tier in doc.get("tiers", [])}
    unknown = [t.name for t in tiers if t.name not in existing]
    if unknown:
        raise HTTPException(status_code=400,
                             detail=f"unknown tier(s): {', '.join(unknown)}")
    changed = False
    for t in tiers:
        if float(existing[t.name]["min_score"]) != t.min_score:
            existing[t.name]["min_score"] = t.min_score
            changed = True
    return changed


def _apply_model_fields(node: CommentedMap, m: ModelUpdate) -> bool:
    changed = False

    def _set(key, value, current):
        nonlocal changed
        if current != value:
            node[key] = value
            changed = True

    _set("id", m.id, node.get("id"))
    _set("provider", m.provider, node.get("provider"))
    _set("tier", m.tier, node.get("tier"))
    _set("cost_per_1k", m.cost_per_1k, float(node.get("cost_per_1k", 0.0)))
    _set("context_window", m.context_window, int(node.get("context_window", 8192)))

    caps = node.get("capabilities")
    if not isinstance(caps, CommentedMap):
        caps = CommentedMap()
        node["capabilities"] = caps
        changed = True
    if bool(caps.get("vision", False)) != m.vision:
        caps["vision"] = m.vision
        changed = True
    if bool(caps.get("tools", False)) != m.tools:
        caps["tools"] = m.tools
        changed = True
    if bool(caps.get("json", False)) != m.json_mode:
        caps["json"] = m.json_mode
        changed = True

    return changed


def _new_model_node(m: ModelUpdate) -> CommentedMap:
    node = CommentedMap()
    node["id"] = m.id
    node["provider"] = m.provider
    node["tier"] = m.tier
    node["cost_per_1k"] = m.cost_per_1k
    node["context_window"] = m.context_window
    caps = CommentedMap()
    caps["vision"] = m.vision
    caps["tools"] = m.tools
    caps["json"] = m.json_mode
    node["capabilities"] = caps
    return node


def _apply_models(doc, models: List[ModelUpdate]) -> bool:
    old = doc.get("models", [])
    changed = len(models) != len(old)
    new_seq = CommentedSeq()
    if isinstance(old, CommentedSeq):
        # carry over the sequence's own trailing comment/blank-line marker
        # (e.g. the blank line before the next top-level key) — rebuilding
        # the sequence loses it even when no items moved. Per-item comments
        # live on each item's own CommentedMap, already preserved above.
        new_seq.ca.end = old.ca.end
        new_seq.ca.comment = old.ca.comment
    for i, m in enumerate(models):
        if m.orig_index is not None and 0 <= m.orig_index < len(old):
            node = old[m.orig_index]
            if m.orig_index != i:
                changed = True
            if _apply_model_fields(node, m):
                changed = True
        else:
            node = _new_model_node(m)
            changed = True
        new_seq.append(node)

    if changed:
        doc["models"] = new_seq
    return changed


def create_app(config_path: str) -> FastAPI:
    config_path = Path(config_path)
    app = FastAPI()
    app.state.config_path = config_path

    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    yaml_rt.width = 4096  # avoid re-wrapping long lines on save

    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(_UI_DIR / "index.html"))

    @app.get("/api/config")
    async def get_config():
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"config file not found: {config_path}")
        with open(config_path) as f:
            return yaml.safe_load(f)

    @app.get("/api/stats")
    async def get_stats():
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"config file not found: {config_path}")
        try:
            config = RouterConfig.from_yaml(str(config_path))
        except (ValidationError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"config is invalid: {e}")

        if not config.logging.enabled:
            return {"logging": "disabled"}

        local_providers = {n for n, p in config.providers.items() if p.is_local()}
        local_models = {m.id for m in config.models if m.provider in local_providers}

        # A fresh, short-lived read against the same SQLite file the live
        # `smartrouter serve` process writes to (WAL mode supports concurrent
        # readers) — this UI process never holds its own long-lived store.
        store = TrainingStore(config.logging.db_path)
        try:
            return {
                "total": store.count(),
                "labeled": store.count(labeled_only=True),
                "by_tier": store.tier_counts(),
                "savings": store.savings_summary(local_models=local_models),
            }
        finally:
            store.close()

    @app.post("/api/config")
    async def update_config(update: ConfigUpdate):
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"config file not found: {config_path}")

        doc = yaml_rt.load(config_path.read_text())

        tiers_changed = _apply_tiers(doc, update.tiers) if update.tiers is not None else False
        models_changed = _apply_models(doc, update.models) if update.models is not None else False

        if tiers_changed or models_changed:
            try:
                RouterConfig.model_validate(doc)
            except ValidationError as e:
                raise HTTPException(status_code=400,
                                     detail=f"resulting config is invalid: {e}")
            try:
                with open(config_path, "w") as f:
                    yaml_rt.dump(doc, f)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        return {
            "status": "success",
            "note": "restart the router process for this to take effect "
                    "(e.g. `pm2 restart smart-router`)",
        }

    return app
