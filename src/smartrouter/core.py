"""RouterCore — the explicit routing API.

``route()`` makes (and logs) a decision without calling a model. ``complete()``
makes the decision, calls the chosen provider with a fallback chain, records
cost/latency, and captures implicit labels. The drop-in client builds on this.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import capabilities, features as features_mod
from .classifiers import build_classifier
from .config import ModelSpec, RouterConfig
from .errors import NoEligibleModel, ProviderError
from .logging_ import TrainingStore
from .policy import Overrides, decide
from .providers import build_provider

Message = Dict[str, Any]


@dataclass
class Decision:
    decision_id: str
    model: str           # provider-specific model id
    provider: str
    tier: str
    score: float
    reason: str
    candidates: List[str]
    rejected: List[str]
    features: Dict[str, Any]
    # Internal: the chosen spec + ordered fallback specs + embedding (not logged
    # in repr). Used by complete() and the training store.
    model_spec: Optional[ModelSpec] = field(default=None, repr=False)
    fallback_specs: List[ModelSpec] = field(default_factory=list, repr=False)
    embedding: Any = field(default=None, repr=False)
    # Effective raw-logging choice for this request (per-request override falling
    # back to config). complete()/stream() consult it before storing the response.
    log_raw: bool = field(default=False, repr=False)


class RouterCore:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.classifier = build_classifier(config.classifier)
        self.providers = {
            name: build_provider(name, pc) for name, pc in config.providers.items()
        }
        self.store: Optional[TrainingStore] = None
        if config.logging.enabled:
            self.store = TrainingStore(config.logging.db_path)

    # ---- decision ---------------------------------------------------------

    def route(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[Any]] = None,
        response_format: Optional[Any] = None,
        force_tier: Optional[str] = None,
        cheap_only: bool = False,
        local_only: bool = False,
        log_raw: Optional[bool] = None,
        source: Optional[str] = None,
        sensitive: bool = False,
    ) -> Decision:
        feats = features_mod.extract(
            messages, tools=tools, response_format=response_format
        )
        gate = capabilities.gate(self.config.models, feats)

        score, embedding = self._score(feats)

        result = decide(
            self.config,
            gate.candidates,
            score,
            Overrides(force_tier=force_tier, cheap_only=cheap_only,
                      local_only=local_only, sensitive=sensitive),
            rejected=gate.rejected,
        )

        decision = Decision(
            decision_id=str(uuid.uuid4()),
            model=result.model.id,
            provider=result.model.provider,
            tier=result.tier,
            score=round(score, 4),
            reason=result.reason,
            candidates=[m.id for m in gate.candidates],
            rejected=gate.rejected,
            features=_features_dict(feats),
            model_spec=result.model,
            fallback_specs=result.fallback,
            embedding=embedding,
            log_raw=self.config.logging.log_raw if log_raw is None else log_raw,
        )
        self._record(decision, feats, source=source, sensitive=sensitive)
        return decision

    def _score(self, feats) -> Tuple[float, Any]:
        try:
            return self.classifier.score_and_embed(feats)
        except Exception:
            # Graceful degradation: classifier failure must not drop the request.
            # Score 0 routes to the cheapest tier unless a default_tier is set,
            # which the caller can express via policy.default_tier.
            return 0.0, None

    def _record(
        self,
        decision: Decision,
        feats,
        source: Optional[str] = None,
        sensitive: bool = False,
    ) -> None:
        if not self.store:
            return
        self.store.record(
            decision_id=decision.decision_id,
            prompt=feats.text,
            embedding=decision.embedding,
            embedding_model_id=self.classifier.embedding_model_id,
            features=decision.features,
            score=decision.score,
            chosen_tier=decision.tier,
            chosen_model=decision.model,
            candidates=decision.candidates,
            log_raw=decision.log_raw,
            source=source,
            sensitive=sensitive,
            reason=decision.reason,
            rejected=decision.rejected,
        )

    # ---- execution --------------------------------------------------------

    def complete(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[Any]] = None,
        response_format: Optional[Any] = None,
        force_tier: Optional[str] = None,
        cheap_only: bool = False,
        local_only: bool = False,
        log_raw: Optional[bool] = None,
        source: Optional[str] = None,
        sensitive: bool = False,
        **params,
    ) -> Tuple[Dict[str, Any], Decision]:
        decision = self.route(
            messages, tools=tools, response_format=response_format,
            force_tier=force_tier, cheap_only=cheap_only, local_only=local_only,
            log_raw=log_raw, source=source, sensitive=sensitive,
        )
        chain = [decision.model_spec] + list(decision.fallback_specs)
        if tools is not None:
            params["tools"] = tools
        if response_format is not None:
            params["response_format"] = response_format

        last_err: Optional[Exception] = None
        for i, spec in enumerate(chain):
            provider = self.providers[spec.provider]
            call_params = dict(spec.params)
            call_params.update(params)
            t0 = time.time()
            try:
                resp = provider.complete(spec.id, list(messages), **call_params)
            except ProviderError as exc:
                last_err = exc
                continue
            latency_ms = (time.time() - t0) * 1000.0

            self._finalize(decision, spec, resp, latency_ms,
                           fell_back=(i > 0), response_format=response_format)
            return resp, decision

        raise ProviderError(
            f"all {len(chain)} candidate model(s) failed; last error: {last_err}"
        )

    def _finalize(self, decision, spec, resp, latency_ms, fell_back, response_format):
        if not self.store:
            return
        cost = _estimate_cost(resp, spec)
        # The response is the distillation target — stored under the same privacy
        # gate as the raw prompt (a stored answer would leak the prompt anyway).
        response_raw = _response_text(resp) if decision.log_raw else None
        self.store.update_outcome(
            decision.decision_id, cost=cost, latency_ms=latency_ms,
            chosen_model=spec.id, response_raw=response_raw,
        )

        # Implicit label: a JSON request whose output isn't valid JSON means the
        # chosen model under-served it -> weak "hard" signal.
        if response_format is not None and _wants_json(response_format):
            content = _first_content(resp)
            if content is not None and not _is_json(content):
                self.store.set_label_if_unset(
                    decision.decision_id, 1, "implicit:json_invalid"
                )

    def stream(
        self,
        messages: Sequence[Message],
        *,
        tools: Optional[Sequence[Any]] = None,
        response_format: Optional[Any] = None,
        force_tier: Optional[str] = None,
        cheap_only: bool = False,
        local_only: bool = False,
        log_raw: Optional[bool] = None,
        source: Optional[str] = None,
        sensitive: bool = False,
        **params,
    ) -> Tuple[Iterator[Dict[str, Any]], Decision]:
        """Return (chunk-iterator, Decision). Falls back if a provider errors
        before the first chunk."""
        decision = self.route(
            messages, tools=tools, response_format=response_format,
            force_tier=force_tier, cheap_only=cheap_only, local_only=local_only,
            log_raw=log_raw, source=source, sensitive=sensitive,
        )
        chain = [decision.model_spec] + list(decision.fallback_specs)
        if tools is not None:
            params["tools"] = tools
        if response_format is not None:
            params["response_format"] = response_format

        t0 = time.time()
        last_err: Optional[Exception] = None
        for spec in chain:
            provider = self.providers[spec.provider]
            call_params = dict(spec.params)
            call_params.update(params)
            gen = provider.stream(spec.id, list(messages), **call_params)
            try:
                first = next(gen)
            except StopIteration:
                first = None
            except ProviderError as exc:
                last_err = exc
                continue

            def _emit():
                pieces: List[str] = []

                def _collect(chunk):
                    if decision.log_raw:
                        piece = _delta_content(chunk)
                        if piece:
                            pieces.append(piece)

                if first is not None:
                    _collect(first)
                    yield first
                for chunk in gen:
                    _collect(chunk)
                    yield chunk
                latency_ms = (time.time() - t0) * 1000.0
                if self.store:
                    self.store.update_outcome(
                        decision.decision_id, latency_ms=latency_ms,
                        chosen_model=spec.id,
                        response_raw="".join(pieces) if pieces else None,
                    )

            return _emit(), decision

        raise ProviderError(
            f"all {len(chain)} candidate model(s) failed to stream; last error: {last_err}"
        )

    # ---- feedback ---------------------------------------------------------

    def feedback(self, decision_id: str, label: int, source: str = "manual") -> bool:
        if not self.store:
            return False
        return self.store.feedback(decision_id, label, source)

    def close(self) -> None:
        if self.store:
            self.store.close()


# ---- helpers --------------------------------------------------------------

def _features_dict(feats) -> Dict[str, Any]:
    return {
        "estimated_tokens": feats.estimated_tokens,
        "has_images": feats.has_images,
        "needs_tools": feats.needs_tools,
        "needs_json": feats.needs_json,
        "num_turns": feats.num_turns,
        "code_ratio": feats.code_ratio,
        "has_cjk": feats.has_cjk,
        "raw_chars": feats.raw_chars,
    }


def _estimate_cost(resp: Dict[str, Any], spec: ModelSpec) -> Optional[float]:
    usage = resp.get("usage") or {}
    total = usage.get("total_tokens")
    if total is None or not spec.cost_per_1k:
        return None
    return (total / 1000.0) * spec.cost_per_1k


def _first_content(resp: Dict[str, Any]) -> Optional[str]:
    try:
        return resp["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _response_text(resp: Dict[str, Any]) -> Optional[str]:
    """The response as trainable text: plain content when the message is text,
    otherwise the whole message as JSON (preserves tool_calls etc.)."""
    content = _first_content(resp)
    if isinstance(content, str) and content:
        return content
    try:
        return json.dumps(resp["choices"][0]["message"])
    except (KeyError, IndexError, TypeError):
        return None


def _delta_content(chunk: Dict[str, Any]) -> Optional[str]:
    try:
        return chunk["choices"][0]["delta"].get("content")
    except (KeyError, IndexError, TypeError):
        return None


def _wants_json(response_format) -> bool:
    if isinstance(response_format, dict):
        return response_format.get("type") in ("json_object", "json_schema")
    return bool(response_format)


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False
