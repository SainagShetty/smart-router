"""RouterCore — the explicit routing API.

``route()`` makes (and logs) a decision without calling a model. ``complete()``
makes the decision, calls the chosen provider with a fallback chain, records
cost/latency, and captures implicit labels. The drop-in client builds on this.
"""
from __future__ import annotations

import json
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import capabilities, features as features_mod
from .classifiers import build_classifier
from .config import ModelSpec, RouterConfig
from .errors import ConfigNotHotReloadable, NoEligibleModel, ProviderError
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
    # Whether this decision runs in cascade mode (cheapest-first, escalate on a
    # bad answer). complete() validates each answer and steps up the chain.
    cascade: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class _Live:
    """A config and the providers built from it, as one indivisible unit.

    The config can be swapped at runtime (see RouterCore.reload), and that makes
    reading it in two steps a bug:

        route()     reads config     ──┐  two independent attribute reads.
        complete()  reads providers  ──┘  A reload landing between them routes
                                          a request against config A and then
                                          executes it against config B's
                                          providers -- remove a model while a
                                          request is in flight and the lookup
                                          KeyErrors into a 500.

    Binding both from a SINGLE read makes the torn state unrepresentable rather
    than merely unlikely. Every public entry point snapshots `self._live` once
    and passes it down; nothing reads it twice.
    """

    config: RouterConfig
    providers: Dict[str, Any]


class RouterCore:
    def __init__(self, config: RouterConfig):
        # The classifier and the store deliberately live OUTSIDE _Live: they are
        # expensive (a joblib load) and stateful (an open SQLite connection)
        # respectively, and reload keeps both. See reload() for why.
        self.classifier = build_classifier(config.classifier)
        self._live = _Live(config, self._build_providers(config))
        self.store: Optional[TrainingStore] = None
        if config.logging.enabled:
            self.store = TrainingStore(config.logging.db_path)

    @staticmethod
    def _build_providers(config: RouterConfig) -> Dict[str, Any]:
        return {name: build_provider(name, pc)
                for name, pc in config.providers.items()}

    # `config` and `providers` stay readable as attributes: callers and tests
    # reach for them freely (test stubs mutate core.providers in place). They
    # are now views onto the frozen pair, so a reader always sees a consistent
    # snapshot -- but internal code must still bind _live ONCE per request
    # rather than going through these.
    @property
    def config(self) -> RouterConfig:
        return self._live.config

    @property
    def providers(self) -> Dict[str, Any]:
        return self._live.providers

    # ---- reload -----------------------------------------------------------

    def reload(self, new_config: RouterConfig) -> None:
        """Put a new config into force without restarting.

            tiers, models, policy   swap freely
            providers, logging      refused -- restart required
            classifier              refused -- see below

        Only the first group is safe to change under live traffic. Providers own
        httpx clients that in-flight requests are holding; logging owns the open
        SQLite connection the store writes through; the classifier is loaded once
        at construction. Swapping any of them here would strand something, and
        silently: the request already in flight would never know. Refusing is
        what makes the unsafe case impossible rather than merely unlikely.

        Because providers provably cannot differ by the time the swap happens,
        the EXISTING providers dict is carried over rather than rebuilt. That
        keeps object identity stable (test stubs patch these in place), avoids
        constructing httpx clients nobody asked for, and removes the only step
        of the swap that could have raised.

        The assignment is a single statement over a frozen pair, so a request
        either sees the whole old config or the whole new one -- never a mix.
        In-flight requests finish against the snapshot they already took.
        """
        self.check_reloadable(new_config)
        live = self._live
        self._live = _Live(new_config, live.providers)

    def check_reloadable(self, new_config: RouterConfig) -> None:
        """Raise if `new_config` could not be hot-applied. Changes nothing.

        Separate from reload() on purpose. A caller wanting to know "would this
        work" must not find out by DOING it: apply() renders to disk before
        reloading, and an earlier draft used a reload as its dry run -- which
        succeeded, swapped the live config, and then left the router serving an
        un-rendered, uncommitted candidate when the render failed. Asking is not
        the same as doing.
        """
        live_config = self._live.config
        for section in ("providers", "logging", "classifier"):
            if getattr(new_config, section) != getattr(live_config, section):
                raise ConfigNotHotReloadable(
                    f"`{section}` changed; that cannot be applied to a running "
                    "process. Save the revision and restart smart-router."
                )

    # ---- replay -----------------------------------------------------------

    def replay(self, candidate: RouterConfig, limit: int = 1000) -> Dict[str, Any]:
        """Score a candidate config against traffic that actually happened.

        Answers "what would this change have done" before it is applied, which
        turns adjusting a threshold from a guess into an observation.

            stored score ──▶ gate(candidate.models, stored features)
                        └──▶ decide(candidate, ...)  ──▶ would-be tier
                                                            │
                             recorded tier ◀────────────────┘ compare

        No provider is called and no embedding is read. The stored score is
        already the classifier's verdict on that prompt, so a *config* replay
        only needs to re-ask the threshold question -- re-scoring would only be
        necessary to evaluate a different classifier.

        Cost and latency deltas are ESTIMATES and labelled as such: we know what
        each request actually cost in the tier it went to, so a request moving
        local -> cheap is priced at the median observed cost of the cheap tier.
        That is an average standing in for a specific prompt. Tier movements, by
        contrast, are exact.

        The returned `coverage` is not optional decoration. A preview reporting
        "0 would change" without saying how many it looked at is
        indistinguishable from "this change is safe", and most of this log
        cannot be replayed at all -- see TrainingStore.replay_corpus.
        """
        if self.store is None:
            return {"coverage": {"replayed": 0, "eligible": 0, "total": 0,
                                 "excluded_override": 0, "excluded_no_receipt": 0,
                                 "reason": "logging is disabled; nothing to replay"},
                    "moves": {}, "unchanged": 0,
                    "before": {}, "after": {}, "estimates": None}

        corpus = self.store.replay_corpus(limit=limit)
        rows = corpus["rows"]

        # Median observed cost/latency per tier, from the SAME corpus -- used to
        # price a move into a tier this prompt never visited.
        # Cost and latency are gathered INDEPENDENTLY, and a missing cost counts
        # as zero rather than as unknown. The local tier records cost NULL
        # because it is free, not because nothing was measured -- and requiring
        # both meant every row that moved OFF local was silently dropped from
        # the estimate, which is precisely the set of rows the estimate exists
        # to describe. Latency stays strictly observed: a missing latency really
        # is unknown.
        observed: Dict[str, Dict[str, List[float]]] = {}
        for r in rows:
            t = observed.setdefault(r["chosen_tier"], {"cost": [], "latency": []})
            t["cost"].append(r["cost"] if r["cost"] is not None else 0.0)
            if r["latency_ms"] is not None:
                t["latency"].append(r["latency_ms"])

        moves: Dict[str, int] = {}
        before: Dict[str, int] = {}
        after: Dict[str, int] = {}
        unchanged = 0
        cost_before = cost_after = 0.0
        lat_before = lat_after = 0.0
        priced = timed = 0

        for r in rows:
            feats = _features_from_stored(r["features"])
            gate = capabilities.gate(candidate.models, feats)
            if not gate.candidates:
                # The candidate config cannot serve this request at all. That is
                # a real, reportable outcome -- not a silent skip.
                moves[f"{r['chosen_tier']} -> (no eligible model)"] = \
                    moves.get(f"{r['chosen_tier']} -> (no eligible model)", 0) + 1
                before[r["chosen_tier"]] = before.get(r["chosen_tier"], 0) + 1
                continue
            result = decide(candidate, gate.candidates, r["score"], Overrides(),
                            rejected=gate.rejected)
            old_tier, new_tier = r["chosen_tier"], result.tier
            before[old_tier] = before.get(old_tier, 0) + 1
            after[new_tier] = after.get(new_tier, 0) + 1
            if old_tier == new_tier:
                unchanged += 1
            else:
                key = f"{old_tier} -> {new_tier}"
                moves[key] = moves.get(key, 0) + 1

            oc, nc = _median(observed, old_tier, "cost"), _median(observed, new_tier, "cost")
            if oc is not None and nc is not None:
                cost_before += oc; cost_after += nc
                priced += 1
            ol, nl = _median(observed, old_tier, "latency"), _median(observed, new_tier, "latency")
            if ol is not None and nl is not None:
                lat_before += ol; lat_after += nl
                timed += 1

        estimates = None
        if priced or timed:
            estimates = {
                "basis": "median observed per tier, from this corpus; an average "
                         "standing in for a specific prompt",
            }
            if priced:
                estimates["priced_rows"] = priced
                estimates["cost_delta"] = round(cost_after - cost_before, 6)
            if timed:
                estimates["timed_rows"] = timed
                estimates["latency_delta_ms_per_request"] = round(
                    (lat_after - lat_before) / timed, 1)

        return {
            "coverage": {
                "replayed": len(rows),
                "eligible": corpus["eligible"],
                "total": corpus["total"],
                "excluded_override": corpus["excluded_override"],
                "excluded_no_receipt": corpus["excluded_no_receipt"],
            },
            "moves": moves,
            "unchanged": unchanged,
            "before": before,
            "after": after,
            "estimates": estimates,
        }

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
        cascade: Optional[bool] = None,
        log_raw: Optional[bool] = None,
        source: Optional[str] = None,
        sensitive: bool = False,
        _live: Optional[_Live] = None,
    ) -> Decision:
        # Callers that also need providers (complete, stream) snapshot _live
        # themselves and pass it in, so the whole request is served by one
        # consistent pair. A bare route() call takes its own snapshot.
        live = _live if _live is not None else self._live
        feats = features_mod.extract(
            messages, tools=tools, response_format=response_format
        )
        gate = capabilities.gate(live.config.models, feats)

        score, embedding = self._score(feats)

        cascade = live.config.policy.cascade if cascade is None else cascade
        result = decide(
            live.config,
            gate.candidates,
            score,
            Overrides(force_tier=force_tier, cheap_only=cheap_only,
                      local_only=local_only, sensitive=sensitive,
                      cascade=cascade),
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
            log_raw=live.config.logging.log_raw if log_raw is None else log_raw,
            cascade=cascade,
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
        cascade: Optional[bool] = None,
        log_raw: Optional[bool] = None,
        source: Optional[str] = None,
        sensitive: bool = False,
        **params,
    ) -> Tuple[Dict[str, Any], Decision]:
        live = self._live
        decision = self.route(
            messages, tools=tools, response_format=response_format,
            force_tier=force_tier, cheap_only=cheap_only, local_only=local_only,
            cascade=cascade, log_raw=log_raw, source=source, sensitive=sensitive,
            _live=live,
        )
        chain = [decision.model_spec] + list(decision.fallback_specs)
        if tools is not None:
            params["tools"] = tools
        if response_format is not None:
            params["response_format"] = response_format

        last_err: Optional[Exception] = None
        for i, spec in enumerate(chain):
            provider = live.providers[spec.provider]
            call_params = dict(spec.params)
            call_params.update(params)
            t0 = time.time()
            try:
                resp = provider.complete(spec.id, list(messages), **call_params)
            except ProviderError as exc:
                last_err = exc
                continue
            latency_ms = (time.time() - t0) * 1000.0

            # Cascade: if a cheaper answer is inadequate and a higher tier is
            # still available, escalate rather than return it. The last spec in
            # the chain is always returned (best effort). For sensitive requests
            # the chain contains only local models, so escalation never egresses.
            if (decision.cascade and i < len(chain) - 1
                    and not _answer_ok(resp, response_format)):
                if self.store:
                    self.store.set_label_if_unset(
                        decision.decision_id, 1, "implicit:cascade_escalate"
                    )
                continue

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
            chosen_model=spec.id, chosen_tier=spec.tier,
            response_raw=response_raw,
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
        live = self._live
        decision = self.route(
            messages, tools=tools, response_format=response_format,
            force_tier=force_tier, cheap_only=cheap_only, local_only=local_only,
            log_raw=log_raw, source=source, sensitive=sensitive, _live=live,
        )
        chain = [decision.model_spec] + list(decision.fallback_specs)
        if tools is not None:
            params["tools"] = tools
        if response_format is not None:
            params["response_format"] = response_format

        t0 = time.time()
        last_err: Optional[Exception] = None
        for spec in chain:
            provider = live.providers[spec.provider]
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
                usage: Dict[str, Any] = {}

                def _collect(chunk):
                    # Providers send usage totals in a trailing chunk (requested
                    # via stream_options); it may arrive with an empty choices
                    # list, so collect it independently of the content deltas.
                    chunk_usage = chunk.get("usage")
                    if chunk_usage:
                        usage.update(chunk_usage)
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
                        decision.decision_id,
                        cost=_estimate_cost({"usage": usage}, spec) if usage else None,
                        latency_ms=latency_ms,
                        chosen_model=spec.id,
                        # Mirrors _finalize: a stream that fell back to another
                        # tier must not leave the tier pointing at the original.
                        chosen_tier=spec.tier,
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

def _median(observed, tier, field):
    """Median of one observed field in a tier, or None if nothing was recorded."""
    t = observed.get(tier)
    if not t or not t[field]:
        return None
    return statistics.median(t[field])


def _features_from_stored(d: Dict[str, Any]) -> "features_mod.RequestFeatures":
    """Rebuild just enough of RequestFeatures for the capability gate.

    The gate reads four fields: has_images, needs_tools, needs_json and
    estimated_tokens. `text` is not among them, which is fortunate -- a raw
    prompt is only persisted when log_raw is on, so reconstructing it is not
    always possible and is never necessary here.
    """
    return features_mod.RequestFeatures(
        text="",
        estimated_tokens=int(d.get("estimated_tokens") or 0),
        has_images=bool(d.get("has_images")),
        needs_tools=bool(d.get("needs_tools")),
        needs_json=bool(d.get("needs_json")),
        num_turns=int(d.get("num_turns") or 0),
        code_ratio=float(d.get("code_ratio") or 0.0),
        has_cjk=bool(d.get("has_cjk")),
        raw_chars=int(d.get("raw_chars") or 0),
    )


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
        # Request shape (see RequestFeatures). Goes into the existing `features`
        # JSON column, so no schema migration — old rows simply lack the keys.
        "n_tool_msgs": feats.n_tool_msgs,
        "n_assistant_msgs": feats.n_assistant_msgs,
        "head_chars": feats.head_chars,
        "tail_chars": feats.tail_chars,
    }


def _estimate_cost(resp: Dict[str, Any], spec: ModelSpec) -> Optional[float]:
    usage = resp.get("usage") or {}
    # Prefer the prompt/completion split when the model has per-direction rates
    # and the provider echoed the split (output tokens cost ~4-5x input).
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if (spec.cost_per_1k_in is not None and spec.cost_per_1k_out is not None
            and prompt is not None and completion is not None):
        return ((prompt / 1000.0) * spec.cost_per_1k_in
                + (completion / 1000.0) * spec.cost_per_1k_out)
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


_REFUSAL_MARKERS = (
    "i cannot", "i can't", "i'm unable", "i am unable", "i don't know",
    "as an ai", "i'm sorry", "i am sorry",
)


def _answer_ok(resp: Dict[str, Any], response_format) -> bool:
    """Cheap deterministic quality check for cascade escalation. Returns False
    (escalate) on an empty/too-short answer, an obvious refusal, or — when JSON
    was requested — output that isn't valid JSON. No model call."""
    content = _first_content(resp)
    if content is None or not isinstance(content, str):
        # A non-text message (e.g. tool_calls) is a legitimate answer.
        return content is not None or _has_tool_calls(resp)
    text = content.strip()
    if len(text) < 2:
        return False
    if response_format is not None and _wants_json(response_format):
        return _is_json(text)
    lowered = text.lower()
    if any(lowered.startswith(m) for m in _REFUSAL_MARKERS):
        return False
    return True


def _has_tool_calls(resp: Dict[str, Any]) -> bool:
    try:
        return bool(resp["choices"][0]["message"].get("tool_calls"))
    except (KeyError, IndexError, TypeError):
        return False


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False
