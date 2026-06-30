"""Turn a difficulty score (plus gated candidates and any per-request overrides)
into a concrete model choice and an ordered fallback chain.

The score only selects a *tier*; the chosen model is the cheapest candidate in
that tier. Overrides (force_tier / cheap_only / local_only) restrict the
candidate set but never bypass the capability gate, so they can't pick a model
that cannot serve the request.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import ModelSpec, RouterConfig
from .errors import NoEligibleModel


@dataclass
class Overrides:
    force_tier: Optional[str] = None
    cheap_only: bool = False
    local_only: bool = False


@dataclass
class PolicyResult:
    model: ModelSpec
    tier: str
    reason: str
    fallback: List[ModelSpec] = field(default_factory=list)


def _by_tier(config: RouterConfig, models: List[ModelSpec]) -> Dict[str, List[ModelSpec]]:
    groups: Dict[str, List[ModelSpec]] = {t: [] for t in config.tier_order()}
    for m in models:
        groups[m.tier].append(m)
    # Cheapest model first within a tier (tie-break).
    for tier in groups:
        groups[tier].sort(key=lambda m: m.cost_per_1k)
    return groups


def _target_tier_index(config: RouterConfig, score: float) -> int:
    """Most capable tier whose min_score <= score."""
    idx = 0
    for i, tier in enumerate(config.tiers):
        if score >= tier.min_score:
            idx = i
    return idx


def decide(
    config: RouterConfig,
    candidates: List[ModelSpec],
    score: float,
    overrides: Optional[Overrides] = None,
    rejected: Optional[List[str]] = None,
) -> PolicyResult:
    overrides = overrides or Overrides()
    rejected = rejected or []

    if not candidates:
        raise NoEligibleModel(
            "no model satisfies the request's capability requirements", rejected
        )

    pool = list(candidates)
    notes: List[str] = []

    # ---- apply overrides (respecting policy.allow_overrides) --------------
    allow = config.policy.allow_overrides
    if overrides.local_only and allow:
        local_providers = {
            name for name, p in config.providers.items() if p.is_local()
        }
        pool = [m for m in pool if m.provider in local_providers]
        notes.append("local_only")
        if not pool:
            raise NoEligibleModel(
                "local_only: no local model can serve this request", rejected
            )

    if overrides.force_tier and allow:
        pool = [m for m in pool if m.tier == overrides.force_tier]
        notes.append(f"force_tier={overrides.force_tier}")
        if not pool:
            raise NoEligibleModel(
                f"force_tier={overrides.force_tier!r}: no eligible model in that tier",
                rejected,
            )

    if config.policy.max_cost_per_1k is not None:
        capped = [m for m in pool if m.cost_per_1k <= config.policy.max_cost_per_1k]
        if capped:
            pool = capped
            notes.append(f"max_cost_per_1k={config.policy.max_cost_per_1k}")

    groups = _by_tier(config, pool)
    order = config.tier_order()
    available = [i for i, t in enumerate(order) if groups[t]]
    if not available:  # pragma: no cover - pool non-empty guarantees a tier
        raise NoEligibleModel("no eligible model after overrides", rejected)

    # ---- pick the target tier ---------------------------------------------
    if overrides.force_tier and allow:
        target = order.index(overrides.force_tier)
        notes.append(f"score={score:.3f} (ignored: forced)")
    else:
        target = _target_tier_index(config, score)
        notes.append(f"score={score:.3f}")

        # cheap_only caps the ceiling below the most-capable tier.
        if overrides.cheap_only and allow and len(order) > 1:
            target = min(target, len(order) - 2)
            notes.append("cheap_only")

        # Snap to the nearest available tier: prefer the highest available tier
        # at or below target; otherwise the cheapest available above it.
        at_or_below = [i for i in available if i <= target]
        target = max(at_or_below) if at_or_below else min(available)

    chosen_tier = order[target]
    chosen = groups[chosen_tier][0]

    # ---- fallback chain ----------------------------------------------------
    fallback: List[ModelSpec] = []
    direction = config.policy.fallback
    if direction == "down":
        seq = range(target - 1, -1, -1)
    elif direction == "up":
        seq = range(target + 1, len(order))
    else:
        seq = range(0)
    for i in seq:
        fallback.extend(groups[order[i]])

    reason = f"tier={chosen_tier}; " + ", ".join(notes)
    return PolicyResult(model=chosen, tier=chosen_tier, reason=reason, fallback=fallback)
