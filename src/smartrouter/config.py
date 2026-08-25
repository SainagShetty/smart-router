"""Configuration models for smartrouter.

Everything the router needs is declared here as Pydantic models so config is
validated once, up front. Load from a dict or a YAML file:

    config = RouterConfig.from_yaml("router.yaml")
    config = RouterConfig.from_dict({...})

The provider is configurable (OpenRouter by default, Ollama/OpenAI/Anthropic
selectable) so the library stays usable outside one ecosystem.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

ProviderType = Literal["openrouter", "ollama", "openai", "anthropic"]
ClassifierKind = Literal["embedding", "heuristic", "routellm"]
FallbackDirection = Literal["down", "up", "none"]


class Capabilities(BaseModel):
    """Hard requirements a model can satisfy. Used by the capability gate."""

    model_config = {"populate_by_name": True}

    vision: bool = False
    tools: bool = False
    # native structured/JSON output mode; accept "json" in config, expose as
    # json_mode in code (avoid shadowing BaseModel.json()).
    json_mode: bool = Field(default=False, alias="json")


class ModelSpec(BaseModel):
    """A single routable model."""

    id: str  # provider-specific model id, e.g. "meta-llama/llama-3.1-8b-instruct"
    provider: str  # key into RouterConfig.providers
    tier: str  # key into RouterConfig.tiers
    cost_per_1k: float = 0.0  # USD per 1k tokens (blended); used for tie-breaks/budget
    # Optional per-direction rates. When both are set, cost is estimated from the
    # prompt/completion split the provider echoes (output is typically ~4-5x
    # input); otherwise it falls back to the blended cost_per_1k over total tokens.
    cost_per_1k_in: Optional[float] = None
    cost_per_1k_out: Optional[float] = None
    context_window: int = 8192
    capabilities: Capabilities = Field(default_factory=Capabilities)
    # Provider call defaults (temperature, etc.) merged into each request.
    params: Dict[str, object] = Field(default_factory=dict)


class TierConfig(BaseModel):
    """A quality tier. Tiers are ordered cheapest -> most capable.

    ``min_score`` is the inclusive lower bound of the difficulty band this tier
    serves. The policy picks the most capable tier whose ``min_score <= score``.
    """

    name: str
    min_score: float = 0.0


class ProviderConfig(BaseModel):
    type: ProviderType
    base_url: Optional[str] = None
    # Provide the key directly, or name an env var to read it from.
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout: float = 60.0
    # Whether this provider runs locally. Defaults True for ollama; set
    # explicitly for self-hosted OpenAI-compatible servers (vLLM, llama.cpp,
    # LM Studio). Drives the per-request ``local_only`` override.
    local: Optional[bool] = None

    def resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    def is_local(self) -> bool:
        if self.local is not None:
            return self.local
        return self.type == "ollama"


class ClassifierConfig(BaseModel):
    kind: ClassifierKind = "embedding"
    # Path to a pre-fit head (joblib). None -> use the bundled default.
    model_path: Optional[str] = None
    # Embedding backend for kind="embedding": "hashing" (no torch) or
    # "sentence-transformer" (requires the local-embed extra).
    embedding_backend: Literal["hashing", "sentence-transformer"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # RouteLLM router name for kind="routellm".
    routellm_router: str = "mf"
    params: Dict[str, object] = Field(default_factory=dict)


class PolicyConfig(BaseModel):
    # On provider error: try the next-lower tier, next-higher tier, or give up.
    fallback: FallbackDirection = "down"
    # Allow per-request force_tier / cheap_only overrides.
    allow_overrides: bool = True
    # Default to cascade execution (cheapest-first, escalate on a bad answer)
    # instead of trusting the difficulty score up front. Per-request cascade=
    # overrides this.
    cascade: bool = False
    # Tier used when the classifier errors out or no candidate matches a band.
    default_tier: Optional[str] = None
    # Optional hard cap on cost_per_1k of the chosen model.
    max_cost_per_1k: Optional[float] = None


class LoggingConfig(BaseModel):
    enabled: bool = True
    db_path: str = "smartrouter.db"
    # Store the raw prompt text (off by default: never persist secrets silently).
    log_raw: bool = False
    # Optional tracing backend ("langfuse" | "langsmith"); tracing only.
    tracing: Optional[str] = None


def _default_providers() -> Dict[str, ProviderConfig]:
    return {
        "openrouter": ProviderConfig(
            type="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key_env="OPENROUTER_API_KEY",
        ),
        "ollama": ProviderConfig(
            type="ollama",
            base_url="http://localhost:11434/v1",
        ),
    }


class RouterConfig(BaseModel):
    models: List[ModelSpec]
    tiers: List[TierConfig]
    providers: Dict[str, ProviderConfig] = Field(default_factory=_default_providers)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Set only in a RENDERED config, by render_revision(). A config being
    # composed or stored has no revision id yet -- the id exists once the row
    # does. Its presence is what lets a running process name what it is serving
    # without consulting the database, which is what keeps the database off the
    # boot path entirely.
    revision: Optional[int] = None

    # ---- validation -------------------------------------------------------

    @model_validator(mode="after")
    def _check_references(self) -> "RouterConfig":
        if not self.models:
            raise ValueError("RouterConfig.models must not be empty")
        if not self.tiers:
            raise ValueError("RouterConfig.tiers must not be empty")

        tier_names = {t.name for t in self.tiers}
        if len(tier_names) != len(self.tiers):
            raise ValueError("tier names must be unique")

        for m in self.models:
            if m.tier not in tier_names:
                raise ValueError(
                    f"model {m.id!r} references unknown tier {m.tier!r}"
                )
            if m.provider not in self.providers:
                raise ValueError(
                    f"model {m.id!r} references unknown provider {m.provider!r}"
                )

        if self.policy.default_tier and self.policy.default_tier not in tier_names:
            raise ValueError(
                f"policy.default_tier {self.policy.default_tier!r} is not a known tier"
            )

        # Keep tiers ordered cheapest -> most capable by their score floor.
        self.tiers.sort(key=lambda t: t.min_score)
        return self

    # ---- ordering helpers -------------------------------------------------

    def tier_order(self) -> List[str]:
        """Tier names from cheapest to most capable."""
        return [t.name for t in self.tiers]

    def tier_index(self, name: str) -> int:
        return self.tier_order().index(name)

    def models_for_tier(self, tier: str) -> List[ModelSpec]:
        return [m for m in self.models if m.tier == tier]

    # ---- loaders ----------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict) -> "RouterConfig":
        return cls.model_validate(data)

    @classmethod
    def from_yaml(cls, path: str) -> "RouterConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))


def render_revision(yaml_text: str, revision_id: int) -> str:
    """Stamp a revision id into a config, ready to be written to disk.

    Textual on purpose. The obvious implementation -- load with ruamel, set the
    key, dump -- was written first and measured against the real
    examples/router.yaml: it preserved all 21 comment lines and parsed
    identically, but rewrote 10 lines because ruamel does not preserve spacing
    inside flow mappings (`{ name: local }` became `{name: local}`). Harmless,
    and still a diff nobody asked for on every save.

    The stamp is a single top-level key and YAML does not care where it sits, so
    appending it leaves every other byte exactly as the author wrote it. Any
    existing top-level `revision:` line is dropped first so re-stamping replaces
    rather than duplicates; the column-0 anchor means an indented key of the
    same name inside some nested mapping is left alone.

    It is a real validated key rather than a comment, which is what lets a
    running process name the revision it is serving by reading the file it was
    already going to read -- and so what keeps the database off the boot path.
    """
    kept = [ln for ln in yaml_text.splitlines()
            if not re.match(r"^revision:\s", ln)]
    body = "\n".join(kept).rstrip("\n")
    return f"{body}\n\nrevision: {revision_id}\n"


def write_rendered(path: str, text: str) -> None:
    """Write a rendered config atomically.

    Same file, one syscall to swap it: a partial write would leave the router
    unable to start on the next boot, and a boot failure on the fleet's gateway
    takes ten services with it. Temp file in the same directory so os.replace
    stays on one filesystem, then fsync so the rename cannot be reordered ahead
    of the contents.
    """
    import os
    import tempfile

    target = os.path.abspath(path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(target), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
