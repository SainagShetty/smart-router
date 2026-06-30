"""smartrouter — capability-aware, classifier-driven model routing."""
from __future__ import annotations

from .client import RouterClient
from .config import (
    Capabilities,
    ClassifierConfig,
    LoggingConfig,
    ModelSpec,
    PolicyConfig,
    ProviderConfig,
    RouterConfig,
    TierConfig,
)
from .core import Decision, RouterCore
from .errors import (
    ConfigError,
    NoEligibleModel,
    ProviderError,
    SmartRouterError,
)

__all__ = [
    "RouterConfig",
    "ModelSpec",
    "TierConfig",
    "Capabilities",
    "ProviderConfig",
    "ClassifierConfig",
    "PolicyConfig",
    "LoggingConfig",
    "RouterCore",
    "RouterClient",
    "Decision",
    "SmartRouterError",
    "NoEligibleModel",
    "ProviderError",
    "ConfigError",
]

__version__ = "0.1.0"
