"""smartrouter exception types."""
from __future__ import annotations


class SmartRouterError(Exception):
    """Base class for all smartrouter errors."""


class NoEligibleModel(SmartRouterError):
    """No model can serve the request under the gate + active overrides.

    Carries the human-readable rejection reasons for observability.
    """

    def __init__(self, message: str, rejected=None):
        super().__init__(message)
        self.rejected = rejected or []


class ProviderError(SmartRouterError):
    """A provider call failed (network, auth, HTTP error)."""


class ConfigError(SmartRouterError):
    """Invalid configuration discovered at runtime."""


class ConfigNotHotReloadable(ConfigError):
    """A config change is valid, but cannot be applied to a running process.

    Providers hold live httpx clients that in-flight requests are using, and
    logging holds an open SQLite connection the store writes through. Swapping
    either under traffic would strand something. Rather than half-apply such a
    change -- or worse, appear to apply it -- reload refuses and says which
    section needs a restart. The revision is still saved; only the way it is
    put into force differs.
    """
