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
