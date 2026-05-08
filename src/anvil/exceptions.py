"""Anvil error hierarchy (design manuscript §16.9).

Every exception raised by Anvil is a subclass of :class:`AnvilError` and
carries a stable ``error_code`` (e.g. ``"ANVIL-E0042"``) so the CaaS KB and
documentation can reference it without depending on the message text.
"""

from __future__ import annotations


class AnvilError(Exception):
    """Base class for every error Anvil raises.

    Subclasses set ``error_code`` to a stable identifier of the form
    ``ANVIL-Ennnn``. The number is allocated once and never reused.
    """

    error_code: str = "ANVIL-E0000"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code

    def __str__(self) -> str:
        base = super().__str__()
        return f"[{self.error_code}] {base}"


class ConfigError(AnvilError):
    """User-supplied configuration is invalid or self-contradictory."""

    error_code = "ANVIL-E0001"


class EngineError(AnvilError):
    """The inference engine failed in a way Anvil cannot transparently handle."""

    error_code = "ANVIL-E0010"


class ModelLoadError(AnvilError):
    """Loading model weights, config, or tokenizer failed."""

    error_code = "ANVIL-E0011"


class TokenizationError(AnvilError):
    """Tokenizer behaved in a way Anvil's invariants forbid (e.g. double BOS)."""

    error_code = "ANVIL-E0020"


class TaskError(AnvilError):
    """A Task raised, returned an unexpected type, or violated its contract."""

    error_code = "ANVIL-E0030"


class ManifestError(AnvilError):
    """Manifest construction, signing, verification, or replay failed."""

    error_code = "ANVIL-E0040"


class CaaSError(AnvilError):
    """CaaS preflight encountered an internal error (not a fix candidate)."""

    error_code = "ANVIL-E0050"


class CaaSCannotFix(CaaSError):
    """CaaS exhausted rule engine + KB (+ LLM tier when present) and cannot fix."""

    error_code = "ANVIL-E0051"


class PluginError(AnvilError):
    """A third-party plugin failed to register or violated the protocol."""

    error_code = "ANVIL-E0060"


__all__ = [
    "AnvilError",
    "ConfigError",
    "EngineError",
    "ModelLoadError",
    "TokenizationError",
    "TaskError",
    "ManifestError",
    "CaaSError",
    "CaaSCannotFix",
    "PluginError",
]
