"""Tests for the AnvilError hierarchy (design §16.9)."""

from __future__ import annotations

from anvil.exceptions import (
    AnvilError,
    CaaSCannotFix,
    CaaSError,
    ConfigError,
    EngineError,
    ManifestError,
    ModelLoadError,
    PluginError,
    TaskError,
    TokenizationError,
)


def test_every_error_inherits_from_anvil_error() -> None:
    for cls in (
        ConfigError,
        EngineError,
        ModelLoadError,
        TokenizationError,
        TaskError,
        ManifestError,
        CaaSError,
        CaaSCannotFix,
        PluginError,
    ):
        assert issubclass(cls, AnvilError)


def test_caas_cannot_fix_is_caas_error() -> None:
    assert issubclass(CaaSCannotFix, CaaSError)


def test_error_code_in_message() -> None:
    e = ConfigError("bad config")
    assert "ANVIL-E0001" in str(e)
    assert "bad config" in str(e)


def test_error_codes_are_unique() -> None:
    classes = (
        AnvilError,
        ConfigError,
        EngineError,
        ModelLoadError,
        TokenizationError,
        TaskError,
        ManifestError,
        CaaSError,
        CaaSCannotFix,
        PluginError,
    )
    codes = [c.error_code for c in classes]
    # CaaSCannotFix has its own code; the rest are unique too.
    assert len(set(codes)) == len(codes)
