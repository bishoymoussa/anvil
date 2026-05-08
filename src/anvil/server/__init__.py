"""OpenAI-compatible server (design §10.2).

Full implementation lands in M6 (design §16.10). v0 currently exports a
stub :func:`serve` that raises ``NotImplementedError`` so the public symbol
is importable.
"""

from __future__ import annotations

from typing import Any


def serve(*args: Any, **kwargs: Any) -> None:
    """Run an OpenAI-compatible HTTP server. (M6 work.)"""
    raise NotImplementedError(
        "anvil.serve is M6 work (design §16.10). It will provide an "
        "OpenAI-compatible API with constrained-decoding tool calls."
    )


__all__ = ["serve"]
