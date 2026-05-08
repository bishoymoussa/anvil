"""CaaS LLM tier — v1 work, kept as a stub in v0 (design §7.4, §16.1).

The LLM tier is gated behind multiple safety constraints (allow-list of
flags, action-space FSM, audit log) — it is *not* part of v0. This stub
exists so that imports referring to the tier raise a helpful error rather
than ``ImportError``.
"""

from __future__ import annotations


def propose_fix(*args: object, **kwargs: object) -> None:
    """v1 will dispatch to a small coder LLM here. v0 raises explicitly."""
    raise NotImplementedError(
        "CaaS LLM tier lands in v1 (design §7.4, §11.3). v0 ships rule engine "
        "and KB only — see anvil.caas.rule_engine and anvil.caas.kb (M3)."
    )


__all__ = ["propose_fix"]
