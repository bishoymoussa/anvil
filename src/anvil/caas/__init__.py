"""CaaS preflight self-healing agent (design §7).

Public surface (M3):

* :func:`load_kb` — load the shipped 15-entry KB.
* :func:`engage` — match an error against the KB, propose a typed
  :class:`Action`, and (per CaaS mode) apply it.
* :class:`Context` — runtime data the rule engine inspects.
* :class:`AuditLog` — append-only record that flows into the manifest's
  ``caas_log`` field.
* :func:`run_sentinel` — run the quality sentinel against any engine.

The v1 LLM tier is documented but not implemented in v0
(:mod:`anvil.caas.llm_tier` raises ``NotImplementedError``; design §7.4).
"""

from __future__ import annotations

from anvil.caas.actions import Action, ActionType, ApplyResult, apply_action
from anvil.caas.audit import AuditLog
from anvil.caas.kb.loader import load_all as load_kb
from anvil.caas.kb.schema import KBEntry, Severity
from anvil.caas.preflight import CaaSMode, PreflightOutcome, engage
from anvil.caas.rule_engine import Context, Match, match
from anvil.caas.sentinel import (
    DEFAULT_SENTINEL_EXPECTED,
    DEFAULT_SENTINEL_PROMPT,
    SentinelResult,
    run_sentinel,
)

__all__ = [
    "Action",
    "ActionType",
    "ApplyResult",
    "apply_action",
    "AuditLog",
    "CaaSMode",
    "Context",
    "KBEntry",
    "Match",
    "PreflightOutcome",
    "Severity",
    "SentinelResult",
    "DEFAULT_SENTINEL_EXPECTED",
    "DEFAULT_SENTINEL_PROMPT",
    "engage",
    "load_kb",
    "match",
    "run_sentinel",
]
