"""Curated known-issue database (design §7.3, §16.7).

The KB is YAML-on-disk so contributors can PR new entries without touching
Python. Every entry has a regex error signature, a typed fix, an
``engines:`` constraint (so the entry only fires on engines whose version
matches), at least one citation, and a human-readable message.

Public surface:

* :func:`load_all` — load every shipped entry, validated against the
  schema.
* :class:`KBEntry`, :class:`FixSpec`, :class:`EngineConstraint` — the
  Pydantic shape (design §16.6).
"""

from __future__ import annotations

from anvil.caas.kb.loader import load_all
from anvil.caas.kb.schema import EngineConstraint, FixSpec, KBEntry, Severity

__all__ = ["KBEntry", "FixSpec", "EngineConstraint", "Severity", "load_all"]
