"""Versioned plugin protocol (design §5.3, §16.9).

Plugins discover via Python entry points namespaced as
``anvil.plugins.v1.<kind>`` (kinds: ``models``, ``tasks``, ``metrics``,
``engines``, ``kb_entries``). v1 has a one-major-release deprecation cycle
when v2 ships.
"""

from __future__ import annotations

from anvil.plugins.v1 import register_model_impl, register_request_type

__all__ = ["register_model_impl", "register_request_type"]
