"""Primitives: the typed, content-hashed objects users touch directly.

This package is a *leaf* in Anvil's import graph (enforced by import-linter):
nothing in ``anvil.primitives`` imports from ``anvil.engine``, ``anvil.models``,
``anvil.tasks``, ``anvil.manifest``, ``anvil.caas``, ``anvil.cli`` or
``anvil.server``. That decoupling is what lets a ``ChatTemplate`` or
``Sampler`` have a stable, machine-checkable hash independent of what backend
is rendering or sampling with it.
"""

from __future__ import annotations
