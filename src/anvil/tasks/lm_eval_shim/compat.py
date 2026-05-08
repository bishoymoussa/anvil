"""Helpers that wire the lm-eval shim into the Anvil task registry.

Exposed via ``anvil.tasks.lm_eval_shim.register_lm_eval_task`` for callers
that already have a YAML-shaped dict in hand (e.g. plugin authors who
embed task specs in Python rather than on disk).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anvil.tasks.lm_eval_shim.compiler import compile_yaml, compile_yaml_dict

if TYPE_CHECKING:
    from pathlib import Path

    from anvil.tasks.base import Task


def register_lm_eval_task(spec: str | Path | dict[str, Any]) -> type[Task]:
    """Compile + register an lm-eval task spec. Returns the compiled class."""
    if isinstance(spec, dict):
        return compile_yaml_dict(spec)
    return compile_yaml(spec)


__all__ = ["register_lm_eval_task"]
