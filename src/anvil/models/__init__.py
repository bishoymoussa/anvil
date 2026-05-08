"""Models layer (design §5).

Two paths into the same engine: the *fast path* (~30 hand-tuned
architectures) and the *slow path* (anything that loads in
``transformers.AutoModelForCausalLM``). M0 only ships the slow path; fast
paths land per-architecture in M1 and M6.
"""

from __future__ import annotations

from anvil.models.registry import LoadedModel, load, load_custom

__all__ = ["load", "load_custom", "LoadedModel"]
