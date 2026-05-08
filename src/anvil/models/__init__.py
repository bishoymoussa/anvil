"""Models layer (design §5).

Two paths into the same engine: the *fast path* (the curated set in
:mod:`anvil.models._fast`) and the *slow path* (anything that loads in
``transformers.AutoModelForCausalLM`` / ``AutoModelForVision2Seq``).

Importing this package triggers fast-path registration as a side effect
— the registry is process-local so plugins added at runtime via
``anvil.plugins.v1`` take precedence over the shipped markers.
"""

from __future__ import annotations

# Importing _fast triggers @register_model_impl on every shipped family.
# F401 silenced because the import is purely for the side-effect.
from anvil.models import _fast as _fast  # noqa: F401
from anvil.models.registry import LoadedModel, load, load_custom

__all__ = ["load", "load_custom", "LoadedModel"]
