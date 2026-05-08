"""Fast-path model implementations (design §5.2).

Each architecture lives in its own ``<family>.py`` and registers via
:func:`anvil.models.registry.register_model_impl`. Importing this package
triggers all registrations as a side effect.

* M4 ships the Qwen-VL family fast-path *marker* (kernel-level fast path
  with custom attention is M6 work).
"""

from __future__ import annotations

# Importing the module registers via @register_model_impl. F401 silenced.
from anvil.models._fast import qwen_vl as _qwen_vl  # noqa: F401
