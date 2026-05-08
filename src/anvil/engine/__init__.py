"""Engine layer (design §3.1, §16.1).

Only the public protocol and factory live here at the package level. The
backends themselves (``_hf``, ``_vllm``, ``_wrappers``) are private — users
go through :func:`anvil.models.load`, which picks a backend.
"""

from __future__ import annotations

from anvil.engine.factory import build_engine
from anvil.engine.public import Engine

__all__ = ["Engine", "build_engine"]
