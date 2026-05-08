"""``anvil.research`` — example research-grade extensions.

For v0 this ships a single :class:`DoLa` placeholder so the public API
surface (``from anvil.research import DoLa``) is stable. Real DoLa lands in
v0.5; v0 raises ``NotImplementedError`` if the processor is invoked.
"""

from __future__ import annotations

from anvil.research.dola import DoLa

__all__ = ["DoLa"]
