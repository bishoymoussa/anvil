"""Single source of truth for ``anvil.__version__``.

Kept as a tiny standalone module so ``anvil/__init__.py`` can import it without
pulling the rest of the package, and so build tooling can scrape the value.
"""

from __future__ import annotations

__version__ = "0.0.1"
