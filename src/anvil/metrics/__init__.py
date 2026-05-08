"""Metrics — pure functions over (predictions, targets) (design §16.1).

Metrics are a leaf in the import graph: they may import from
:mod:`anvil.primitives` only.
"""

from __future__ import annotations
