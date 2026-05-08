"""Slow-path model loaders (design §5.1).

Currently only the causal-LM HF path lands in M0 (it is implemented inside
:mod:`anvil.engine._hf.runner`). The non-causal/multimodal slow paths are
M4 / M5 work.
"""

from __future__ import annotations
