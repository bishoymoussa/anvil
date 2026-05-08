"""Engine wrapper layer (design §3.1).

Reimplements abstractions vLLM V1 dropped — per-request logits processors
and hidden-state extraction — on top of any backend. Empty in M0; lands in
M2 alongside :class:`HiddenStateSpec` plumbing.
"""

from __future__ import annotations
