"""Per-request logits processor proxy for vLLM (design §3.1, §4.4).

vLLM V1 dropped per-request ``logits_processors`` (RFC #13360, issue #21672).
Anvil restores the V0-style API in user space by intercepting the engine's
forward pass, dispatching each request's :class:`LogitsProcessor` chain,
and feeding back the modified logits.

For M1 this is a documented stub: simple greedy/temperature samplers do
not need the proxy, and the M1 acceptance tests (MMLU, HumanEval+) use
sampler-only requests. The full proxy lands in M2 alongside
:class:`HiddenStateSpec` plumbing.

Calling :func:`apply` raises ``NotImplementedError`` rather than silently
passing through, so research code that depends on this surface fails loudly
with a pointer to the design section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from anvil.primitives.logits_processor import LogitsProcessor


def apply(
    request_id: str,
    token_ids: torch.Tensor,
    logits: torch.Tensor,
    hidden_states: torch.Tensor | None,
    processors: tuple[LogitsProcessor, ...],
) -> torch.Tensor:
    """Run each ``LogitsProcessor`` on the request's logits.

    Currently raises — implementation lands in M2 (design §16.10).
    """
    del request_id, token_ids, logits, hidden_states, processors
    raise NotImplementedError(
        "Per-request logits processors for the vLLM backend land in M2 "
        "(design §3.1, §4.4). For now, use the HF backend "
        "(``engine='hf'``) for research workflows that need them."
    )


__all__ = ["apply"]
