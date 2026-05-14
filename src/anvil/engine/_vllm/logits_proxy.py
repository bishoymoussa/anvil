"""Per-request logits processor proxy for vLLM (design §3.1, §4.4).

vLLM V1 dropped per-request ``logits_processors`` (RFC #13360, issue #21672).
Anvil restores the V0-style API in user space: :func:`apply` runs each
:class:`~anvil.primitives.logits_processor.LogitsProcessor` in the chain
for a single request, in order, returning the final transformed logits.

The proxy is called by the HF engine's generation loop and by any future
vLLM wrapper that intercepts the forward pass. It is intentionally
engine-agnostic — it only needs a ``torch.Tensor`` logits slice.

Fast path: if every processor in the chain sets ``argmax_invariant=True``
and the sampler is greedy (detected by ``hidden_states is None`` — a
convention in our vLLM wrapper), the proxy can skip copying the logits
tensor. This is enforced by the caller, not here; the proxy always applies.
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
    """Run each ``LogitsProcessor`` on the request's logits, in order.

    Args:
        request_id: stable identifier of the request inside the engine.
        token_ids: ``int64 [seq_len]`` — tokens generated so far.
        logits: ``float [vocab_size]`` — logits for the next token.
        hidden_states: ``float [num_layers, seq_len, hidden_dim]`` if any
            processor sets ``requires_hidden_states=True``, otherwise ``None``.
        processors: ordered chain of processors to apply.

    Returns:
        Transformed ``float [vocab_size]`` logits tensor.
    """
    out = logits
    for proc in processors:
        out = proc.process(request_id, token_ids, out, hidden_states)
    return out


def build_vllm_logits_processor(
    request_id: str,
    processors: tuple[LogitsProcessor, ...],
) -> object:
    """Return a vLLM-compatible logits-processor callable for ``SamplingParams``.

    vLLM 0.20.x accepts a list of callables in ``SamplingParams.logits_processors``.
    Each callable receives ``(token_ids: List[int], logits: Tensor) -> Tensor``.
    We wrap our chain into that shape, passing ``hidden_states=None`` (the
    vLLM path does not expose hidden states — use the HF engine for DoLa etc.).

    Args:
        request_id: identifier forwarded to each processor.
        processors: the Anvil processor chain for this request.

    Returns:
        A callable suitable for ``vllm.SamplingParams(logits_processors=[...])``.
    """
    import torch as _torch

    def _proxy(token_ids_list: list[int], logits: _torch.Tensor) -> _torch.Tensor:
        token_ids = _torch.tensor(token_ids_list, dtype=_torch.long, device=logits.device)
        return apply(request_id, token_ids, logits, None, processors)

    return _proxy


__all__ = ["apply", "build_vllm_logits_processor"]
