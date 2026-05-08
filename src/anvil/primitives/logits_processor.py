"""``LogitsProcessor`` — V0-style per-request logits processor API (design §4.4).

This is the API vLLM V1 dropped (RFC #13360, issue #21672). Anvil restores it
in user space: the engine wrapper handles batching, hidden-state plumbing,
and argmax-invariance fast paths. Researchers write a single ``process``
method per *request*; the engine takes care of everything else.

Two metadata flags drive the engine:

* ``requires_hidden_states``: if any active processor sets this, the engine
  configures :class:`~anvil.primitives.hidden_state_spec.HiddenStateSpec` for
  the forward pass and threads activations through.
* ``argmax_invariant``: if every active processor sets this and the sampler
  is greedy, the engine takes a fast path that skips per-token logit copies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


@runtime_checkable
class LogitsProcessor(Protocol):
    """Per-request logits processor.

    The engine batches requests; processors that don't apply to a particular
    request should return its logits unchanged. The wrapper layer
    (``anvil.engine._wrappers.logits_processor``) handles the batching and
    request-id dispatch — research code only writes :meth:`process`.

    Example:
        >>> class ZeroOutEOS:
        ...     requires_hidden_states = False
        ...     argmax_invariant = False  # we modify the argmax
        ...     def __init__(self, eos_id: int) -> None:
        ...         self.eos_id = eos_id
        ...     def process(self, request_id, token_ids, logits, hidden_states):
        ...         logits = logits.clone()
        ...         logits[..., self.eos_id] = float("-inf")
        ...         return logits
        >>> isinstance(ZeroOutEOS(2), LogitsProcessor)
        True
    """

    requires_hidden_states: bool
    argmax_invariant: bool

    def process(
        self,
        request_id: str,
        token_ids: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Transform a single request's logits in place or return a new tensor.

        Args:
            request_id: stable identifier of the request inside the engine.
            token_ids: ``int64 [seq_len]`` — tokens generated so far.
            logits: ``float [vocab_size]`` — logits for the next token.
            hidden_states: ``float [num_layers, seq_len, hidden_dim]`` if
                ``requires_hidden_states`` is True, otherwise ``None``.

        Returns:
            ``float [vocab_size]`` — transformed logits.
        """
        ...


__all__ = ["LogitsProcessor"]
