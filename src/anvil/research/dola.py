"""DoLa (Decoding by Contrasting Layers) — Chuang et al. 2023.

This is a placeholder so the public symbol exists in v0. The full processor
needs hidden-state extraction wired through both the HF and vLLM backends —
that's M2/M3 territory in this codebase. Calling :meth:`process` raises
``NotImplementedError`` with a pointer to the design section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DoLa:
    """Per-request logits processor implementing DoLa contrastive decoding.

    Args:
        mature_layer: the "trustworthy" layer (default ``-1``, the last layer).
        premature_layers: layers to contrast against (default ``(0, 12, 24)``).
    """

    requires_hidden_states: bool = True
    argmax_invariant: bool = False

    def __init__(
        self,
        mature_layer: int = -1,
        premature_layers: tuple[int, ...] = (0, 12, 24),
    ) -> None:
        self.mature_layer = mature_layer
        self.premature_layers = premature_layers

    def process(
        self,
        request_id: str,
        token_ids: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "DoLa needs hidden-state extraction wired through the engine wrapper "
            "(see design §4.4 / §4.5). Implementation lands in v0.5."
        )


__all__ = ["DoLa"]
