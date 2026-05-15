"""DoLa (Decoding by Contrasting Layers) — Chuang et al. 2023.

The :class:`DoLa` processor implements the :class:`~anvil.primitives.logits_processor.LogitsProcessor`
protocol and is wired through the HF engine's hidden-state extraction path.

At each decoding step the engine passes:
- ``logits``: ``[vocab_size]`` from the mature layer (already computed).
- ``hidden_states``: ``[num_layers, 1, hidden_dim]`` for the current step.

DoLa subtracts the average premature-layer projection from the mature-layer
logits, amplifying tokens where the mature layer is more confident relative
to the earlier layers. This implements equation (3) from the paper.

Call :meth:`bind` with the loaded model before passing this processor to the
engine — the runner does this automatically when it detects a processor that
exposes a ``bind`` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class DoLa:
    """Per-request logits processor implementing DoLa contrastive decoding.

    Args:
        mature_layer: index of the "trustworthy" layer (default ``-1``, last).
        premature_layers: layers to contrast against (default ``(0, 12, 24)``).
        alpha: scaling factor applied to the premature distribution before
            subtraction. Default ``1.0`` matches the paper.
    """

    requires_hidden_states: bool = True
    argmax_invariant: bool = False

    def __init__(
        self,
        mature_layer: int = -1,
        premature_layers: tuple[int, ...] = (0, 12, 24),
        alpha: float = 1.0,
    ) -> None:
        self.mature_layer = mature_layer
        self.premature_layers = premature_layers
        self.alpha = alpha
        self._lm_head_weight: Any = None  # set by bind()
        self._norm: Any = None  # final layer norm, if separate from lm_head

    def bind(self, model: Any) -> None:
        """Extract the lm_head weight from the loaded model.

        Called automatically by the HF runner before generation starts.
        Supports the common HF pattern where the head weight is tied to the
        embedding (``model.lm_head.weight``) and an optional final layer norm
        (``model.model.norm`` or ``model.transformer.ln_f``).
        """

        lm_head = getattr(model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise ValueError(
                f"DoLa.bind: model {type(model).__name__!r} has no lm_head.weight. "
                "DoLa requires a decoder-only model with a linear language-model head."
            )
        # Keep as float32 for numerical stability during contrastive subtraction.
        self._lm_head_weight = lm_head.weight.detach().float()  # [vocab, hidden]

        # Try common final-layer-norm attribute names.
        model_body = getattr(model, "model", None) or getattr(model, "transformer", None)
        if model_body is not None:
            for attr in ("norm", "ln_f", "final_layer_norm"):
                norm = getattr(model_body, attr, None)
                if norm is not None:
                    self._norm = norm
                    break

    def process(
        self,
        request_id: str,
        token_ids: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> torch.Tensor:
        """Subtract average premature-layer logits from mature-layer logits.

        ``hidden_states`` shape: ``[num_layers, seq_len, hidden_dim]``.
        ``logits`` shape: ``[vocab_size]`` — the mature layer's prediction.
        """
        import torch

        if self._lm_head_weight is None:
            raise RuntimeError(
                "DoLa.process called before bind(). "
                "Pass this processor to the engine; it calls bind() automatically."
            )
        if hidden_states is None:
            raise RuntimeError(
                "DoLa.process received hidden_states=None. "
                "Set requires_hidden_states=True (it is) and ensure the engine "
                "is configured to extract hidden states."
            )

        num_layers = hidden_states.shape[0]
        # Clamp premature layer indices to valid range.
        premature = [p % num_layers for p in self.premature_layers]
        # Select premature hidden states: [len(premature), seq_len, hidden_dim]
        premature_hs = hidden_states[premature].float()

        # Apply final layer norm if available (matches what the model does
        # internally before projecting to vocab).
        if self._norm is not None:
            with torch.no_grad():
                # Process each layer's hidden state through the norm.
                normed = torch.stack([self._norm(premature_hs[i]) for i in range(len(premature))])
        else:
            normed = premature_hs

        # Average across premature layers, take the last token position.
        # Shape: [hidden_dim]
        avg_premature = normed.mean(dim=0)[-1]

        # Project to vocabulary. [hidden_dim] @ [hidden_dim, vocab] = [vocab]
        premature_logits = avg_premature @ self._lm_head_weight.T.to(avg_premature.device)

        # Contrastive: shift mature logits away from premature distribution.
        mature = logits.float()
        contrast = mature - self.alpha * premature_logits

        # Re-normalise so downstream sampling sees a proper log-prob scale.
        contrast = contrast - torch.logsumexp(contrast, dim=-1, keepdim=True)
        return contrast.to(logits.dtype)  # type: ignore[no-any-return]


__all__ = ["DoLa"]
