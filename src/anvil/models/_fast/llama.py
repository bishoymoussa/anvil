"""Llama family fast-path marker (design §5.2).

Llama 2, 3.1, 3.2, 3.3 (and Llama 4 once transformers ships it). v0
routes through the HF slow path; the kernel-level fast path with custom
attention + fused MLPs lands in subsequent releases. The §5.2 promise
holds: any HF-loadable Llama works *today* via the slow path.

Anvil-specific defaults applied at routing time:

* The §16.7 KB entry ``llama3_eot_runaway`` adds ``<|eot_id|>`` (128009)
  to ``stop_token_ids`` for Llama-3-Instruct sampling. The CaaS rule
  engine engages this when the user picks a Llama-3-Instruct model;
  no manual config required.
"""

from __future__ import annotations

from anvil.models.registry import register_model_impl


@register_model_impl("LlamaForCausalLM")
class LlamaFast:
    """Llama family fast-path marker (Llama 2/3.x/4 text)."""

    family = "llama"


__all__ = ["LlamaFast"]
