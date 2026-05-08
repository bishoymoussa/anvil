"""Mistral family fast-path marker (design §5.2).

Covers Mistral 7B, Small, Large, Mistral 3.1. Mixtral (the MoE variant)
gets its own marker class — the architectures differ at the
transformers level (``MistralForCausalLM`` vs ``MixtralForCausalLM``).
"""

from __future__ import annotations

from anvil.models.registry import register_model_impl


@register_model_impl("MistralForCausalLM")
class MistralFast:
    """Mistral family fast-path marker (dense 7B, Small, Large, 3.1)."""

    family = "mistral"


@register_model_impl("MixtralForCausalLM")
class MixtralFast:
    """Mixtral family fast-path marker (8x7B, 8x22B MoE)."""

    family = "mixtral"


__all__ = ["MistralFast", "MixtralFast"]
