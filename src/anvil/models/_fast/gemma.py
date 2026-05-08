"""Gemma family fast-path marker (design §5.2).

Gemma 2 and Gemma 3 share ``GemmaForCausalLM`` / ``Gemma2ForCausalLM`` /
``Gemma3ForCausalLM`` at the transformers level depending on revision;
we register the canonical names. Gemma 3 vision (Gemma 3 4B IT-VLM) is a
separate ``Gemma3ForConditionalGeneration`` and routes through the VLM
path automatically (factory's ``_is_multimodal`` substring match).
"""

from __future__ import annotations

from anvil.models.registry import register_model_impl


@register_model_impl("Gemma2ForCausalLM")
class Gemma2Fast:
    """Gemma 2 family fast-path marker."""

    family = "gemma"


@register_model_impl("Gemma3ForCausalLM")
class Gemma3Fast:
    """Gemma 3 family fast-path marker (text)."""

    family = "gemma"


__all__ = ["Gemma2Fast", "Gemma3Fast"]
