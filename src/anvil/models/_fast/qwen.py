"""Qwen 2 / 2.5 / 3 family fast-path marker (design §5.2).

Text-only Qwen models route here; the multimodal Qwen-VL family lives
in :mod:`anvil.models._fast.qwen_vl` because the routing decision
(VLM vs text) needs to happen earlier in the engine factory.
"""

from __future__ import annotations

from anvil.models.registry import register_model_impl


@register_model_impl("Qwen2ForCausalLM")
class Qwen2Fast:
    """Qwen2 / 2.5 / 3 family fast-path marker (text)."""

    family = "qwen"


__all__ = ["Qwen2Fast"]
