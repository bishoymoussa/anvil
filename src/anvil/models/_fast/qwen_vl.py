"""Qwen2.5-VL fast-path marker (design §5.4, §5.2 fast-path table).

For v0 the "fast path" for VLMs is mostly a routing decision: the same
``HFVLMEngine`` slow path runs the model, but registration here lets the
engine factory recognize the architecture and apply Qwen-VL-specific
defaults (lifted to the engine layer in
:mod:`anvil.engine._hf.mm_runner` to satisfy the layered import graph).

The kernel-level fast path with custom attention + fused MLPs is M6 work
(design §5.2). The wedge M4 ships is *correctness*: image preprocessing,
image-token-span tracking, and CaaS-aware preflight all flow through the
slow-path machinery.
"""

from __future__ import annotations

from anvil.engine._hf.mm_runner import (
    QWEN_VL_ARCHITECTURES,
    apply_qwen_vl_defaults,
    is_qwen_vl,
)
from anvil.models.registry import register_model_impl


@register_model_impl("Qwen2_5_VLForConditionalGeneration")
class Qwen2_5_VL_Fast:  # noqa: N801 — mirrors transformers' Qwen2_5_VLForConditionalGeneration
    """Marker class for Qwen2.5-VL family (§5.2 fast-path table)."""

    family = "qwen-vl"


__all__ = [
    "Qwen2_5_VL_Fast",
    "QWEN_VL_ARCHITECTURES",
    "apply_qwen_vl_defaults",
    "is_qwen_vl",
]
