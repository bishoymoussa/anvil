"""Phi family fast-path marker (design §5.2).

Phi 3 / 3.5 / 4 / 4-mini text models. Phi-3-Vision (and Phi-4-Vision)
routes through the VLM path automatically — those architectures
contain "Vision" / "Multi" markers the factory recognizes.
"""

from __future__ import annotations

from anvil.models.registry import register_model_impl


@register_model_impl("Phi3ForCausalLM")
class Phi3Fast:
    """Phi 3 / 3.5 family fast-path marker (text)."""

    family = "phi"


@register_model_impl("Phi4ForCausalLM")
class Phi4Fast:
    """Phi 4 / 4-mini family fast-path marker (text)."""

    family = "phi"


__all__ = ["Phi3Fast", "Phi4Fast"]
