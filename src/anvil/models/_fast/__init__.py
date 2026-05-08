"""Fast-path model implementations (design §5.2).

Each architecture lives in its own ``<family>.py`` and registers via
:func:`anvil.models.registry.register_model_impl`. M0 ships none; M1 lands
the first round (Llama 3, Qwen 2.5, Mistral, Gemma 3, Phi 4); M4 adds
Qwen-VL.
"""

from __future__ import annotations
