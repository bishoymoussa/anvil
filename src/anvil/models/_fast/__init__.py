"""Fast-path model implementations (design §5.2).

Each architecture lives in its own ``<family>.py`` and registers via
:func:`anvil.models.registry.register_model_impl`. Importing this package
triggers all registrations as a side effect.

v0 fast-path roster (design §5.2 + §16.10 M6 acceptance):

* Llama 2 / 3.x / 4 — :mod:`.llama`
* Qwen 2 / 2.5 / 3 (text) — :mod:`.qwen`
* Mistral / Mixtral — :mod:`.mistral`
* Gemma 2 / 3 (text) — :mod:`.gemma`
* Phi 3 / 4 — :mod:`.phi`
* Qwen 2.5-VL (multimodal) — :mod:`.qwen_vl`

The kernel-level fast-paths (custom attention + fused MLPs) land
post-v0; the markers here let the factory route VLMs vs text and apply
family-specific defaults today.
"""

from __future__ import annotations

# Importing each module registers via @register_model_impl. F401 silenced.
from anvil.models._fast import gemma as _gemma  # noqa: F401
from anvil.models._fast import llama as _llama  # noqa: F401
from anvil.models._fast import mistral as _mistral  # noqa: F401
from anvil.models._fast import phi as _phi  # noqa: F401
from anvil.models._fast import qwen as _qwen  # noqa: F401
from anvil.models._fast import qwen_vl as _qwen_vl  # noqa: F401
