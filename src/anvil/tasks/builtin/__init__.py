"""Built-in benchmark tasks (design §6.5).

Importing this module registers every task under
:func:`anvil.tasks.registry.register_task`. M0 ships GSM8K only;
M1 adds MMLU and HumanEval+, M4 adds MMMU.
"""

from __future__ import annotations

from anvil.tasks.builtin import gsm8k as _gsm8k  # noqa: F401  (registers on import)

__all__: list[str] = []
