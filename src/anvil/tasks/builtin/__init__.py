"""Built-in benchmark tasks (design §6.5).

Importing this module registers every task under
:func:`anvil.tasks.registry.register_task`. M0 ships GSM8K only;
M1 adds MMLU and HumanEval+, M4 adds MMMU.
"""

from __future__ import annotations

# Importing each module triggers @register_task. Don't re-format this list —
# F401 is silenced via noqa so unused-import doesn't drop the side effect.
from anvil.tasks.builtin import gsm8k as _gsm8k  # noqa: F401
from anvil.tasks.builtin import humaneval as _humaneval  # noqa: F401
from anvil.tasks.builtin import mmlu as _mmlu  # noqa: F401

__all__: list[str] = []
