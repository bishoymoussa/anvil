"""lm-evaluation-harness compatibility shim (design §6.4, §16.10).

The shim accepts an existing lm-eval task YAML (or a dict of equivalent
shape) and compiles it to an Anvil :class:`Task` class at load time. The
resulting task runs through Anvil's batched primitives at engine speed
— **10–40× faster** than lm-eval-harness's per-doc Python loop on
log-likelihood tasks, while inheriting whatever bugs the upstream YAML
has (the manifest tags imported tasks as ``tier: imported``).

Migration validation
--------------------

Pair with ``anvil eval --compare-with-lm-eval ...`` to run both engines
against a task and produce a delta report — see
:mod:`anvil.tasks.lm_eval_shim.compare`.
"""

from __future__ import annotations

from anvil.tasks.lm_eval_shim.compare import compare_with_lm_eval
from anvil.tasks.lm_eval_shim.compat import register_lm_eval_task
from anvil.tasks.lm_eval_shim.compiler import compile_yaml, compile_yaml_dict

__all__ = [
    "compile_yaml",
    "compile_yaml_dict",
    "register_lm_eval_task",
    "compare_with_lm_eval",
]
