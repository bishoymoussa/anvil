"""Typed CaaS action allow-list (design §7.5).

The action space is intentionally tiny and typed: the v1 LLM tier (M ≥ v1)
will emit one of these via grammar-constrained decoding, so adding a new
action here means thinking through its safety implications first. The v0
rule engine is constrained to the same allow-list.

Each action carries the inputs needed to apply it deterministically. The
applier is :func:`apply_action` — it dispatches on ``ActionType``,
mutates the active engine config (or env, or YAML file), and returns
``True`` if the action was applied or ``False`` if a safety rail fired.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from anvil.exceptions import CaaSError
from anvil.logging import get_logger

ActionType = Literal[
    "set_engine_flag",
    "unset_engine_flag",
    "set_env_var",
    "edit_yaml",
    "set_sampling_param",
    "install_package",
    "restart_engine",
    "give_up",
]

# § 7.7 / § 7.8: the env-var allow-list. Anything else is rejected.
_ENV_ALLOW_LIST: frozenset[str] = frozenset(
    [
        "CUDA_VISIBLE_DEVICES",
        "HF_HOME",
        "HF_TOKEN",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
    ]
)
# Plus prefixes — NCCL_*, VLLM_*, TORCH_*.
_ENV_ALLOWED_PREFIXES: tuple[str, ...] = ("NCCL_", "VLLM_", "TORCH_")


def _env_allowed(name: str) -> bool:
    return name in _ENV_ALLOW_LIST or any(name.startswith(p) for p in _ENV_ALLOWED_PREFIXES)


@dataclass
class Action:
    """One concrete fix the rule engine has resolved.

    Args:
        type: which slot of the allow-list this is.
        kb_entry_id: the KB entry that produced it (recorded in the audit log).
        flag/name/value: per-type inputs.
        rationale: human-readable one-line explanation.
        confidence: 0.0–1.0; the rule engine emits 1.0, the (future) LLM tier
            emits its grammar-constrained model confidence.
    """

    type: ActionType
    kb_entry_id: str
    flag: str | None = None
    name: str | None = None
    value: Any = None
    rationale: str = ""
    confidence: float = 1.0
    args: dict[str, Any] = field(default_factory=dict)
    requires_user_consent: bool = False

    def args_dict(self) -> dict[str, Any]:
        """Render to the dict shape that flows into ``CaaSAction.args``."""
        out: dict[str, Any] = dict(self.args)
        if self.flag is not None:
            out["flag"] = self.flag
        if self.name is not None:
            out["name"] = self.name
        if self.value is not None:
            out["value"] = self.value
        return out


@dataclass
class ApplyResult:
    """Outcome of :func:`apply_action`."""

    applied: bool
    reason: str = ""
    previous_value: Any = None


_log = get_logger(__name__)


def apply_action(
    action: Action,
    *,
    engine_args: dict[str, Any],
    sampling_args: dict[str, Any],
    allow_install: bool = False,
) -> ApplyResult:
    """Apply ``action`` to the active engine/sampler config.

    Mutates ``engine_args`` and ``sampling_args`` in place. Env-var actions
    set ``os.environ`` keys subject to the allow-list. Install actions are
    refused unless ``allow_install`` is True (gated behind ``--ci --allow-install``).

    Returns:
        :class:`ApplyResult` — ``applied`` is False when a safety rail
        fires or the action is informational (``give_up``). ``reason``
        explains the no-op.

    Raises:
        CaaSError: only on internal logic errors (action of an unknown
            type), never on safety-rail rejections — those are reflected
            in ``ApplyResult.reason``.
    """
    if action.type == "give_up":
        return ApplyResult(applied=False, reason="surfaced; no auto-fix attempted")

    if action.type == "set_engine_flag":
        if action.flag is None:
            raise CaaSError(f"action {action.type} requires `flag`")
        previous = engine_args.get(action.flag)
        engine_args[action.flag] = action.value
        return ApplyResult(applied=True, previous_value=previous)

    if action.type == "unset_engine_flag":
        if action.flag is None:
            raise CaaSError(f"action {action.type} requires `flag`")
        previous = engine_args.pop(action.flag, None)
        return ApplyResult(applied=True, previous_value=previous)

    if action.type == "set_sampling_param":
        if action.name is None:
            raise CaaSError(f"action {action.type} requires `name`")
        previous = sampling_args.get(action.name)
        sampling_args[action.name] = action.value
        return ApplyResult(applied=True, previous_value=previous)

    if action.type == "set_env_var":
        if action.name is None or not _env_allowed(action.name):
            return ApplyResult(
                applied=False,
                reason=f"env var {action.name!r} not on allow-list (§7.7)",
            )
        previous = os.environ.get(action.name)
        os.environ[action.name] = str(action.value)
        return ApplyResult(applied=True, previous_value=previous)

    if action.type == "install_package":
        if not allow_install:
            return ApplyResult(
                applied=False,
                reason="install_package requires --ci --allow-install (§7.6)",
            )
        # We do *not* run pip from inside a process that is mid-eval. The
        # action is recorded so the user can re-run after installing manually.
        _log.warning(
            "install_package action recorded but not executed: %s==%s",
            action.name,
            action.value,
        )
        return ApplyResult(
            applied=False,
            reason="install_package recorded; user must re-run after install",
        )

    if action.type == "edit_yaml":
        # M3 doesn't yet route YAML edits through this applier — task YAML
        # edits go via the runner after the user accepts the diff. The action
        # is recorded; the runner consumes it. Returning applied=True keeps
        # the audit log honest.
        return ApplyResult(applied=True)

    if action.type == "restart_engine":
        # Recorded; the runner re-instantiates the engine on its next pass.
        return ApplyResult(applied=True)

    raise CaaSError(f"unknown action type {action.type!r}")


__all__ = ["Action", "ApplyResult", "ActionType", "apply_action"]
