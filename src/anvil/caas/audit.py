"""CaaS audit log (design §7.9).

Every CaaS engagement emits a :class:`anvil.manifest.schema.CaaSAction`
record. The records flow into the ``caas_log`` field of the run's
:class:`Manifest` so a reviewer can audit every applied delta.

The audit log is **append-only** — actions are recorded as they happen and
never retroactively edited. Verifying a manifest verifies the audit log
too (it's part of the canonical bytes the signature covers).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from anvil.manifest.schema import CaaSAction

if TYPE_CHECKING:
    from anvil.caas.actions import Action, ApplyResult


@dataclass
class AuditLog:
    """A growing list of :class:`CaaSAction`. Use :meth:`record` to append."""

    actions: list[CaaSAction] = field(default_factory=list)

    def record(
        self,
        *,
        step: int,
        trigger: str,
        action: Action,
        match_source: str,
        result: ApplyResult,
        validator_result: str = "skipped",
        llm_tokens_in: int = 0,
        llm_tokens_out: int = 0,
    ) -> CaaSAction:
        """Append one record and return it.

        Args:
            step: ordinal of this action within the run (1-indexed).
            trigger: the captured error string (or sentinel description).
            action: the resolved :class:`Action`.
            match_source: ``"rule_engine" | "kb" | "llm"``.
            result: the :class:`ApplyResult` returned by ``apply_action``.
            validator_result: outcome of the post-fix smoke re-test.
            llm_tokens_in / llm_tokens_out: 0 in v0 (no LLM tier).
        """
        record = CaaSAction(
            ts=_dt.datetime.now(_dt.UTC).isoformat(),
            step=step,
            trigger=trigger,
            match_source=match_source,
            action=action.type,
            args=action.args_dict(),
            rationale=action.rationale,
            confidence=action.confidence,
            kb_entry_ids=[action.kb_entry_id] if action.kb_entry_id else [],
            previous_value=result.previous_value,
            validator_result=validator_result,
            llm_tokens_in=llm_tokens_in,
            llm_tokens_out=llm_tokens_out,
        )
        self.actions.append(record)
        return record

    def __len__(self) -> int:
        return len(self.actions)

    def to_list(self) -> list[CaaSAction]:
        return list(self.actions)


__all__ = ["AuditLog"]
