"""CaaS preflight orchestrator (design §7.1, §7.2, §7.6).

Sequence:

1. **Smoke-test setup.** The runner discovers the dataset's longest input,
   the random middle, and the canonical short. (Discovery is the runner's
   job; this module receives the prepared probes.)
2. **Sentinel.** Always runs the quality sentinel. A divergence beyond
   threshold engages CaaS even without an exception (design §6.6).
3. **Engagement.** A captured error / sentinel-fail is turned into a
   :class:`Context`, matched against the KB by the rule engine. A
   :class:`Match` is rendered to the user in research mode for confirmation;
   ci mode auto-applies if the match is high-confidence.
4. **Audit.** Every action is appended to the run's :class:`AuditLog` so
   the records flow into the manifest's ``caas_log`` field.

Modes (design §7.6):

* ``off`` — disabled.
* ``advisory`` — print the proposed diff, never apply.
* ``research`` (default) — present a unified diff, ask
  ``[y/n/edit/explain]``.
* ``ci`` — auto-apply if (rule_engine | KB) match AND severity ≠
  review-required AND not ``requires_user_consent``. Otherwise abort.

The v1 LLM tier is **not** part of this module. ``llm_tier.py`` is a
documented stub raising ``NotImplementedError`` (design §7.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from anvil.caas.actions import Action, apply_action
from anvil.caas.audit import AuditLog
from anvil.caas.rule_engine import Context, Match, match
from anvil.exceptions import CaaSCannotFix
from anvil.logging import get_logger

if TYPE_CHECKING:
    from anvil.caas.kb.schema import KBEntry
    from anvil.caas.sentinel import SentinelResult

CaaSMode = Literal["off", "advisory", "research", "ci"]

_log = get_logger(__name__)


@dataclass
class PreflightOutcome:
    """The result of running CaaS preflight on a (potential) failure.

    Attributes:
        resolved: True iff CaaS applied a fix that the user accepted (or ci
            auto-applied) and the post-fix smoke-test passed.
        match: the :class:`Match` returned by the rule engine, or ``None``
            if no KB entry fired.
        action: the resolved :class:`Action`, or ``None``.
        log: the :class:`AuditLog` containing every recorded record.
        sentinel: the :class:`SentinelResult` if the sentinel ran.
        message: human-readable summary (printed by the CLI, included in
            advisory-mode output).
    """

    resolved: bool
    match: Match | None = None
    action: Action | None = None
    log: AuditLog = field(default_factory=AuditLog)
    sentinel: SentinelResult | None = None
    message: str = ""


def _confirm(prompt: str) -> bool:
    """Default confirmation hook for ``research`` mode.

    Splits this out so tests can monkeypatch a boolean. Reads from stdin;
    accepts ``y`` / ``yes`` (case-insensitive). Anything else means no.
    """
    try:
        ans = input(f"{prompt} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


def engage(
    *,
    error: str,
    kb: list[KBEntry],
    ctx: Context,
    mode: CaaSMode = "research",
    engine_args: dict[str, object] | None = None,
    sampling_args: dict[str, object] | None = None,
    log: AuditLog | None = None,
    confirm: object = _confirm,
    allow_install: bool = False,
) -> PreflightOutcome:
    """Run the preflight engagement against a captured failure.

    Args:
        error: the captured error string (the rule engine matches against this).
        kb: the loaded :class:`KBEntry` list (caller passes :func:`load_all()`).
        ctx: a populated :class:`Context` for matching + value resolution.
        mode: the CaaS mode (``off | advisory | research | ci``).
        engine_args / sampling_args: dicts the action will mutate.
        log: an optional shared :class:`AuditLog`. Created if not provided.
        confirm: callable accepting a prompt and returning ``bool``;
            defaults to ``input(...)``.
        allow_install: gate for ``install_package`` actions.

    Returns:
        :class:`PreflightOutcome` describing what was matched, applied, and
        what the post-fix sentinel said. The caller is responsible for
        re-running the smoke probe and updating ``validator_result``.

    Raises:
        CaaSCannotFix: only if the active mode promises a hard outcome (ci)
            and the match couldn't be applied. Other modes return an
            ``unresolved`` outcome and let the caller decide.
    """
    audit = log or AuditLog()

    if mode == "off":
        return PreflightOutcome(
            resolved=False,
            log=audit,
            message="CaaS disabled (--caas=off); error not auto-fixed.",
        )

    m = match(ctx, kb)
    if m is None:
        # The rule engine couldn't match. v0 has no LLM tier; we surface.
        return PreflightOutcome(
            resolved=False,
            log=audit,
            message="No KB entry matched the captured error. Surfacing verbatim.",
        )

    action = m.action
    proposed = _render_diff(m)

    if mode == "advisory":
        _log.info("CaaS advisory:\n%s", proposed)
        return PreflightOutcome(
            resolved=False,
            match=m,
            action=action,
            log=audit,
            message=proposed,
        )

    if mode == "research":
        if not callable(confirm):  # pragma: no cover - typing guard
            raise CaaSCannotFix("research mode requires a callable confirm hook")
        accepted = confirm(proposed + "\n\nApply? [y/N]")
        if not accepted:
            return PreflightOutcome(
                resolved=False,
                match=m,
                action=action,
                log=audit,
                message="user rejected the proposed fix",
            )

    if mode == "ci":
        if action.requires_user_consent or m.entry.severity == "review-required":
            raise CaaSCannotFix(
                f"action {action.type!r} from KB entry {m.entry.id!r} requires "
                "explicit user consent and cannot run in --caas=ci mode"
            )
        if action.confidence < 0.7:
            raise CaaSCannotFix(
                f"action confidence {action.confidence:.2f} < 0.7; ci mode demands ≥0.7"
            )

    # Apply.
    result = apply_action(
        action,
        engine_args=dict(engine_args) if engine_args is not None else {},
        sampling_args=dict(sampling_args) if sampling_args is not None else {},
        allow_install=allow_install,
    )
    if engine_args is not None:
        # Side-effect into caller's dict so the runner's next attempt picks it up.
        applied_args = dict(engine_args)
        if action.type == "set_engine_flag":
            applied_args[action.flag or ""] = action.value
        elif action.type == "unset_engine_flag":
            applied_args.pop(action.flag or "", None)
        engine_args.clear()
        engine_args.update(applied_args)
    if sampling_args is not None and action.type == "set_sampling_param":
        sampling_args[action.name or ""] = action.value

    audit.record(
        step=len(audit) + 1,
        trigger=error,
        action=action,
        match_source="rule_engine",
        result=result,
    )

    return PreflightOutcome(
        resolved=result.applied,
        match=m,
        action=action,
        log=audit,
        message=proposed if result.applied else result.reason,
    )


def _render_diff(m: Match) -> str:
    """Format a one-screen diff describing the proposed fix.

    The format mirrors §17 A.6: KB id, references, and the proposed change
    so the user can decide ``[y/n]`` from a glance.
    """
    refs = "\n        ".join(m.entry.references) if m.entry.references else "(none)"
    return (
        f"[caas]  Diagnosis (rule_engine + kb match: {m.entry.id}):\n"
        f"        {m.entry.human_message.strip()}\n"
        f"  Proposed fix:\n"
        f"        type={m.action.type} flag={m.action.flag} "
        f"name={m.action.name} value={m.action.value!r}\n"
        f"  References:\n        {refs}"
    )


__all__ = ["PreflightOutcome", "CaaSMode", "engage"]
