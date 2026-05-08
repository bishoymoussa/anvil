"""Pure-Python rule engine: error signature + KB → typed Action (design §7.3).

Matching rules:

1. Iterate KB entries. For each, check the engine constraints against the
   active engine name + version. Skip if no match.
2. For each ``signature`` in the entry, evaluate it against the captured
   error string. Two signature shapes:

   * ``model_id_matches: <regex>`` — special prefix; the regex is matched
     against the model id, not the error string.
   * Anything else — compiled regex, matched against the error string.

3. The entry fires if **any** of its signatures match (OR semantics).
4. The first matching entry wins. ``KBEntry.id`` defines a stable order
   when ties occur; in practice each entry's signatures are distinct.

The function :func:`match` returns ``Optional[Match]`` so callers can
distinguish "no fix in KB" from "fix found". :func:`resolve_value` resolves
KB ``value_strategy`` into a concrete value using a passed-in
:class:`Context`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern
from typing import TYPE_CHECKING, Any

from anvil.caas.actions import Action
from anvil.exceptions import CaaSError
from anvil.logging import get_logger

if TYPE_CHECKING:
    from anvil.caas.kb.schema import EngineConstraint, FixSpec, KBEntry

_log = get_logger(__name__)

# Special pseudo-signature prefix for matching against the model id.
_MODEL_ID_PREFIX = "model_id_matches:"


@dataclass
class Context:
    """The runtime data the rule engine inspects when matching + resolving values.

    The fields here cover what every shipped KB entry needs; new
    ``value_strategy`` values will require new fields (and a corresponding
    branch in :func:`resolve_value`).
    """

    error: str = ""
    """The captured error / signal string."""

    model_id: str = ""
    """HF model id of the active model (matched by ``model_id_matches:``)."""

    engine_name: str = "anvil"
    """``vllm`` | ``transformers`` | ``anvil`` | …"""

    engine_version: str = "0"
    """The installed engine's version, used to match ``KBEntry.engines[*].spec``."""

    available_gpus: int = 1
    """For ``largest_divisor_of_attn_heads``."""

    num_attention_heads: int | None = None
    """For ``largest_divisor_of_attn_heads``."""

    max_input_token_length: int | None = None
    """For ``max_input_token_length_plus_max_output``."""

    max_output_tokens: int = 0
    """For ``max_input_token_length_plus_max_output``."""

    max_image_pixels: int | None = None
    """For ``max_pixels_from_dataset``."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Strategy-specific extras (model card lookups, etc.)."""


@dataclass
class Match:
    """A KB entry matched the active context, plus the resolved Action."""

    entry: KBEntry
    action: Action


def match(ctx: Context, kb: list[KBEntry]) -> Match | None:
    """Find the first KB entry that fires for ``ctx``. Return the resolved Action.

    Returns ``None`` if no entry matches — the caller falls back to
    surfacing the raw error to the user.
    """
    for entry in kb:
        if not _engines_match(ctx, entry.engines):
            continue
        if not _signatures_match(ctx, entry.signatures):
            continue
        try:
            action = _action_from_fix(ctx, entry)
        except CaaSError as exc:
            _log.warning("KB entry %s matched but action build failed: %s", entry.id, exc)
            continue
        _log.info("rule_engine matched KB entry %s", entry.id)
        return Match(entry=entry, action=action)
    return None


def _engines_match(ctx: Context, constraints: list[EngineConstraint]) -> bool:
    """Apply the engine-version constraint per design §13.4 (KB rot mitigation)."""
    if not constraints:
        return True
    for constraint in constraints:
        if constraint.engine != "any" and constraint.engine != ctx.engine_name:
            continue
        if _version_matches(ctx.engine_version, constraint.spec):
            return True
    return False


def _version_matches(version: str, spec: str) -> bool:
    """Best-effort PEP 440 spec match. Falls back to ">=0" semantics on parse error.

    We parse via :mod:`packaging` if available (it's a transitive of
    pip/uv). We deliberately don't add packaging as a hard dep — the KB
    only carries simple specifiers in v0.
    """
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:  # pragma: no cover - packaging is a transitive
        return True  # be permissive when we can't parse
    try:
        return Version(version) in SpecifierSet(spec)
    except Exception:  # noqa: BLE001
        return True  # tolerate odd version strings (git SHAs etc.)


def _signatures_match(ctx: Context, signatures: list[str]) -> bool:
    """Test signature semantics. Two signature kinds with different combinators:

    * ``model_id_matches: <regex>`` — **conjunctive prerequisite**. ALL
      such signatures must match the active model id, otherwise the entry
      can't fire. This is the model-family filter pattern (e.g.
      ``llama3_eot_runaway`` only applies to Llama-3-Instruct family).
    * Any other regex — **disjunctive trigger**. At least one such
      signature must match the captured error. If the entry has no such
      signatures, the model-id filter alone is enough (rare).

    This split is what makes ``llama3_eot_runaway`` not collide with
    ``tp_attn_heads_divisibility`` on a Llama-3-Instruct TP-error: both
    entries' model-id filters (or lack thereof) accept, but only the TP
    entry's error-string trigger matches.
    """
    model_id_filters = [s for s in signatures if s.startswith(_MODEL_ID_PREFIX)]
    error_triggers = [s for s in signatures if not s.startswith(_MODEL_ID_PREFIX)]

    for filt in model_id_filters:
        pattern = filt[len(_MODEL_ID_PREFIX) :].strip()
        if not _safe_search(pattern, ctx.model_id):
            return False  # model-family filter rejects

    if not error_triggers:
        # Filter-only entry: model-id match alone is sufficient.
        return bool(model_id_filters)

    return any(_safe_search(trigger, ctx.error) for trigger in error_triggers)


def _safe_search(pattern: str, text: str) -> Pattern[str] | bool | None:
    """Compile + search, returning a truthy match or False."""
    try:
        compiled = re.compile(pattern)
    except re.error:  # pragma: no cover - schema validates regex at load
        return False
    return bool(compiled.search(text))


# -------------------------------------------------------------- value resolution


def _action_from_fix(ctx: Context, entry: KBEntry) -> Action:
    """Translate the KBEntry's typed FixSpec into a concrete :class:`Action`."""
    fix = entry.fix
    value = _resolve_value(ctx, fix)

    rationale = _format_rationale(ctx, entry, value)

    return Action(
        type=fix.type,
        kb_entry_id=entry.id,
        flag=fix.flag,
        name=fix.name,
        value=value,
        rationale=rationale,
        confidence=1.0,
        requires_user_consent=entry.requires_user_consent,
    )


def _resolve_value(ctx: Context, fix: FixSpec) -> Any:
    """Resolve ``fix.value`` or ``fix.value_strategy`` into a concrete scalar."""
    if fix.value_strategy is None:
        return fix.value

    strat = fix.value_strategy
    if strat == "literal":
        return fix.value

    if strat == "largest_divisor_of_attn_heads":
        if ctx.num_attention_heads is None:
            raise CaaSError(
                "strategy 'largest_divisor_of_attn_heads' requires Context.num_attention_heads"
            )
        return _largest_divisor(ctx.num_attention_heads, ctx.available_gpus)

    if strat == "max_input_token_length_plus_max_output":
        if ctx.max_input_token_length is None:
            raise CaaSError(
                "strategy 'max_input_token_length_plus_max_output' requires "
                "Context.max_input_token_length"
            )
        return ctx.max_input_token_length + ctx.max_output_tokens

    if strat == "max_pixels_from_dataset":
        if ctx.max_image_pixels is None:
            raise CaaSError("strategy 'max_pixels_from_dataset' requires Context.max_image_pixels")
        # Bucket to the model's expected granularity (Qwen-VL uses 28x28 patches).
        # Round down to the nearest multiple of 28*28.
        granule = 28 * 28
        return (ctx.max_image_pixels // granule) * granule

    if strat == "lookup_from_model_card":
        # The actual model-card lookup is a v0.5 feature; for v0 the action
        # carries the strategy name and the user resolves it in the diff.
        return ctx.extra.get("from_model_card")

    raise CaaSError(f"unknown value_strategy {strat!r}")


def _largest_divisor(heads: int, max_value: int) -> int:
    """Return the largest divisor of ``heads`` that is ``≤ max_value`` (and ≥ 1)."""
    capped = min(max_value, heads)
    for d in range(capped, 0, -1):
        if heads % d == 0:
            return d
    return 1  # unreachable: 1 always divides


def _format_rationale(ctx: Context, entry: KBEntry, value: Any) -> str:
    """Render the human_message with simple ``{placeholder}`` substitution."""
    subs: dict[str, Any] = {
        "fix_value": value,
        "model_id": ctx.model_id,
        "heads": ctx.num_attention_heads,
        "tp": ctx.extra.get("tp_size"),
        "max_in": ctx.max_input_token_length,
        "max_out": ctx.max_output_tokens,
        "max_dim": ctx.max_image_pixels,
        "current": ctx.extra.get("current"),
        "detected": ctx.extra.get("detected"),
    }
    try:
        return entry.human_message.strip().format(**subs)
    except (KeyError, IndexError):
        # Missing substitution fields → return the raw template; better to
        # see ``{heads}`` than to error.
        return entry.human_message.strip()


__all__ = ["Context", "Match", "match"]
