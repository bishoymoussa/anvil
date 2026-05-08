"""KB entry schema (design §16.6).

Every entry is a typed object with regex error signatures, an engine-version
constraint, and a deterministic fix. The runtime rule engine matches a
captured error against ``signatures``, dispatches the typed ``fix``, and
writes a :class:`anvil.manifest.schema.CaaSAction` record.

Three rules the schema enforces:

1. ``id`` is a stable lowercase identifier (regex-locked) — used for
   citations, audit logs, and disabling individual entries.
2. ``signatures`` are regex *patterns* — they're compiled and matched
   exactly (we do **not** let an LLM match regex; the design forbids it).
3. ``engines`` is a list of (engine, spec) pairs; the entry fires only
   when the active engine's version matches one of the listed specs (PEP
   440-ish, parsed by :mod:`packaging.specifiers`).

A few entries (``glm47_requires_transformers_main``,
``cuda_libcudart_version_mismatch``) carry ``fix.type='give_up'`` —
intentional: surface to the user, never auto-fix.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["blocking", "warning", "review-required"]
"""Match-vs-fix risk level. ``blocking`` aborts the run if not fixed;
``warning`` is reported but not auto-fail; ``review-required`` always
demands human confirmation regardless of CaaS mode."""

Category = Literal[
    "install",
    "model_loading",
    "memory",
    "tokenization",
    "sampler",
    "multimodal",
    "tool_calling",
    "harness",
    "parallelism",
]


class EngineConstraint(BaseModel):
    """A version pin against a particular engine (design §7.3).

    ``engine`` is the backend name (``"vllm"``, ``"transformers"``,
    ``"anvil"``, or ``"any"``). ``spec`` is a PEP 440-style specifier
    (``">=0.5,<0.20"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    engine: str
    spec: str


FixType = Literal[
    "set_engine_flag",
    "unset_engine_flag",
    "set_env_var",
    "edit_yaml",
    "set_sampling_param",
    "install_package",
    "restart_engine",
    "give_up",
]


_VALUE_STRATEGIES: frozenset[str] = frozenset(
    [
        "literal",
        "lookup_from_model_card",
        "largest_divisor_of_attn_heads",
        "max_pixels_from_dataset",
        "max_input_token_length_plus_max_output",
    ]
)


class FixSpec(BaseModel):
    """A typed, allow-listed action the rule engine may apply (design §7.5)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: FixType
    flag: str | None = None
    name: str | None = None
    value: Any | None = None
    value_strategy: str | None = None

    @field_validator("value_strategy")
    @classmethod
    def _strategy_in_allowlist(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if v not in _VALUE_STRATEGIES:
            raise ValueError(f"value_strategy={v!r} not in allow-list {sorted(_VALUE_STRATEGIES)}")
        return v


_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class KBEntry(BaseModel):
    """One entry in the curated known-issue database (design §16.6).

    Attributes:
        id: stable lowercase identifier. Used for citations and
            audit-log :attr:`CaaSAction.kb_entry_ids`.
        category: organizational tag — ``install``, ``memory``,
            ``tokenization``, … see :data:`Category`.
        engines: which engines this entry fires for. The rule engine
            consults the active engine's version against each spec.
        signatures: regex patterns matched against the captured error
            string; the entry fires if **any** signature matches.
        fix: the typed action to apply.
        severity: ``blocking`` (abort if unfixed), ``warning``, or
            ``review-required`` (always asks).
        references: GitHub issue URLs and similar citations.
        human_message: rendered to the user before the fix is applied
            (or surfaced if ``fix.type='give_up'``).
        expires_after: optional engine-version specifier after which the
            entry is considered stale (design §13.4 — KB rot mitigation).
        requires_user_consent: if True, never auto-apply even in
            ``--caas=ci`` mode. ``trust_remote_code`` is the canonical
            example (it's a code-execution surface).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: Category
    engines: list[EngineConstraint]
    signatures: list[str] = Field(default_factory=list, min_length=1)
    fix: FixSpec
    severity: Severity = "blocking"
    references: list[str] = Field(default_factory=list)
    human_message: str
    expires_after: str | None = None
    requires_user_consent: bool = False

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not _ID_PATTERN.match(v):
            raise ValueError(
                f"KB id {v!r} must match {_ID_PATTERN.pattern} (lowercase, "
                "starting with a letter, snake_case)"
            )
        return v

    @field_validator("signatures")
    @classmethod
    def _signatures_compile(cls, v: list[str]) -> list[str]:
        for sig in v:
            if sig.startswith("model_id_matches:"):
                # Special pseudo-signature: handled by the rule engine, not regex.
                continue
            try:
                re.compile(sig)
            except re.error as exc:
                raise ValueError(f"signature {sig!r} is not a valid regex: {exc}") from exc
        return v


__all__ = [
    "KBEntry",
    "FixSpec",
    "FixType",
    "EngineConstraint",
    "Severity",
    "Category",
]
