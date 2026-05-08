"""Manifest Pydantic schema (design §8.1, §16.4).

A :class:`Manifest` is the **primary output** of an evaluation, not a sidecar.
The library refuses to publish a leaderboard upload without one, refuses to
mix manifests of different library versions in the same comparison, and
exposes ``manifest diff`` (M2) to explain every score-affecting delta between
two runs.

Field types match design §8.1 / §16.4 exactly. Adding fields, changing
types, or renaming fields is a breaking change in semver terms (§16.9).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from anvil.exceptions import ManifestError


class ModelInfo(BaseModel):
    """Identifies the model used in a run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    revision: str
    dtype: str
    quantization: str | None = None
    config_hash: str
    architecture: str


class TaskInfo(BaseModel):
    """Identifies a task and how it was run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    tier: Literal["curated", "imported", "custom"]
    version: str
    dataset_revision: str
    n_fewshot: int = 0
    metric: str
    request_type: Literal["Generate", "LogLikelihood", "Embed", "Classify", "Custom"] = "Generate"


class CaaSAction(BaseModel):
    """One immutable record from the CaaS audit log (design §7.9)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    ts: str
    step: int
    trigger: str
    match_source: Literal["rule_engine", "kb", "llm"]
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    confidence: float = 1.0
    kb_entry_ids: list[str] = Field(default_factory=list)
    previous_value: Any | None = None
    validator_result: Literal["pass", "fail", "timeout", "skipped"] = "skipped"
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0


class Manifest(BaseModel):
    """The reproducibility manifest (design §8).

    The :attr:`manifest_signature` field is *excluded* from the canonical JSON
    representation when computing or verifying the signature; everything else
    is included. See :mod:`anvil.manifest.canonical`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    anvil_version: str
    engine: dict[str, Any]
    model: ModelInfo
    tokenization: dict[str, Any]
    chat_template: dict[str, Any] | None = None
    sampler: dict[str, Any] | None = None
    tasks: list[TaskInfo]
    scores: dict[str, dict[str, float]]
    smoke_test: dict[str, Any] = Field(default_factory=dict)
    caas_log: list[CaaSAction] = Field(default_factory=list)
    hardware: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    ended_at: str
    manifest_signature: str = ""

    # ------------------------------------------------------------ canonical
    def canonical_json(self) -> str:
        """The signed/verified form. Defers to :mod:`anvil.manifest.canonical`."""
        from anvil.manifest.canonical import canonical_json

        return canonical_json(self)

    # ----------------------------------------------------------- sign/verify
    def sign(self) -> Self:
        """Return a copy of this manifest with :attr:`manifest_signature` set.

        Thin facade over :func:`anvil.manifest.sign.sign`; idempotent.
        """
        from anvil.manifest.sign import sign as _sign

        return _sign(self)  # type: ignore[return-value]

    def verify(self) -> bool:
        """Recompute the signature and check it matches.

        Thin facade over :func:`anvil.manifest.verify.verify`. Returns False
        for unsigned manifests; use :func:`anvil.manifest.verify.verify_or_raise`
        to fail loud.
        """
        from anvil.manifest.verify import verify as _verify

        return _verify(self)

    # ------------------------------------------------------------------ diff
    @classmethod
    def diff(cls, a: Manifest, b: Manifest) -> dict[str, Any]:
        """Flat ``{path: (a_value, b_value)}`` diff. Severity-tagged form
        is :func:`anvil.manifest.diff.diff_entries`."""
        from anvil.manifest.diff import diff as _diff

        return _diff(a, b)

    # ------------------------------------------------------------------ I/O
    def save(self, path: str | Path) -> Path:
        """Write the manifest to ``path`` as canonical JSON. Auto-signs first.

        The on-disk JSON includes ``manifest_signature`` (so a downstream
        :meth:`load` + :meth:`verify` can recheck it); the signature itself
        is computed over :meth:`canonical_json`, which excludes it.

        Returns the resolved ``Path`` written. The parent directory is created
        if it does not exist.
        """
        from anvil.manifest.canonical import serialize_with_signature

        signed = self.sign() if not self.manifest_signature else self
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(serialize_with_signature(signed), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        """Load and parse a manifest from disk.

        Raises:
            ManifestError: if the file is missing, malformed, or contains
                fields the schema does not recognize.
        """
        p = Path(path)
        if not p.exists():
            raise ManifestError(f"manifest not found: {p}")
        try:
            return cls.model_validate_json(p.read_text(encoding="utf-8"))
        except (ValueError, TypeError) as exc:
            raise ManifestError(f"invalid manifest at {p}: {exc}") from exc


def manifest_from_canonical_json(text: str) -> Manifest:
    """Parse canonical JSON into a Manifest, with strict error wrapping."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    return Manifest.model_validate(data)


__all__ = [
    "Manifest",
    "ModelInfo",
    "TaskInfo",
    "CaaSAction",
    "manifest_from_canonical_json",
]
