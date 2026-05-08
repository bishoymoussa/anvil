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

import hashlib
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

        Uses sha256 over the canonical JSON form (excluding the signature
        field). Idempotent: signing an already-signed manifest produces the
        same signature.
        """
        encoded = self.canonical_json().encode("utf-8")
        sig = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return self.model_copy(update={"manifest_signature": sig})

    def verify(self) -> bool:
        """Recompute the signature and check it matches :attr:`manifest_signature`."""
        if not self.manifest_signature:
            return False
        encoded = self.canonical_json().encode("utf-8")
        expected = "sha256:" + hashlib.sha256(encoded).hexdigest()
        return expected == self.manifest_signature

    # ------------------------------------------------------------------ diff
    @classmethod
    def diff(cls, a: Manifest, b: Manifest) -> dict[str, Any]:
        """Recursive diff that flags every field which could explain a score delta.

        Lands as a fully-fledged tool in M2; for now this is a usable but
        unoptimized recursive comparison that returns ``{path: (a_value, b_value)}``.
        """
        return _diff_dicts(
            a.model_dump(mode="json"),
            b.model_dump(mode="json"),
            ignore=("manifest_signature", "started_at", "ended_at"),
        )

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


def _diff_dicts(a: Any, b: Any, *, path: str = "", ignore: tuple[str, ...] = ()) -> dict[str, Any]:
    """Recursive structural diff used by :meth:`Manifest.diff`."""
    out: dict[str, Any] = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k in ignore:
                continue
            sub_path = f"{path}.{k}" if path else k
            inner = _diff_dicts(a.get(k), b.get(k), path=sub_path, ignore=ignore)
            out.update(inner)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out[path] = (a, b)
            return out
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            inner = _diff_dicts(x, y, path=f"{path}[{i}]", ignore=ignore)
            out.update(inner)
        return out
    if a != b:
        out[path] = (a, b)
    return out


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
