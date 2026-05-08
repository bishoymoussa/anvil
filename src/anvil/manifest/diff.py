"""Manifest diffing (design §8.3).

The score-deltas-explained tool: given two manifests, return a structured
description of every field that differs and could plausibly explain a
difference in scores. Timestamps and the signature are excluded by design
(they always vary across runs and don't affect output).

Field categorization
--------------------

Each entry in the diff is annotated with a ``severity``:

* ``critical``: the field directly affects model output. Examples: model
  revision, sampler hash, chat-template hash, tokenization hash.
* ``probable``: changes here often shift scores. Examples: anvil version,
  task version, dataset revision.
* ``benign``: never affects scores. Hardware info, engine backend hash.

A reviewer reading a 12-point score gap should look at ``critical`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from anvil.manifest.schema import Manifest

Severity = Literal["critical", "probable", "benign"]

_IGNORED_PATHS: tuple[str, ...] = ("manifest_signature", "started_at", "ended_at")

_CRITICAL_PREFIXES: tuple[str, ...] = (
    "model.",
    "sampler.",
    "chat_template.",
    "tokenization.",
)
_PROBABLE_PREFIXES: tuple[str, ...] = (
    "anvil_version",
    "tasks.",
    "scores.",
    "caas_log",
)


@dataclass(frozen=True)
class DiffEntry:
    """One field that differs between two manifests."""

    path: str
    a: Any
    b: Any
    severity: Severity

    def render(self) -> str:
        return f"[{self.severity:>8}] {self.path}: {self.a!r} → {self.b!r}"


def diff(a: Manifest, b: Manifest) -> dict[str, Any]:
    """Return a flat ``{path: (a_value, b_value)}`` of every differing field.

    This is the simple shape callers expect; for severity-annotated output
    use :func:`diff_entries`.
    """
    return _flat_diff(a.model_dump(mode="json"), b.model_dump(mode="json"))


def diff_entries(a: Manifest, b: Manifest) -> list[DiffEntry]:
    """Return a sorted list of :class:`DiffEntry`, severity-tagged.

    Entries are sorted by severity (critical → probable → benign) then by
    ``path``. The CLI's ``anvil manifest diff`` formats this list directly.
    """
    flat = _flat_diff(a.model_dump(mode="json"), b.model_dump(mode="json"))
    entries = [DiffEntry(path=p, a=av, b=bv, severity=_classify(p)) for p, (av, bv) in flat.items()]
    severity_order: dict[Severity, int] = {"critical": 0, "probable": 1, "benign": 2}
    entries.sort(key=lambda e: (severity_order[e.severity], e.path))
    return entries


def _flat_diff(a: Any, b: Any, *, prefix: str = "") -> dict[str, tuple[Any, Any]]:
    """Recursive structural diff. Internal — exposed via :func:`diff`."""
    out: dict[str, tuple[Any, Any]] = {}
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            full = f"{prefix}.{k}" if prefix else k
            if any(full == ig or full.startswith(f"{ig}.") for ig in _IGNORED_PATHS):
                continue
            out.update(_flat_diff(a.get(k), b.get(k), prefix=full))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out[prefix] = (a, b)
            return out
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.update(_flat_diff(x, y, prefix=f"{prefix}[{i}]"))
        return out
    if a != b:
        out[prefix] = (a, b)
    return out


def _classify(path: str) -> Severity:
    if any(path.startswith(p) for p in _CRITICAL_PREFIXES):
        return "critical"
    if any(path.startswith(p) for p in _PROBABLE_PREFIXES):
        return "probable"
    return "benign"


__all__ = ["diff", "diff_entries", "DiffEntry", "Severity"]
