"""Exact-match metric.

Two flavors:

* :func:`exact_match` — returns 1.0 if normalized prediction == normalized
  target else 0.0. Intended for use as a per-doc metric whose results are
  averaged.
* :func:`exact_match_aggregate` — operates over parallel lists, returns
  ``{"accuracy": float}``.

The "normalization" is conservative: leading/trailing whitespace and ASCII
case-folding only. Anything more aggressive (boxed-answer parsing,
math-expression equivalence, etc.) belongs in a task-specific extractor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _normalize(s: str) -> str:
    return s.strip().casefold()


def exact_match(prediction: str, target: str) -> float:
    """1.0 if ``prediction`` and ``target`` agree after whitespace/case folding.

    Example:
        >>> exact_match("Paris", "  paris ")
        1.0
        >>> exact_match("London", "Paris")
        0.0
    """
    return 1.0 if _normalize(prediction) == _normalize(target) else 0.0


def exact_match_aggregate(predictions: Sequence[str], targets: Sequence[str]) -> dict[str, float]:
    """Aggregate exact-match accuracy over parallel sequences."""
    if len(predictions) != len(targets):
        raise ValueError(f"length mismatch: predictions={len(predictions)} targets={len(targets)}")
    if not predictions:
        return {"accuracy": 0.0}
    correct = sum(exact_match(p, t) for p, t in zip(predictions, targets, strict=True))
    return {"accuracy": correct / len(predictions)}


__all__ = ["exact_match", "exact_match_aggregate"]
