"""pass@k — the HumanEval-style code metric.

Implements the unbiased estimator from Chen et al. 2021:

    pass@k = 1 - C(n - c, k) / C(n, k)

where ``n`` is the number of samples per problem and ``c`` is the number of
correct samples among them. When ``c == 0`` we return 0.0 directly to avoid
``C(n-c, k)`` overflowing or dividing by zero for problems with no successes.
"""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Args:
        n: total samples per problem (must be ≥ ``k``).
        c: correct samples among the ``n``.
        k: cutoff.

    Example:
        >>> pass_at_k(n=10, c=3, k=1) == 0.3
        True
        >>> pass_at_k(n=10, c=0, k=5)
        0.0
    """
    if k < 1:
        raise ValueError(f"k must be ≥ 1, got {k}")
    if n < k:
        raise ValueError(f"n must be ≥ k; got n={n}, k={k}")
    if c < 0 or c > n:
        raise ValueError(f"c must be in [0, n]; got c={c}, n={n}")
    if c == 0:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_aggregate(
    correctness_per_problem: Sequence[Sequence[bool]],
    *,
    k: int,
) -> dict[str, float]:
    """Average pass@k over a list-of-list of per-sample correctness flags.

    Each entry of ``correctness_per_problem`` is the booleans for one problem
    (length = n samples per problem). Returns ``{"pass@{k}": float}``.
    """
    if not correctness_per_problem:
        return {f"pass@{k}": 0.0}
    n_each = [len(samples) for samples in correctness_per_problem]
    n = min(n_each)
    if n < k:
        raise ValueError(f"shortest problem has {n} samples, need ≥ {k} for pass@{k}")
    scores = [
        pass_at_k(n=n, c=sum(1 for s in samples[:n] if s), k=k)
        for samples in correctness_per_problem
    ]
    return {f"pass@{k}": sum(scores) / len(scores)}


__all__ = ["pass_at_k", "pass_at_k_aggregate"]
