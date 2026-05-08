"""Tests for ``anvil.metrics``."""

from __future__ import annotations

import pytest

from anvil.metrics.exact_match import exact_match, exact_match_aggregate
from anvil.metrics.pass_at_k import pass_at_k, pass_at_k_aggregate


class TestExactMatch:
    def test_strict_equality(self) -> None:
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self) -> None:
        assert exact_match("PARIS", "paris") == 1.0

    def test_whitespace_tolerant(self) -> None:
        assert exact_match("  Paris  ", "Paris\n") == 1.0

    def test_mismatch(self) -> None:
        assert exact_match("London", "Paris") == 0.0

    def test_aggregate_on_empty(self) -> None:
        assert exact_match_aggregate([], []) == {"accuracy": 0.0}

    def test_aggregate_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            exact_match_aggregate(["a"], [])


class TestPassAtK:
    def test_three_of_ten_at_one(self) -> None:
        # Standard estimator: 3/10 == 0.3.
        assert pass_at_k(n=10, c=3, k=1) == pytest.approx(0.3)

    def test_zero_correct(self) -> None:
        assert pass_at_k(n=10, c=0, k=5) == 0.0

    def test_all_correct(self) -> None:
        assert pass_at_k(n=10, c=10, k=5) == 1.0

    def test_k_greater_than_n_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must be"):
            pass_at_k(n=3, c=1, k=5)

    def test_aggregate(self) -> None:
        # Two problems: one with 1/3 success, one with 3/3.
        out = pass_at_k_aggregate([[True, False, False], [True, True, True]], k=1)
        assert out["pass@1"] == pytest.approx((1 / 3 + 1.0) / 2)
