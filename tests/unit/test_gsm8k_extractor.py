"""Tests for GSM8K's strict-match number extractor.

The extractor is the part most likely to silently regress (lm-eval-harness
issues #2278/#3214: "flexible-extract picks the first number" — design
§1.3). We test:

* ``####`` answer line is preferred over earlier numbers.
* Comma thousand-separators are stripped.
* Trailing zeros after a decimal are normalized.
* No-number cases return the empty string (so aggregate scores 0).
"""

from __future__ import annotations

from anvil.tasks.builtin.gsm8k import _extract_number


class TestExtractor:
    def test_hash_answer_preferred(self) -> None:
        text = "Step 1: We compute 2+2 = 4. Step 2: 4*3 = 12.\n#### 12"
        assert _extract_number(text) == "12"

    def test_falls_back_to_last_number(self) -> None:
        text = "We have 3 apples and 4 oranges, total 7."
        assert _extract_number(text) == "7"

    def test_comma_separators_stripped(self) -> None:
        assert _extract_number("#### 1,234,567") == "1234567"

    def test_decimal_zero_normalized(self) -> None:
        assert _extract_number("#### 5.0") == "5"

    def test_negative_numbers_kept(self) -> None:
        assert _extract_number("#### -42") == "-42"

    def test_no_number_returns_empty(self) -> None:
        assert _extract_number("I do not know the answer.") == ""

    def test_intermediate_numbers_dont_win_with_hash_marker(self) -> None:
        text = "First I think 99. Then I revise to 100.\n#### 100"
        assert _extract_number(text) == "100"
