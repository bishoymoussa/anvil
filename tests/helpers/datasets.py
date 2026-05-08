"""Programmatic dataset fixtures for offline tests.

Each row has the same shape as a real GSM8K row: ``{"question": str, "answer": str}``.
The questions are constructed so the last number in the question equals
the gold answer (which is what the StubEngine extractor returns), so the
StubEngine scores 100% on these fixtures. That is deliberate — the test
asserts the runner ferries the data correctly, not that a real model
solves the problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


def tiny_gsm8k() -> Iterator[dict[str, str]]:
    """Yield 5 rows whose last-number-in-question equals the gold answer."""
    yield from [
        {
            "question": "If I add 2 and 3 together, the result equals 5.",
            "answer": "Trivially the result is 5.\n#### 5",
        },
        {
            "question": "Twice four is the answer; the answer is 8.",
            "answer": "2 * 4 = 8.\n#### 8",
        },
        {
            "question": "Six minus three plus six equals 9.",
            "answer": "6 - 3 + 6 = 9.\n#### 9",
        },
        {
            "question": "Ten plus eleven equals 21.",
            "answer": "10 + 11 = 21.\n#### 21",
        },
        {
            "question": "Five times five is 25.",
            "answer": "5 * 5 = 25.\n#### 25",
        },
    ]


__all__ = ["tiny_gsm8k"]
