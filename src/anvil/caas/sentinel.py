"""Quality sentinel (design §6.6, §7.2).

The sentinel is a fixed-answer prompt whose right answer is a deterministic
short string. If the model can't produce the expected output, something
silent is wrong (chat template missing, EOS misconfigured, base-model
degradation) — and CaaS engages even if the engine technically didn't
crash.

For v0 we ship one universal sentinel ("the capital of France") plus
per-task overrides via :attr:`Task.sentinel_prompt` /
:attr:`Task.sentinel_expected`.
"""

from __future__ import annotations

from dataclasses import dataclass

from anvil.logging import get_logger
from anvil.primitives.request import Generate
from anvil.primitives.response import Generation
from anvil.primitives.sampler import Sampler

_log = get_logger(__name__)


# Conservative universal sentinel — short answer, low ambiguity.
DEFAULT_SENTINEL_PROMPT = "What is the capital of France? Answer in one word."
DEFAULT_SENTINEL_EXPECTED = "paris"


@dataclass
class SentinelResult:
    """Outcome of running the sentinel against the engine."""

    passed: bool
    actual: str
    expected: str
    reason: str = ""

    def __bool__(self) -> bool:
        return self.passed


def run_sentinel(
    engine: object,
    *,
    prompt: str | None = None,
    expected: str | None = None,
    max_tokens: int = 32,
) -> SentinelResult:
    """Run the sentinel against ``engine``. Returns a :class:`SentinelResult`.

    The check is intentionally lenient: we lowercase + strip + check that
    the expected substring appears in the actual output. The sentinel is a
    *signal*, not a metric — false positives are way worse than false
    negatives, since a false-positive blocks a legitimate run.
    """
    p = prompt or DEFAULT_SENTINEL_PROMPT
    expected_norm = (expected or DEFAULT_SENTINEL_EXPECTED).strip().lower()

    req = Generate(
        messages=({"role": "user", "content": p},),
        sampler=Sampler.greedy(max_tokens=max_tokens),
    )
    try:
        outputs = engine.generate_logprobs([req])  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        return SentinelResult(
            passed=False,
            actual="",
            expected=expected_norm,
            reason=f"engine raised: {exc}",
        )

    if not outputs or not isinstance(outputs[0], Generation):
        return SentinelResult(
            passed=False,
            actual="",
            expected=expected_norm,
            reason="engine returned no output",
        )

    actual = outputs[0].text.strip().lower()
    passed = expected_norm in actual
    if not passed:
        _log.warning("quality sentinel failed: expected %r in %r", expected_norm, actual)
    return SentinelResult(
        passed=passed,
        actual=actual,
        expected=expected_norm,
        reason="" if passed else "expected substring not found",
    )


__all__ = [
    "SentinelResult",
    "run_sentinel",
    "DEFAULT_SENTINEL_PROMPT",
    "DEFAULT_SENTINEL_EXPECTED",
]
