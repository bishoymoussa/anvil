"""A deterministic stub engine for offline integration tests.

Doesn't load any model, doesn't allocate any GPU memory. ``generate_logprobs``
turns each prompt into a synthesized "answer" by extracting the last number
in the prompt and emitting it after ``####`` — the GSM8K extractor will
parse that out and the test scores the synthesized prediction against the
gold answer. Concretely: for prompts whose last number happens to match the
gold answer, the score is 1.0. We use this in :file:`test_milestone_0.py`
with a small, hand-crafted dataset where the last number in each prompt is
the right answer, so the run produces a score in the manifest's accepted
range without needing a real LM.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Any

from anvil.primitives.request import Classify, Embed, Generate, LogLikelihood
from anvil.primitives.response import (
    ClassifyResult,
    EmbedResult,
    Generation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_LAST_NUMBER = re.compile(r"(-?\d+(?:[.,]\d+)*)")


class StubEngine:
    """Implements :class:`anvil.engine.public.Engine` without a real model.

    The synthesizer is deliberately stupid: it copies the last number out of
    the user's question into a ``#### N`` answer line. To produce a score
    in (0.4, 0.95) — the M0 acceptance bound rejects suspiciously-perfect
    runs — every Nth response is intentionally wrong (returns ``0``). The
    miss rate is configurable for tests that need a specific ratio.
    """

    def __init__(self, model_id: str = "stub/test-model", *, miss_every: int = 3) -> None:
        self._model_id = model_id
        self._miss_every = miss_every
        self._counter = 0

    # ------------------------------------------------------------ generate
    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]:
        del top_k
        out: list[Generation] = []
        for req in requests:
            self._counter += 1
            prompt = self._render(req)
            # Miss every Nth request to keep the score below the
            # M0-acceptance upper bound (0.95).
            if self._miss_every > 0 and self._counter % self._miss_every == 0:
                text = "I'm unsure.\n#### 0"
            else:
                number = self._last_number(prompt)
                text = (
                    f"Let's check carefully.\n#### {number}"
                    if number is not None
                    else "I don't know.\n#### 0"
                )
            out.append(Generation(text=text, prompt_token_count=len(prompt.split())))
        return out

    def generate_until(self, requests: list[tuple[str, list[str]]]) -> list[Generation]:
        gens = self.generate_logprobs([Generate(prompt=p) for p, _ in requests])
        return list(gens)

    # ---------------------------------------------------------- placeholders
    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]:
        # Return a constant for each request — the runner doesn't use this in
        # M0. Length-aware so callers can still distinguish requests.
        return [(-len(r.continuation) * 0.1, False) for r in requests]

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        return [-len(r) * 0.1 for r in requests]

    def embed(self, requests: list[Embed]) -> list[EmbedResult]:
        raise NotImplementedError

    def classify(self, requests: list[Classify]) -> list[ClassifyResult]:
        raise NotImplementedError

    def custom(self, fn: Callable[[list[Any]], list[Any]], inputs: list[Any]) -> list[Any]:
        return fn(inputs)

    # ----------------------------------------------------------- bookkeeping
    @property
    def model_info(self) -> dict[str, Any]:
        cfg_hash = "sha256:" + hashlib.sha256(self._model_id.encode()).hexdigest()
        return {
            "id": self._model_id,
            "revision": "main",
            "dtype": "float32",
            "quantization": None,
            "config_hash": cfg_hash,
            "architecture": "StubModel",
        }

    @property
    def backend_hash(self) -> str:
        return "sha256:" + hashlib.sha256(b"stub-engine").hexdigest()

    @property
    def backend_info(self) -> dict[str, str]:
        return {"name": "stub", "version": "0", "backend_hash": self.backend_hash}

    def shutdown(self) -> None:
        return None

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _render(req: Generate) -> str:
        if req.prompt is not None:
            return req.prompt
        assert req.messages is not None
        return "\n".join(str(m.get("content", "")) for m in req.messages)

    @staticmethod
    def _last_number(text: str) -> str | None:
        matches = list(_LAST_NUMBER.finditer(text))
        if not matches:
            return None
        return matches[-1].group(1).replace(",", "")


__all__ = ["StubEngine"]
