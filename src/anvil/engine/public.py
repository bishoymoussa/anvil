"""Engine protocol — what every backend must implement (design §3.1, §16.4).

The engine is a thin facade over a real backend (HF transformers in M0,
vLLM in M1+). The wrapper layer (``anvil.engine._wrappers``) reimplements
abstractions that vLLM V1 dropped: per-request logits processors and
hidden-state extraction. By keeping that logic *above* the engine, Anvil is
robust to engine churn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from anvil.primitives.request import Classify, Embed, Generate, LogLikelihood
    from anvil.primitives.response import (
        ClassifyResult,
        EmbedResult,
        Generation,
    )


class Engine(Protocol):
    """The contract every engine backend honors.

    Each method is *batched*: implementations should drive the underlying
    engine at full throughput rather than running a per-doc Python loop.
    """

    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]:
        """For each (context, continuation): ``(logprob, is_greedy)``.

        ``is_greedy`` is True if the continuation is the argmax under the
        model — a useful signal for multiple-choice scoring.
        """
        ...

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        """Total log-likelihood of each string under the model."""
        ...

    def generate_until(self, requests: list[tuple[str, list[str]]]) -> list[Generation]:
        """Generate until any of ``until_strings`` appears.

        Stop strings are detected via *streaming logprob extraction* — design
        §6.2: no more "model rambled past the answer" silent failures.
        """
        ...

    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]:
        """Open-ended generation, returning per-token top-k logprobs."""
        ...

    def embed(self, requests: list[Embed]) -> list[EmbedResult]:
        """Pooled hidden representations for arbitrary inputs."""
        ...

    def classify(self, requests: list[Classify]) -> list[ClassifyResult]:
        """Score each input against its label set."""
        ...

    def custom(self, fn: Callable[[list[Any]], list[Any]], inputs: list[Any]) -> list[Any]:
        """Universal escape hatch (design §6.2): user callable, batched."""
        ...

    @property
    def model_info(self) -> dict[str, Any]:
        """Identifying info for the manifest's ``model`` field."""
        ...

    @property
    def backend_hash(self) -> str:
        """sha256 of ``(engine name + version + key compile flags)``."""
        ...

    def shutdown(self) -> None:
        """Release model weights, KV caches, and any background threads."""
        ...


__all__ = ["Engine"]
