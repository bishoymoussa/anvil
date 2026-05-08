"""Request types — the union over what an engine can be asked to do (design §4 / §6.7).

Five kinds:

* :class:`Generate` — text or multimodal chat completion.
* :class:`LogLikelihood` — score (context, continuation) pairs (the
  lm-evaluation-harness primitive for multiple-choice tasks).
* :class:`Embed` — produce a pooled hidden representation. Universal across
  modalities (text, RNA, audio, …).
* :class:`Classify` — score a fixed label set against an input.
* :class:`Custom` — arbitrary callable that operates on a batch. The escape
  hatch for non-text modalities (design §6.7).

These are intentionally thin: a request carries inputs and per-call hooks
(``Sampler``, ``LogitsProcessor``, ``HiddenStateSpec``); everything else
(model weights, chat template, tokenizer) belongs to the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union

if TYPE_CHECKING:
    from collections.abc import Callable

    from .hidden_state_spec import HiddenStateSpec
    from .logits_processor import LogitsProcessor
    from .sampler import Sampler


@dataclass(frozen=True, slots=True)
class Generate:
    """Open-ended generation. Either ``messages`` (chat) or ``prompt`` (raw)."""

    messages: tuple[dict[str, Any], ...] | None = None
    prompt: str | None = None
    sampler: Sampler | None = None
    logits_processors: tuple[LogitsProcessor, ...] = ()
    capture: HiddenStateSpec | None = None

    def __post_init__(self) -> None:
        if (self.messages is None) == (self.prompt is None):
            raise ValueError("Generate: provide exactly one of messages or prompt")


@dataclass(frozen=True, slots=True)
class LogLikelihood:
    """Score a single (context, continuation) pair.

    The engine batches many of these and extracts logprobs at the right offsets
    in one prefill pass — see ``engine.loglikelihood``.
    """

    context: str
    continuation: str


PoolStrategy = Literal["mean", "cls", "last", "max", "none"]


@dataclass(frozen=True, slots=True)
class Embed:
    """Produce an embedding from any input the model accepts.

    ``input`` is intentionally typed ``Any`` — for text models it is ``str``,
    for an RNA model it is a sequence string, for an audio model it is a
    waveform/array, etc. (design §6.7).
    """

    input: Any
    layer: int = -1
    pool: PoolStrategy = "mean"


@dataclass(frozen=True, slots=True)
class Classify:
    """Score a fixed set of labels against an input."""

    input: Any
    label_set: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label_set:
            raise ValueError("Classify.label_set must be non-empty")
        if len(set(self.label_set)) != len(self.label_set):
            raise ValueError("Classify.label_set must contain unique labels")


@dataclass(frozen=True, slots=True)
class Custom:
    """Universal escape hatch — any callable that operates on a batch.

    Used for modalities that don't fit ``Generate``/``Embed``/``Classify``
    (graphs, multi-input fusion models, custom protocols). The callable's
    source is hashed into the manifest so reruns are reproducible if it is
    deterministic.
    """

    fn: Callable[[list[Any]], list[Any]]
    inputs: tuple[Any, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


Request = Union[Generate, LogLikelihood, Embed, Classify, Custom]
"""Union of all engine-acceptable request types."""


__all__ = [
    "Generate",
    "LogLikelihood",
    "Embed",
    "Classify",
    "Custom",
    "Request",
    "PoolStrategy",
]
