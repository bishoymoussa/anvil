"""Response types — what the engine returns for each kind of request (design §4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class Generation:
    """The output of a :class:`Generate` request.

    Attributes:
        text: decoded text. For chat requests this is the assistant turn only.
        token_ids: generated token ids (excluding the prompt).
        logprobs: per-token logprobs of the chosen token, if requested.
        top_logprobs: per-token list of (token_id, logprob) pairs of the top-k
            alternatives at each step, if requested.
        finish_reason: ``"stop"``, ``"length"``, ``"eos"``, or engine-specific.
        prompt_token_count: input length (text + vision tokens for VLMs).
        image_token_counts: for VLM requests, number of vision tokens per image.
        hidden_states: per-layer captures keyed by layer index, if a
            :class:`HiddenStateSpec` was requested.
        hidden_states_image_spans: for VLM captures, ``(start, end)`` ranges
            inside the captured tensors that correspond to each image.
    """

    text: str
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float, ...] = ()
    top_logprobs: tuple[tuple[tuple[int, float], ...], ...] = ()
    finish_reason: str = "stop"
    prompt_token_count: int = 0
    image_token_counts: tuple[int, ...] = ()
    hidden_states: dict[int, torch.Tensor] = field(default_factory=dict)
    hidden_states_image_spans: tuple[tuple[int, int], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbedResult:
    """The output of an :class:`Embed` request."""

    embedding: torch.Tensor
    layer: int
    pool: str
    input_token_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ClassifyResult:
    """The output of a :class:`Classify` request."""

    label: str
    label_logprobs: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Response:
    """Top-level response from a multi-request batched call.

    ``outputs`` is parallel to the input request list; index N of ``outputs``
    is the result for index N of the inputs. The element type depends on
    request type: :class:`Generation`, :class:`EmbedResult`,
    :class:`ClassifyResult`, ``tuple[float, bool]`` for log-likelihood, or
    arbitrary for :class:`Custom`.
    """

    outputs: tuple[Any, ...]
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["Response", "Generation", "EmbedResult", "ClassifyResult"]
