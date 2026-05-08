"""Tests for request and response dataclasses (design §4)."""

from __future__ import annotations

import pytest

from anvil import Classify, Custom, Embed, Generate, LogLikelihood
from anvil.primitives.response import Generation


class TestGenerate:
    def test_messages_xor_prompt(self) -> None:
        # Both set: rejected.
        with pytest.raises(ValueError, match="exactly one"):
            Generate(messages=({"role": "user", "content": "hi"},), prompt="hi")

    def test_neither_messages_nor_prompt(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Generate()

    def test_messages_only(self) -> None:
        g = Generate(messages=({"role": "user", "content": "hi"},))
        assert g.messages is not None
        assert g.prompt is None

    def test_prompt_only(self) -> None:
        g = Generate(prompt="The capital of France is")
        assert g.prompt == "The capital of France is"
        assert g.messages is None


class TestClassify:
    def test_label_set_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Classify(input="x", label_set=())

    def test_label_set_unique(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            Classify(input="x", label_set=("a", "a"))


def test_loglikelihood_is_simple_pair() -> None:
    ll = LogLikelihood(context="The capital of France is", continuation=" Paris")
    assert ll.context.endswith("is")
    assert ll.continuation == " Paris"


def test_embed_defaults() -> None:
    e = Embed(input="ACUGACU")
    assert e.layer == -1
    assert e.pool == "mean"


def test_custom_holds_arbitrary_callable() -> None:
    def fn(xs: list[int]) -> list[int]:
        return [x * 2 for x in xs]

    c = Custom(fn=fn, inputs=(1, 2, 3))
    assert c.fn([1, 2, 3]) == [2, 4, 6]


def test_generation_default_construction() -> None:
    g = Generation(text="Paris.")
    assert g.text == "Paris."
    assert g.token_ids == ()
    assert g.finish_reason == "stop"
