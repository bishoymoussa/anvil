"""Tests for MultiTurnFewshot mixin and MMLUMultiTurn."""

from __future__ import annotations

from typing import Any

from anvil.primitives.request import LogLikelihood
from anvil.tasks.base import MultiTurnFewshot  # noqa: F401 — used via _ToyMultiTurn
from anvil.tasks.builtin.mmlu import MMLUMultiTurn

_EXEMPLARS = [
    {"question": "Q1?", "choices": ["a", "b", "c", "d"], "answer": 0, "subject": "math"},
    {"question": "Q2?", "choices": ["e", "f", "g", "h"], "answer": 1, "subject": "math"},
]

_TEST_DOC = {
    "question": "What is 2+2?",
    "choices": ["3", "4", "5", "6"],
    "answer": 1,
    "subject": "math",
}


class _ToyMultiTurn(MultiTurnFewshot):
    """Minimal MultiTurnFewshot for unit-testing the mixin."""

    name = "_toy_multiturn"
    dataset = "fake/ds"

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        return f"Q: {doc['q']}\nAnswer:"

    def doc_to_target(self, doc: dict[str, Any]) -> int:
        return "ABCD".index(doc["gold"])

    def exemplar_to_answer(self, doc: dict[str, Any]) -> str:
        return doc["gold"]

    def doc_to_exemplars(self) -> list[dict[str, Any]]:
        return [
            {"q": "What is 1+1?", "gold": "B"},
            {"q": "What is 2+2?", "gold": "C"},
        ]


class TestMultiTurnFewshotMixin:
    def test_chat_templated_is_forced_true(self) -> None:
        assert _ToyMultiTurn.chat_templated is True

    def test_doc_to_messages_interleaves_exemplars(self) -> None:
        task = _ToyMultiTurn(n_fewshot=2)
        messages = task.doc_to_messages({"q": "Final?", "gold": "A"})
        # 2 exemplars × 2 roles + 1 final user = 5 messages
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "B"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "C"
        assert messages[4]["role"] == "user"
        assert "Final?" in messages[4]["content"]

    def test_doc_to_messages_zero_shot_is_single_user(self) -> None:
        task = _ToyMultiTurn(n_fewshot=0)
        messages = task.doc_to_messages({"q": "X?", "gold": "A"})
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_doc_to_request_returns_loglikelihood_with_messages(self) -> None:
        task = _ToyMultiTurn(n_fewshot=1)
        reqs = task.doc_to_request({"q": "Test?", "gold": "A"})
        assert isinstance(reqs, list) and len(reqs) == 4
        for r in reqs:
            assert isinstance(r, LogLikelihood)
            assert r.messages is not None
            assert r.context == ""
            assert r.chat_templated is True

    def test_n_fewshot_caps_exemplars(self) -> None:
        task = _ToyMultiTurn(n_fewshot=1)
        messages = task.doc_to_messages({"q": "X?", "gold": "D"})
        # 1 exemplar × 2 roles + 1 final user = 3
        assert len(messages) == 3

    def test_request_to_prediction_picks_argmax(self) -> None:
        task = _ToyMultiTurn(n_fewshot=0)
        responses = [(-2.0, False), (-0.5, False), (-1.5, False), (-3.0, False)]
        assert task.request_to_prediction(responses, {}) == 1


class TestMMLUMultiTurn:
    def test_name(self) -> None:
        assert MMLUMultiTurn.name == "mmlu_multiturn"

    def test_chat_templated_true(self) -> None:
        assert MMLUMultiTurn.chat_templated is True

    def test_doc_to_choices_no_leading_space(self) -> None:
        task = MMLUMultiTurn(n_fewshot=0)
        assert task.doc_to_choices({}) == ["A", "B", "C", "D"]

    def test_doc_to_messages_zero_shot(self) -> None:
        task = MMLUMultiTurn(n_fewshot=0)
        messages = task.doc_to_messages(_TEST_DOC)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "What is 2+2?" in messages[0]["content"]

    def test_exemplar_to_answer(self) -> None:
        task = MMLUMultiTurn(n_fewshot=0)
        assert task.exemplar_to_answer({"answer": 0}) == "A"
        assert task.exemplar_to_answer({"answer": 3}) == "D"

    def test_doc_to_request_returns_loglikelihood_with_messages(self) -> None:
        task = MMLUMultiTurn(n_fewshot=0)
        reqs = task.doc_to_request(_TEST_DOC)
        assert len(reqs) == 4
        for r in reqs:
            assert isinstance(r, LogLikelihood)
            assert r.messages is not None
            assert r.continuation in ("A", "B", "C", "D")

    def test_inherits_mmlu_doc_to_target(self) -> None:
        task = MMLUMultiTurn(n_fewshot=0)
        assert task.doc_to_target({"answer": 2}) == 2


class TestLogLikelihoodMessagesField:
    def test_messages_and_context_mutually_exclusive(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="context or messages"):
            LogLikelihood(context="ctx", messages=({"role": "user", "content": "hi"},))

    def test_messages_only_is_valid(self) -> None:
        r = LogLikelihood(messages=({"role": "user", "content": "hi"},), continuation=" A")
        assert r.messages is not None
        assert r.context == ""

    def test_context_only_is_valid(self) -> None:
        r = LogLikelihood(context="some prompt", continuation=" A")
        assert r.messages is None
