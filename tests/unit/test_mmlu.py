"""Tests for the MMLU task and the multiple-choice machinery."""

from __future__ import annotations

from typing import Any

import pytest

from anvil.primitives.request import LogLikelihood
from anvil.tasks.base import MultipleChoice
from anvil.tasks.builtin.mmlu import MMLU


class _ToyMCQ(MultipleChoice):
    """A 4-way MCQ where the gold letter is encoded in the doc."""

    name = "_toy_mcq"
    dataset = "fake/ds"

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        return f"Q: {doc['q']}\nAnswer: {doc['gold']}"

    def doc_to_choices(self, doc: dict[str, Any]) -> list[str]:
        del doc
        return [" A", " B", " C", " D"]

    def doc_to_target(self, doc: dict[str, Any]) -> int:
        return "ABCD".index(doc["gold"])


class TestMultipleChoiceMachinery:
    def test_doc_to_request_returns_one_per_choice(self) -> None:
        task = _ToyMCQ()
        reqs = task.doc_to_request({"q": "what?", "gold": "B"})
        assert isinstance(reqs, list)
        assert len(reqs) == 4
        assert all(isinstance(r, LogLikelihood) for r in reqs)
        assert [r.continuation for r in reqs] == [" A", " B", " C", " D"]

    def test_request_to_prediction_picks_argmax(self) -> None:
        task = _ToyMCQ()
        # Highest logprob at index 2 (option C).
        responses = [(-1.0, False), (-1.5, False), (-0.5, False), (-2.0, False)]
        assert task.request_to_prediction(responses, {}) == 2

    def test_aggregate_uses_doc_to_target(self) -> None:
        task = _ToyMCQ()
        docs = [{"q": "1", "gold": "A"}, {"q": "2", "gold": "B"}, {"q": "3", "gold": "C"}]
        # Predict A, A, C — 2/3 correct.
        preds = [0, 0, 2]
        out = task.aggregate(preds, docs)
        assert out == {"accuracy": pytest.approx(2 / 3)}

    def test_request_type_is_loglikelihood(self) -> None:
        task = _ToyMCQ()
        assert task.request_type == "LogLikelihood"


class TestMMLUTask:
    def test_doc_to_choices_is_letter_continuations(self) -> None:
        task = MMLU(n_fewshot=0)
        choices = task.doc_to_choices({})
        assert choices == [" A", " B", " C", " D"]

    def test_doc_to_target_is_int_answer(self) -> None:
        task = MMLU(n_fewshot=0)
        assert task.doc_to_target({"answer": 2}) == 2

    def test_zero_shot_text_includes_options_and_answer_cue(self) -> None:
        task = MMLU(n_fewshot=0)
        text = task.doc_to_text(
            {
                "question": "What is 2+2?",
                "choices": ["3", "4", "5", "6"],
            }
        )
        assert "What is 2+2?" in text
        assert "A. 3" in text
        assert "B. 4" in text
        assert text.endswith("Answer:")

    def test_n_fewshot_default_is_5(self) -> None:
        assert MMLU.n_fewshot_default == 5

    def test_metric_name_is_accuracy(self) -> None:
        assert MMLU.metric_name == "accuracy"

    def test_request_type_is_loglikelihood(self) -> None:
        assert MMLU.request_type == "LogLikelihood"
