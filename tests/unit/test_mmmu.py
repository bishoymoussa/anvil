"""Tests for the MMMU task (design §6.5)."""

from __future__ import annotations

from typing import Any

from anvil.primitives.request import Generate
from anvil.primitives.response import Generation
from anvil.tasks.builtin.mmmu import (
    MMMU,
    _collect_images,
    _extract_letter,
    _parse_options,
)


class TestParseOptions:
    def test_stringified_list(self) -> None:
        assert _parse_options("['Apple', 'Banana', 'Cherry', 'Durian']") == [
            "Apple",
            "Banana",
            "Cherry",
            "Durian",
        ]

    def test_already_a_list(self) -> None:
        assert _parse_options(["A", "B", "C"]) == ["A", "B", "C"]

    def test_malformed_string(self) -> None:
        assert _parse_options("not a list") == []

    def test_none(self) -> None:
        assert _parse_options(None) == []


class TestExtractLetter:
    def test_first_capital_letter_wins(self) -> None:
        assert _extract_letter("The answer is C.") == "C"

    def test_lowercase_not_matched(self) -> None:
        assert _extract_letter("the answer is c") is None

    def test_no_letter(self) -> None:
        assert _extract_letter("¯\\_(ツ)_/¯") is None

    def test_multi_letter_picks_first(self) -> None:
        # The model often emits "C. Cherry" — we pick C.
        assert _extract_letter("C. Cherry, because Y") == "C"


class TestCollectImages:
    def test_collects_in_order(self) -> None:
        doc = {
            "image_1": "img-a",
            "image_2": "img-b",
            "image_3": None,
            "image_4": "img-d",
        }
        assert _collect_images(doc) == ["img-a", "img-b", "img-d"]

    def test_no_images_returns_empty(self) -> None:
        assert _collect_images({}) == []


class TestMMMUTask:
    def test_task_metadata(self) -> None:
        assert MMMU.name == "mmmu"
        assert MMMU.dataset == "MMMU/MMMU"
        assert MMMU.metric_name == "accuracy"
        assert MMMU.request_type == "Generate"

    def test_doc_to_request_emits_multimodal_messages(self) -> None:
        from PIL import Image

        task = MMMU()
        doc: dict[str, Any] = {
            "question": "What is shown in the image?",
            "options": "['Cat', 'Dog', 'Bird', 'Fish']",
            "image_1": Image.new("RGB", (64, 64)),
            "answer": "A",
        }
        req = task.doc_to_request(doc)
        assert isinstance(req, Generate)
        assert req.messages is not None and len(req.messages) == 2
        # User content is multimodal: image part + text part.
        user_content = req.messages[1]["content"]
        assert isinstance(user_content, list)
        types = [p.get("type") for p in user_content]
        assert "image" in types
        assert "text" in types

    def test_request_to_prediction_extracts_letter(self) -> None:
        task = MMMU()
        gen = Generation(text="The correct answer is B.")
        assert task.request_to_prediction(gen, {}) == "B"

    def test_aggregate_only_scores_letter_questions(self) -> None:
        task = MMMU()
        docs = [
            {"answer": "A"},
            {"answer": "B"},
            {"answer": ""},  # open-ended; skipped
            {"answer": "C"},
        ]
        preds = ["A", "B", "irrelevant", "D"]
        out = task.aggregate(preds, docs)
        # 2 of 3 scored questions are correct (A and B).
        assert out["accuracy"] == 2 / 3
