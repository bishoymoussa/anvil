"""Tests for the lm-evaluation-harness compatibility shim (design §6.4)."""

from __future__ import annotations

from typing import Any

import pytest

from anvil.exceptions import ConfigError
from anvil.tasks.base import MultipleChoice, Task
from anvil.tasks.lm_eval_shim import compile_yaml_dict, register_lm_eval_task
from anvil.tasks.lm_eval_shim.compiler import (
    UnsupportedYAML,
    _apply_filters,
    _eval_doc_to_target,
    _eval_doc_to_text,
    _normalize_target_to_index,
)
from anvil.tasks.registry import _REGISTRY


def teardown_function() -> None:
    """Drop test-only tasks so the shared registry doesn't leak."""
    for name in list(_REGISTRY):
        if name.startswith("_"):
            _REGISTRY.pop(name, None)


class TestEvalDocToText:
    def test_format_template(self) -> None:
        out = _eval_doc_to_text("Q: {q}\nA:", {"q": "two plus two"})
        assert out == "Q: two plus two\nA:"

    def test_field_name_lookup(self) -> None:
        assert _eval_doc_to_text("question", {"question": "hi", "other": "x"}) == "hi"

    def test_missing_field_raises(self) -> None:
        with pytest.raises(UnsupportedYAML, match="missing field"):
            _eval_doc_to_text("Q: {missing}", {"q": "x"})

    def test_none_falls_back_to_text(self) -> None:
        assert _eval_doc_to_text(None, {"text": "hello"}) == "hello"


class TestEvalDocToTarget:
    def test_field_name(self) -> None:
        assert _eval_doc_to_target("answer", {"answer": "C"}) == "C"

    def test_template(self) -> None:
        assert _eval_doc_to_target("{label}", {"label": 2}) == "2"


class TestNormalizeTarget:
    def test_int_index(self) -> None:
        assert _normalize_target_to_index(2, ["A", "B", "C", "D"]) == 2

    def test_letter_to_index(self) -> None:
        assert _normalize_target_to_index("C", ["A", "B", "C", "D"]) == 2

    def test_string_match(self) -> None:
        assert _normalize_target_to_index("Paris", ["London", "Paris", "Rome"]) == 1

    def test_int_string(self) -> None:
        assert _normalize_target_to_index("2", ["A", "B", "C", "D"]) == 2

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            _normalize_target_to_index(5, ["A", "B"])


class TestCompileMultipleChoice:
    def test_basic_compile(self) -> None:
        spec: dict[str, Any] = {
            "task": "_test_arc_compile",
            "dataset_path": "ai2_arc",
            "dataset_name": "ARC-Challenge",
            "test_split": "test",
            "output_type": "multiple_choice",
            "doc_to_text": "Question: {question}\nAnswer:",
            "doc_to_target": "answerKey",
            "doc_to_choice": "choices",
            "num_fewshot": 5,
        }
        compiled = compile_yaml_dict(spec)
        assert issubclass(compiled, MultipleChoice)
        assert compiled.name == "_test_arc_compile"
        assert compiled.tier == "imported"
        assert compiled.n_fewshot_default == 5
        assert compiled.lm_eval_dataset_name == "ARC-Challenge"  # type: ignore[attr-defined]

    def test_compile_round_trip_via_register(self) -> None:
        spec = {
            "task": "_test_register_round_trip",
            "dataset_path": "ai2_arc",
            "output_type": "multiple_choice",
            "doc_to_text": "Q: {q}",
            "doc_to_target": "answer",
            "doc_to_choice": "choices",
        }
        cls = register_lm_eval_task(spec)
        assert cls.name == "_test_register_round_trip"
        # The class is now in the registry.
        from anvil.tasks.registry import get_task

        assert get_task("_test_register_round_trip") is cls

    def test_compile_resolves_doc_to_index_from_letter(self) -> None:
        spec = {
            "task": "_test_letter_target",
            "dataset_path": "x/y",
            "output_type": "multiple_choice",
            "doc_to_text": "{q}",
            "doc_to_target": "answer",
            "doc_to_choice": "choices",
        }
        compiled = compile_yaml_dict(spec)
        task = compiled()
        idx = task.doc_to_target({"q": "?", "answer": "C", "choices": ["W", "X", "Y", "Z"]})
        assert idx == 2


class TestCompileGenerateUntil:
    def test_basic_generate_compile(self) -> None:
        spec = {
            "task": "_test_generate_compile",
            "dataset_path": "openai/gsm8k",
            "output_type": "generate_until",
            "doc_to_text": "Q: {question}\nA:",
            "doc_to_target": "answer",
            "generation_kwargs": {"max_gen_toks": 256, "until": ["\n"]},
        }
        compiled = compile_yaml_dict(spec)
        assert not issubclass(compiled, MultipleChoice)
        assert issubclass(compiled, Task)
        assert compiled.tier == "imported"
        assert compiled.request_type == "Generate"


class TestUnsupportedFeatures:
    def test_missing_task_field(self) -> None:
        with pytest.raises(UnsupportedYAML, match="task' field"):
            compile_yaml_dict({})

    def test_unknown_output_type(self) -> None:
        with pytest.raises(UnsupportedYAML, match="output_type"):
            compile_yaml_dict({"task": "_test_bad_output_type", "output_type": "fancy_new_thing"})

    def test_compile_missing_yaml_path(self, tmp_path: object) -> None:
        from pathlib import Path

        from anvil.tasks.lm_eval_shim import compile_yaml

        with pytest.raises(ConfigError, match="not found"):
            compile_yaml(Path(tmp_path) / "nope.yaml")  # type: ignore[arg-type]


class TestApplyFilters:
    def test_regex_extracts_first_capture(self) -> None:
        out = _apply_filters(
            "the answer is 42 maybe",
            [{"filter": [{"function": "regex", "regex_pattern": r"is (\d+)"}]}],
        )
        assert out == "42"

    def test_take_first_word(self) -> None:
        out = _apply_filters(
            "  hello   world  ",
            [{"filter": [{"function": "take_first"}]}],
        )
        assert out == "hello"

    def test_unknown_filter_passes_through(self) -> None:
        out = _apply_filters(
            "no change",
            [{"filter": [{"function": "completely_invented_filter"}]}],
        )
        assert out == "no change"
