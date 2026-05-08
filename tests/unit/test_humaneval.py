"""Tests for HumanEval+ extractor + sandboxed test runner."""

from __future__ import annotations

import pytest

from anvil.tasks.builtin.humaneval import HumanEval, extract_code, run_completion


class TestExtractCode:
    def test_pulls_from_python_code_fence(self) -> None:
        text = "Here you go:\n```python\ndef f():\n    return 1\n```\nThat's it."
        assert extract_code(text) == "def f():\n    return 1"

    def test_pulls_from_bare_code(self) -> None:
        text = "def f():\n    return 1\n"
        assert extract_code(text).strip() == "def f():\n    return 1"

    def test_handles_imports_at_top(self) -> None:
        text = "Sure!\nimport math\n\ndef f(x):\n    return math.sqrt(x)"
        out = extract_code(text)
        assert out.startswith("import math")
        assert "def f" in out

    def test_prepends_prompt_when_body_only(self) -> None:
        prompt = 'def add(a, b):\n    """Return a + b."""\n'
        body = "    return a + b"
        out = extract_code(body, prompt=prompt)
        assert "def add" in out
        assert "return a + b" in out

    def test_no_prompt_prepend_when_function_already_present(self) -> None:
        prompt = "def add(a, b):\n    pass\n"
        body = "def add(a, b):\n    return a + b"
        out = extract_code(body, prompt=prompt)
        # Should not duplicate the function definition.
        assert out.count("def add") == 1


class TestRunCompletion:
    def test_correct_solution_passes(self) -> None:
        code = "def add(a, b):\n    return a + b"
        test = "def check(f):\n    assert f(2, 3) == 5\n    assert f(0, 0) == 0"
        assert run_completion(code, test, "add", timeout=5.0) is True

    def test_wrong_solution_fails(self) -> None:
        code = "def add(a, b):\n    return a - b"
        test = "def check(f):\n    assert f(2, 3) == 5"
        assert run_completion(code, test, "add", timeout=5.0) is False

    def test_infinite_loop_times_out(self) -> None:
        code = "def slow():\n    while True: pass"
        test = "def check(f):\n    f()"
        assert run_completion(code, test, "slow", timeout=1.0) is False

    def test_syntax_error_fails(self) -> None:
        code = "def broken("
        test = "def check(f):\n    pass"
        assert run_completion(code, test, "broken", timeout=5.0) is False


class TestHumanEvalTask:
    def test_request_type_is_generate(self) -> None:
        assert HumanEval.request_type == "Generate"

    def test_metric_name(self) -> None:
        assert HumanEval.metric_name == "pass@1"

    def test_doc_to_request_wraps_prompt_in_chat(self) -> None:
        task = HumanEval()
        req = task.doc_to_request({"prompt": "def add(a, b):\n    pass\n"})
        assert req.messages is not None
        # Last message is the user turn carrying the prompt.
        assert req.messages[-1]["role"] == "user"
        assert "def add" in req.messages[-1]["content"]

    def test_aggregate_runs_tests_and_returns_pass_at_1(self) -> None:
        task = HumanEval()
        docs = [
            {
                "prompt": "def add(a, b):\n    pass\n",
                "test": "def check(f):\n    assert f(2, 3) == 5",
                "entry_point": "add",
            },
            {
                "prompt": "def mul(a, b):\n    pass\n",
                "test": "def check(f):\n    assert f(2, 3) == 6",
                "entry_point": "mul",
            },
        ]
        # Predictions: first correct, second wrong.
        preds = [
            "def add(a, b):\n    return a + b",
            "def mul(a, b):\n    return a - b",
        ]
        out = task.aggregate(preds, docs)
        # 1 of 2 correct → pass@1 = 0.5.
        assert out["pass@1"] == pytest.approx(0.5)
