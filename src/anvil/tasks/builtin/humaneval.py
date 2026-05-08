"""HumanEval / HumanEval+ — code generation pass@k (Chen et al. 2021; EvalPlus 2023).

The model is given a Python function signature + docstring and must emit
a complete implementation. Each completion is concatenated with the
problem's test harness and executed in a subprocess with a timeout; the
metric is :func:`anvil.metrics.pass_at_k`.

For M1 we ship pass@1 with greedy sampling — the published-baseline default.
Sampling-with-n>1 (pass@10/100) is supported via the ``Sampler.n`` field but
the default keeps things deterministic.

Sandboxing
----------

Generated code is executed in a fresh subprocess with a hard timeout
(``DEFAULT_TIMEOUT`` seconds), inheriting only ``PATH`` and ``HOME``.
Stdout/stderr are captured, not displayed. This is *not* a security
sandbox — researchers running untrusted weights on a private machine. If
you need stronger isolation, run inside a container.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from typing import Any

from anvil.primitives.request import Generate
from anvil.primitives.response import Generation
from anvil.primitives.sampler import Sampler
from anvil.tasks.base import Task
from anvil.tasks.registry import register_task

DEFAULT_TIMEOUT: float = 10.0
"""Per-completion subprocess timeout in seconds."""


# Match a Python code fence; greedy across newlines so we get the full body.
_CODE_FENCE_PY = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(text: str, *, prompt: str | None = None) -> str:
    """Pull a runnable Python implementation out of ``text``.

    Handles three common shapes models produce:

    1. ```python …```` — fenced code block (instruct-tuned chat output).
    2. Raw code starting at column 0 (base-model continuation).
    3. ``def …`` somewhere mid-text.

    If ``prompt`` is supplied (the original HumanEval prompt that ends with
    a function header), and the extracted code does not already define that
    function, we prepend the prompt — this lets a model that emitted only
    the body still produce a runnable program.

    Example:
        >>> extract_code("Here you go:\\n```python\\ndef f():\\n    return 1\\n```")
        'def f():\\n    return 1'
    """
    fence = _CODE_FENCE_PY.search(text)
    if fence is not None:
        body = fence.group(1).rstrip()
    else:
        # No fence: take everything from the first `def` or `import` line.
        m = re.search(r"^(def |import |from |class )", text, re.MULTILINE)
        body = text[m.start() :].rstrip() if m else text.strip()

    if prompt is not None:
        # Heuristic: if the prompt declares a function and the body doesn't
        # already include `def <same_name>`, prepend the prompt.
        decl = re.search(r"^def\s+(\w+)\s*\(", prompt, re.MULTILINE)
        if decl is not None:
            name = decl.group(1)
            if not re.search(rf"^def\s+{re.escape(name)}\s*\(", body, re.MULTILINE):
                # Strip a leading "return ..."-only body and join with prompt.
                if body and not body.startswith("def "):
                    body = textwrap.indent(body, "    ")
                body = prompt.rstrip() + "\n" + body
    return body


def run_completion(
    code: str,
    test: str,
    entry_point: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """Execute ``code + test + check(<entry_point>)`` in a subprocess.

    Returns True iff the subprocess exits with code 0. Captures (and discards)
    stdout/stderr so noisy generations don't pollute the test log.
    """
    program = "\n".join(
        [
            code,
            "",
            test,
            "",
            f"check({entry_point})",
        ]
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            timeout=timeout,
            check=False,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        )
    except subprocess.TimeoutExpired:
        return False
    return result.returncode == 0


class _HumanEvalBase(Task):
    """Common machinery for HumanEval and HumanEval+ (design §6.5)."""

    fewshot_style = "none"
    n_fewshot_default = 0
    metric_name = "pass@1"
    request_type = "Generate"
    tier = "curated"

    sentinel_prompt = 'def add(a: int, b: int) -> int:\n    """Return a + b."""\n'
    sentinel_expected = "    return a + b"

    def __init__(
        self,
        *,
        n_fewshot: int | None = None,
        limit: int | None = None,
        max_new_tokens: int = 512,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(n_fewshot=n_fewshot, limit=limit)
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout

    def doc_to_request(self, doc: dict[str, Any]) -> Generate:
        prompt = str(doc["prompt"])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert Python programmer. Complete the function "
                    "below. Reply with the complete function in a single ```python "
                    "code block."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        sampler = Sampler.greedy(max_tokens=self.max_new_tokens)
        return Generate(messages=tuple(messages), sampler=sampler)

    def request_to_prediction(self, response: Any, doc: dict[str, Any]) -> str:
        if not isinstance(response, Generation):
            raise TypeError(f"HumanEval expected Generation, got {type(response).__name__}")
        return extract_code(response.text, prompt=str(doc.get("prompt", "")))

    def aggregate(self, predictions: list[Any], docs: list[dict[str, Any]]) -> dict[str, float]:
        from anvil.metrics.pass_at_k import pass_at_k

        if not predictions:
            return {self.metric_name: 0.0}
        if len(predictions) != len(docs):
            raise ValueError(f"HumanEval aggregate: {len(predictions)} preds vs {len(docs)} docs")
        passes = 0
        for pred, doc in zip(predictions, docs, strict=True):
            test = self._test_for(doc)
            entry_point = str(doc["entry_point"])
            if run_completion(pred, test, entry_point, timeout=self.timeout):
                passes += 1
        return {self.metric_name: pass_at_k(n=len(predictions), c=passes, k=1)}

    def _test_for(self, doc: dict[str, Any]) -> str:
        """Return the test harness string for ``doc``. Subclasses may override."""
        return str(doc["test"])


@register_task
class HumanEval(_HumanEvalBase):
    """Original HumanEval (Chen et al. 2021). 164 problems, single-test harness."""

    name = "humaneval"
    dataset = "openai_humaneval"


@register_task
class HumanEvalPlus(_HumanEvalBase):
    """HumanEval+ (Liu et al. 2023). Same 164 problems, ~80x more tests per problem.

    Uses the ``test_plus`` field if present in the EvalPlus dataset variant;
    falls back to ``test`` otherwise.
    """

    name = "humaneval_plus"
    dataset = "evalplus/humanevalplus"
    metric_name = "pass@1"

    def _test_for(self, doc: dict[str, Any]) -> str:
        if "test_plus" in doc:
            return str(doc["test_plus"])
        return str(doc.get("test", ""))


__all__ = ["HumanEval", "HumanEvalPlus", "extract_code", "run_completion"]
