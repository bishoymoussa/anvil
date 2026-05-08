"""GSM8K — grade-school math word problems (Cobbe et al. 2021).

The "answer" in GSM8K is a number written after ``####`` in the dataset's
``answer`` field — that is the *gold* answer. Models are usually asked to
emit a chain-of-thought followed by ``#### <number>``, and the metric
extracts the final number.

Anvil ships the **strict-match** extractor by default (the last number on
the last ``####`` line). We deliberately do not implement the
"flexible-extract / first number" filter — the lm-evaluation-harness team
documents at length that it scores intermediate reasoning numbers as final
answers (issues #2278, #3214). See KB entry
``gsm8k_flexible_extract_picks_first_number``.

For M0 we run the eval against the full GSM8K test split (1,319 docs). The
``limit`` arg in :func:`anvil.eval` truncates that for fast smoke runs.
"""

from __future__ import annotations

import re
from typing import Any

from anvil.primitives.request import Generate
from anvil.primitives.response import Generation
from anvil.primitives.sampler import Sampler
from anvil.tasks.base import Task
from anvil.tasks.registry import register_task

_HASH_ANSWER = re.compile(r"####\s*([\-\d\.,]+)")
_LAST_NUMBER = re.compile(r"(-?\d+(?:[.,]\d+)*)")
_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she sold half "
            "as many clips in May. How many clips did Natalia sell altogether in "
            "April and May?"
        ),
        "answer": (
            "Natalia sold 48/2 = 24 clips in May.\n"
            "Natalia sold 48+24 = 72 clips altogether in April and May.\n"
            "#### 72"
        ),
    },
    {
        "question": (
            "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 "
            "minutes of babysitting. How much did she earn?"
        ),
        "answer": (
            "Weng earns 12/60 = $0.20 per minute.\n"
            "Working 50 minutes, she earned 0.20 x 50 = $10.\n"
            "#### 10"
        ),
    },
    {
        "question": (
            "Betty is saving money for a new wallet which costs $100. Betty has only "
            "half of the money she needs. Her parents decided to give her $15 for "
            "that purpose, and her grandparents twice as much as her parents. How "
            "much more money does Betty need to buy the wallet?"
        ),
        "answer": (
            "Betty has $100 / 2 = $50.\n"
            "Her grandparents gave her $15 * 2 = $30.\n"
            "Betty has $50 + $15 + $30 = $95 in total.\n"
            "She needs $100 - $95 = $5 more.\n"
            "#### 5"
        ),
    },
    {
        "question": (
            "Julie is reading a 120-page book. Yesterday, she was able to read 12 "
            "pages and today, she read twice as many pages as yesterday. If she "
            "wants to read half of the remaining pages tomorrow, how many pages "
            "should she read?"
        ),
        "answer": (
            "Julie read 12 * 2 = 24 pages today.\n"
            "She has read 12 + 24 = 36 pages so far.\n"
            "She has 120 - 36 = 84 pages remaining.\n"
            "Half of the remaining pages is 84 / 2 = 42.\n"
            "#### 42"
        ),
    },
    {
        "question": (
            "James decides to run 3 sprints 3 times a week. He runs 60 meters each "
            "sprint. How many total meters does he run a week?"
        ),
        "answer": (
            "He runs 3 * 3 = 9 sprints a week.\nHe runs 9 * 60 = 540 meters a week.\n#### 540"
        ),
    },
    {
        "question": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How "
            "many bolts in total does it take?"
        ),
        "answer": ("It takes 2 / 2 = 1 bolt of white fiber.\nTotal bolts = 2 + 1 = 3.\n#### 3"),
    },
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
            "morning and bakes muffins for her friends every day with four. She "
            "sells the remainder at the farmers' market daily for $2 per fresh duck "
            "egg. How much in dollars does she make every day at the farmers' "
            "market?"
        ),
        "answer": ("Eggs left = 16 - 3 - 4 = 9.\nEarnings = 9 * 2 = 18 dollars.\n#### 18"),
    },
    {
        "question": (
            "If a number is doubled and then increased by 6, the result is 50. What is the number?"
        ),
        "answer": ("Let x be the number. 2x + 6 = 50, so 2x = 44, so x = 22.\n#### 22"),
    },
]


def _extract_number(text: str) -> str:
    """Return the canonical numeric answer string from ``text``.

    Strategy: first look for a ``####`` answer line (strict-match). If none is
    found, fall back to the last number anywhere in ``text``. Comma group
    separators are stripped; trailing zeros after a decimal point are
    preserved (so ``1.0`` and ``1`` are not collapsed silently — that is the
    extractor's job, not the metric's).
    """
    m = _HASH_ANSWER.search(text)
    if m is not None:
        return _normalize_number(m.group(1))
    matches = list(_LAST_NUMBER.finditer(text))
    if matches:
        return _normalize_number(matches[-1].group(1))
    return ""


def _normalize_number(raw: str) -> str:
    """Strip ``,`` thousand-separators and trailing whitespace; keep sign and decimals."""
    s = raw.replace(",", "").strip()
    if s.endswith("."):
        s = s[:-1]
    # Normalize integer-valued floats: "5.0" → "5"
    if s.endswith(".0"):
        s = s[:-2]
    return s


def _gold_answer(doc: dict[str, Any]) -> str:
    raw = str(doc.get("answer", ""))
    m = _HASH_ANSWER.search(raw)
    if m is not None:
        return _normalize_number(m.group(1))
    matches = list(_LAST_NUMBER.finditer(raw))
    return _normalize_number(matches[-1].group(1)) if matches else ""


@register_task
class GSM8K(Task):
    """GSM8K — strict-match on ``#### <number>`` (design §6.5)."""

    name = "gsm8k"
    dataset = "openai/gsm8k"
    fewshot_style = "interleaved"
    n_fewshot_default = 5
    metric_name = "accuracy"
    request_type = "Generate"
    tier = "curated"

    sentinel_prompt = "What is two plus two? Answer with #### N."
    sentinel_expected = "4"
    sentinel_baseline_scores = {
        # Approximate published numbers; CaaS sentinel uses these as a floor.
        "meta-llama/Llama-3.1-8B-Instruct": 0.84,
        "Qwen/Qwen2.5-7B-Instruct": 0.84,
    }

    def __init__(
        self,
        *,
        n_fewshot: int | None = None,
        limit: int | None = None,
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(n_fewshot=n_fewshot, limit=limit)
        self.max_new_tokens = max_new_tokens
        # Hold a fixed pool of fewshot exemplars; the runner asks for the
        # first ``n_fewshot``. Order-stability is part of the manifest.
        self._fewshot_pool = list(_FEW_SHOT_EXAMPLES)

    def _fewshot_messages(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for shot in self._fewshot_pool[: self.n_fewshot]:
            out.append({"role": "user", "content": shot["question"]})
            out.append({"role": "assistant", "content": shot["answer"]})
        return out

    def doc_to_request(self, doc: dict[str, Any]) -> Generate:
        messages: list[dict[str, Any]] = []
        if self.n_fewshot > 0:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "You are a math tutor. After your reasoning, write the final "
                        "answer on its own line in the format `#### N`."
                    ),
                }
            )
            messages.extend(self._fewshot_messages())
        else:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Solve the problem. Write the final answer on its own line as `#### N`."
                    ),
                }
            )
        messages.append({"role": "user", "content": doc["question"]})

        sampler = Sampler.greedy(max_tokens=self.max_new_tokens)
        return Generate(messages=tuple(messages), sampler=sampler)

    def request_to_prediction(self, response: Any, doc: dict[str, Any]) -> str:
        del doc
        if not isinstance(response, Generation):
            raise TypeError(f"GSM8K expected Generation, got {type(response).__name__}")
        return _extract_number(response.text)

    def aggregate(self, predictions: list[Any], docs: list[dict[str, Any]]) -> dict[str, float]:
        if not predictions:
            return {self.metric_name: 0.0}
        if len(predictions) != len(docs):
            raise ValueError(f"GSM8K aggregate: {len(predictions)} preds vs {len(docs)} docs")
        correct = 0
        for pred, doc in zip(predictions, docs, strict=True):
            gold = _gold_answer(doc)
            if pred and gold and pred == gold:
                correct += 1
        return {self.metric_name: correct / len(predictions)}


__all__ = ["GSM8K"]
