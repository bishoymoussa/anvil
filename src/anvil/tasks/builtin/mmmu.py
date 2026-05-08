"""MMMU — Massive Multi-discipline Multimodal Understanding (Yue et al. 2024).

Multiple-choice questions across 30 subjects, each carrying one or more
images. Published baselines (Qwen2.5-VL-7B-Instruct on MMMU validation
4-way subset): ~0.50 accuracy.

Scoring strategy
----------------

For VLMs the published-baseline convention is **generation-based**:
prompt the model with question + images + lettered options, ask it to
respond with a single letter, parse the first ``[A-J]`` from the output,
score against the gold. We use the same approach because:

* The mm-runner's ``loglikelihood`` is documented as not supporting
  image-conditioned scoring in v0 (M5 work alongside the lm-eval shim).
* lm-evaluation-harness, OpenCompass, and the MMMU paper itself all use
  generation-based scoring.
* It's the published-baseline configuration so the
  ``test_milestone_4_mmmu_known_baseline`` check is meaningful.

For v0 we filter to the 4-option ``multiple-choice`` subset (most of the
validation split). Open-ended and 5+-option questions are skipped (they
score 0 if encountered). M5 may extend with proper logit-based scoring
once the VLM logit-extraction story matures.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from anvil.exceptions import TaskError
from anvil.primitives.request import Generate
from anvil.primitives.response import Generation
from anvil.primitives.sampler import Sampler
from anvil.tasks.base import Task
from anvil.tasks.registry import register_task

_LETTERS = ("A", "B", "C", "D")
_LETTER_PATTERN = re.compile(r"\b([A-J])\b")


def _parse_options(raw: Any) -> list[str]:
    """MMMU stores options as a stringified Python list. Decode safely.

    Returns an empty list if the field is missing or unparseable — the
    runner sees an empty options list and skips the row at scoring time.
    """
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not isinstance(raw, str):
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def _collect_images(doc: dict[str, Any], n_max: int = 7) -> list[Any]:
    """Walk ``image_1`` … ``image_n_max`` and return non-null payloads in order.

    MMMU's ``question`` text references images as ``<image 1>``, ``<image 2>``,
    etc. The runtime substitutes those tokens with actual image content via
    Anvil's multimodal request shape.
    """
    out: list[Any] = []
    for i in range(1, n_max + 1):
        payload = doc.get(f"image_{i}")
        if payload is not None:
            out.append(payload)
    return out


def _extract_letter(text: str) -> str | None:
    """Return the first ``A`` … ``J`` letter found in ``text``, or None."""
    m = _LETTER_PATTERN.search(text)
    if m is None:
        return None
    return m.group(1)


@register_task
class MMMU(Task):
    """MMMU — generation-scored 4-way multiple-choice (design §6.5)."""

    name = "mmmu"
    dataset = "MMMU/MMMU"
    fewshot_style = "none"
    n_fewshot_default = 0
    metric_name = "accuracy"
    request_type = "Generate"
    tier = "curated"

    sentinel_prompt = (
        "What color is a typical school bus?\n"
        "A. Red\nB. Yellow\nC. Blue\nD. Green\nAnswer with a single letter."
    )
    sentinel_expected = "B"
    sentinel_baseline_scores = {
        "Qwen/Qwen2.5-VL-7B-Instruct": 0.50,
    }

    def __init__(
        self,
        *,
        n_fewshot: int | None = None,
        limit: int | None = None,
        max_new_tokens: int = 16,
    ) -> None:
        super().__init__(n_fewshot=n_fewshot, limit=limit)
        self.max_new_tokens = max_new_tokens

    def doc_to_request(self, doc: dict[str, Any]) -> Generate:
        question = str(doc.get("question", ""))
        options = _parse_options(doc.get("options"))
        images = _collect_images(doc)

        # Build the multimodal user content: text segment with options + every
        # image referenced. Image-tokens in the question text (``<image 1>``)
        # are kept verbatim — the model's processor handles substitution.
        rendered_options = "\n".join(
            f"{letter}. {opt}"
            for letter, opt in zip(_LETTERS, options[: len(_LETTERS)], strict=False)
        )

        content: list[dict[str, Any]] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append(
            {
                "type": "text",
                "text": (
                    f"{question}\n{rendered_options}\nAnswer with a single letter (A, B, C, or D)."
                ),
            }
        )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are an expert at multiple-choice questions about academic "
                    "and professional domains. Reply with the single letter of "
                    "the correct option only."
                ),
            },
            {"role": "user", "content": content},
        ]

        sampler = Sampler.greedy(max_tokens=self.max_new_tokens)
        return Generate(messages=tuple(messages), sampler=sampler)

    def request_to_prediction(self, response: Any, doc: dict[str, Any]) -> str:
        del doc
        if not isinstance(response, Generation):
            raise TaskError(f"MMMU expected Generation, got {type(response).__name__}")
        letter = _extract_letter(response.text)
        return letter or ""

    def aggregate(self, predictions: list[Any], docs: list[dict[str, Any]]) -> dict[str, float]:
        if not predictions:
            return {self.metric_name: 0.0}
        if len(predictions) != len(docs):
            raise TaskError(f"MMMU aggregate: {len(predictions)} preds vs {len(docs)} docs")
        scored = 0
        correct = 0
        for pred, doc in zip(predictions, docs, strict=True):
            gold = str(doc.get("answer", "")).strip().upper()[:1]
            if gold not in _LETTERS:
                # Open-ended question or non-letter answer — skip.
                continue
            scored += 1
            if pred and pred.upper() == gold:
                correct += 1
        if scored == 0:
            return {self.metric_name: 0.0}
        return {self.metric_name: correct / scored}


__all__ = ["MMMU"]
