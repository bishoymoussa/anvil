"""MMLU — Massive Multitask Language Understanding (Hendrycks et al. 2021).

4-way multiple choice across 57 subjects. The published Llama-3.1-8B-Instruct
5-shot baseline is ~0.69 accuracy with letter-based scoring (the
lm-evaluation-harness convention: score the continuations ``" A"``, ``" B"``,
``" C"``, ``" D"`` and pick argmax).

We use the ``cais/mmlu`` HF dataset with config ``"all"``. Few-shot exemplars
come from the dataset's ``dev`` split, which provides 5 exemplars per
subject by design — the canonical 5-shot setup.

Per-subject sampling: when running the full dataset, the test set covers all
57 subjects mixed together; each test doc carries its ``subject`` field, and
we look up that subject's dev exemplars (5 per subject) for the few-shot
context. That keeps the 5-shot exemplars in-domain — the published-baseline
convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anvil.tasks.base import MultipleChoice
from anvil.tasks.registry import register_task

if TYPE_CHECKING:
    from collections.abc import Sequence

_LETTERS = ("A", "B", "C", "D")


def _format_doc(question: str, choices: Sequence[str]) -> str:
    """Render the shared MCQ stem: question + 4 numbered options + ``Answer:`` cue."""
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(_LETTERS, choices, strict=True)
    )
    return f"{question}\n{options}\nAnswer:"


@register_task
class MMLU(MultipleChoice):
    """MMLU — 4-way MCQ scored by letter log-likelihood (design §6.5).

    Default 5-shot interleaved per the published baseline. Use ``n_fewshot=0``
    for zero-shot.
    """

    name = "mmlu"
    dataset = "cais/mmlu"
    dataset_config = "all"
    dataset_split = "test"
    fewshot_style = "interleaved"
    n_fewshot_default = 5
    metric_name = "accuracy"
    tier = "curated"

    sentinel_prompt = (
        "What is the capital of France?\nA. Berlin\nB. Madrid\nC. Paris\nD. Rome\nAnswer:"
    )
    sentinel_expected = "C"
    sentinel_baseline_scores = {
        "meta-llama/Llama-3.1-8B-Instruct": 0.69,
        "Qwen/Qwen2.5-7B-Instruct": 0.74,
    }

    def __init__(
        self,
        *,
        n_fewshot: int | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(n_fewshot=n_fewshot, limit=limit)
        # Built once on first ``doc_to_text`` call. Maps subject → list of dev
        # docs. The dev split has 5 docs per subject by design.
        self._fewshot_pool: dict[str, list[dict[str, Any]]] | None = None

    def _ensure_fewshot_pool(self) -> dict[str, list[dict[str, Any]]]:
        """Lazily load and group the dev split by subject."""
        if self._fewshot_pool is not None:
            return self._fewshot_pool
        if self.n_fewshot == 0:
            self._fewshot_pool = {}
            return self._fewshot_pool
        from datasets import load_dataset

        ds = load_dataset(self.dataset, "all", split="dev")
        pool: dict[str, list[dict[str, Any]]] = {}
        for row in ds:
            subj = str(row.get("subject", "default"))
            pool.setdefault(subj, []).append(dict(row))
        self._fewshot_pool = pool
        return pool

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        """Render the prompt: optional 5-shot exemplars + this question's stem."""
        question = str(doc["question"])
        choices = list(doc["choices"])
        if self.n_fewshot == 0:
            return _format_doc(question, choices)

        pool = self._ensure_fewshot_pool()
        subject = str(doc.get("subject", "default"))
        exemplars = pool.get(subject, [])[: self.n_fewshot]
        parts: list[str] = []
        if exemplars:
            parts.append(
                f"The following are multiple choice questions (with answers) about "
                f"{subject.replace('_', ' ')}.\n"
            )
        for ex in exemplars:
            ex_text = _format_doc(str(ex["question"]), list(ex["choices"]))
            ex_letter = _LETTERS[int(ex["answer"])]
            parts.append(f"{ex_text} {ex_letter}\n")
        parts.append(_format_doc(question, choices))
        return "\n".join(parts)

    # ``doc_to_choices`` inherited from :class:`MultipleChoice` — defaults
    # to ``["A", "B", "C", "D"]`` under ``chat_templated=True`` (the
    # published-baseline configuration for instruct-tuned reference
    # checkpoints) and ``[" A", " B", " C", " D"]`` for base models.

    def doc_to_target(self, doc: dict[str, Any]) -> int:
        return int(doc["answer"])


__all__ = ["MMLU"]
