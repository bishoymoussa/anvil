"""Runner integration test for multi-request-per-doc tasks (MMLU shape).

Exercises the M1 runner's MCQ dispatch path end-to-end with a programmatic
dataset and the :class:`StubEngine`. No model, no GPU, no network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anvil.tasks.base import MultipleChoice
from anvil.tasks.runner import run_eval
from helpers import StubEngine

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _toy_mcq_dataset() -> Iterator[dict[str, Any]]:
    """6 docs whose gold letter is embedded in the prompt's `Answer: X` cue."""
    yield from [
        {"q": "Q1", "gold": "A"},
        {"q": "Q2", "gold": "B"},
        {"q": "Q3", "gold": "C"},
        {"q": "Q4", "gold": "D"},
        {"q": "Q5", "gold": "A"},
        {"q": "Q6", "gold": "B"},
    ]


class _ToyMCQ(MultipleChoice):
    name = "_toy_mcq_runner"
    dataset = staticmethod(_toy_mcq_dataset)

    def doc_to_text(self, doc: dict[str, Any]) -> str:
        # The StubEngine looks at the last standalone uppercase letter as the
        # gold cue, so encoding it in the context lets us produce realistic
        # logprob orderings without a model.
        return f"Q: {doc['q']}\nAnswer: {doc['gold']}"

    def doc_to_choices(self, doc: dict[str, Any]) -> list[str]:
        del doc
        return [" A", " B", " C", " D"]

    def doc_to_target(self, doc: dict[str, Any]) -> int:
        return "ABCD".index(doc["gold"])


def test_runner_dispatches_mcq_through_loglikelihood(tmp_path: Path) -> None:
    task = _ToyMCQ(n_fewshot=0, limit=6)
    engine = StubEngine(model_id="stub/mcq")
    result = run_eval(engine=engine, tasks=[task], output_dir=tmp_path)

    score = result.scores[task.name]["accuracy"]
    # StubEngine misses every 3rd doc (miss_every=3); 6 docs → 4 correct → 2/3.
    assert 0.4 < score < 0.95
    # The manifest records LogLikelihood as the request type for this task.
    assert result.manifest.tasks[0].request_type == "LogLikelihood"


def test_runner_groups_responses_per_doc() -> None:
    task = _ToyMCQ(n_fewshot=0, limit=2)
    engine = StubEngine(model_id="stub/mcq", miss_every=0)
    result = run_eval(engine=engine, tasks=[task])
    # Each doc emits 4 LogLikelihood requests; raw_outputs[task] should be
    # a list of len(docs) where each element is a list of 4 (logprob, greedy).
    raw = result.outputs[task.name]
    assert len(raw) == 2
    for per_doc in raw:
        assert len(per_doc) == 4
