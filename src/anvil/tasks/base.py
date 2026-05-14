"""``Task`` ABC (design §6.3, §16.4).

Five-line registration of any task that fits the abstraction:

* ``doc_to_request(doc) -> Request``
* ``request_to_prediction(response, doc) -> Any``
* (optional) ``aggregate(predictions, docs) -> dict[str, float]``
* (optional) ``metric``, ``fewshot_style``, ``n_fewshot_default``,
  ``sentinel_*`` — see design §6.7 for non-text examples.

The default :meth:`aggregate` averages :attr:`metric` across docs. Override
it for rank correlations, top-k, weighted averages, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from anvil.exceptions import TaskError

if TYPE_CHECKING:
    from anvil.primitives.chat_template import FewshotStyle
    from anvil.primitives.request import Request

DatasetSpec = str | Path | Callable[[], Iterable[dict[str, Any]]]
Tier = Literal["curated", "imported", "custom"]


class Task(ABC):
    """Base class for all tasks.

    Class attributes (override on subclasses):
        name: stable identifier used in CLI flags and manifests.
        dataset: HF id (``"cais/mmlu"``), local path, or a zero-arg callable
            returning an iterable of dicts.
        fewshot_style: how few-shot examples are packed (design §4.1).
        n_fewshot_default: default fewshot count when the user doesn't override.
        metric: optional per-doc metric (``Callable[[pred, target], float]``).
        sentinel_prompt / sentinel_expected: fixed-answer probe used by the
            CaaS quality sentinel (design §6.6).
        sentinel_baseline_scores: per-model expected sentinel scores.
    """

    name: ClassVar[str]
    dataset: ClassVar[DatasetSpec]
    # HF ``datasets.load_dataset(path, name=...)`` config; e.g. cais/mmlu
    # requires "all", openai/gsm8k requires "main". ``None`` means the
    # dataset has only a default config and ``load_dataset(path)`` works.
    dataset_config: ClassVar[str | None] = None
    dataset_split: ClassVar[str] = "test"
    fewshot_style: ClassVar[FewshotStyle] = "interleaved"
    n_fewshot_default: ClassVar[int] = 0
    metric: ClassVar[Callable[[Any, Any], float] | None] = None
    metric_name: ClassVar[str] = "accuracy"
    tier: ClassVar[Tier] = "curated"
    request_type: ClassVar[Literal["Generate", "LogLikelihood", "Embed", "Classify", "Custom"]] = (
        "Generate"
    )

    sentinel_prompt: ClassVar[str | None] = None
    sentinel_expected: ClassVar[str | None] = None
    sentinel_baseline_scores: ClassVar[dict[str, float]] = {}

    # Each Task is instantiated once per run; subclasses may override
    # __init__ to plumb subset/limit args, but must call super().
    def __init__(self, *, n_fewshot: int | None = None, limit: int | None = None) -> None:
        self.n_fewshot = self.n_fewshot_default if n_fewshot is None else n_fewshot
        self.limit = limit
        if self.n_fewshot < 0:
            raise TaskError(f"n_fewshot must be ≥ 0, got {self.n_fewshot}")

    # ---------------------------------------------------- contract methods
    @abstractmethod
    def doc_to_request(self, doc: dict[str, Any]) -> Request | Sequence[Request]:
        """Materialize a document into one or more engine requests.

        Most tasks return a single :class:`Request`. Multiple-choice tasks
        return a parallel sequence (one request per option); the runner
        flattens, dispatches in batches at engine throughput, then re-groups
        per-doc responses for :meth:`request_to_prediction`.
        """

    @abstractmethod
    def request_to_prediction(self, response: Any, doc: dict[str, Any]) -> Any:
        """Extract a prediction from the engine's response(s).

        For single-request tasks, ``response`` is one element of the engine's
        batched output (e.g. a :class:`~anvil.primitives.response.Generation`
        for ``Generate`` requests, a ``(logprob, is_greedy)`` tuple for
        log-likelihood). For multi-request tasks (multiple-choice),
        ``response`` is the parallel ``list`` of per-option responses.
        """

    def aggregate(self, predictions: list[Any], docs: list[dict[str, Any]]) -> dict[str, float]:
        """Default: average :attr:`metric` over docs.

        Override for rank correlations, top-k, weighted averages, etc.
        """
        # Access through the class to bypass mypy's bound-method binding for
        # ClassVar[Callable]; the runtime semantics are identical.
        metric_fn = type(self).metric
        if metric_fn is None:
            raise TaskError(
                f"task {self.name!r}: no metric set and no aggregate() override; "
                "either define `metric = ...` on the subclass or implement aggregate()"
            )
        if len(predictions) != len(docs):
            raise TaskError(
                f"task {self.name!r}: aggregate got {len(predictions)} preds vs {len(docs)} docs"
            )
        if not predictions:
            return {self.metric_name: 0.0}
        scores = [metric_fn(p, self._target(d)) for p, d in zip(predictions, docs, strict=True)]
        return {self.metric_name: sum(scores) / len(scores)}

    # The default aggregate uses this hook; subclasses with non-trivial
    # targets override it.
    def _target(self, doc: dict[str, Any]) -> Any:
        if "target" in doc:
            return doc["target"]
        if "answer" in doc:
            return doc["answer"]
        raise TaskError(
            f"task {self.name!r}: doc has no 'target' or 'answer' field; override "
            "_target() or aggregate()"
        )


class MultipleChoice(Task):
    """N-way multiple-choice task scored by per-option log-likelihood (design §6.3).

    Subclasses provide:

    * ``doc_to_text(doc)`` — the shared context (question + the rendered options).
    * ``doc_to_choices(doc)`` — the per-option continuation strings.
    * ``doc_to_target(doc)`` — the gold option index, ``int`` in
      ``[0, len(choices))``.

    Two scoring modes (controlled by :attr:`chat_templated`):

    * ``chat_templated=True`` (default — instruct-tuned models): the
      engine wraps the context in a single user message, applies the
      model's chat template with ``add_generation_prompt=True``, and
      scores the continuation as the assistant turn's first tokens. The
      default :meth:`doc_to_choices` therefore returns letters
      *without* a leading space (``["A", "B", "C", "D"]``) — the chat
      template's trailing newline does the separation.
    * ``chat_templated=False`` (base models): raw prompt, the
      continuation includes a leading space.

    Why this matters: lm-evaluation-harness #1841 documents that score
    differences of 5–15pp on Llama-3-Instruct GSM8K/MMLU come from
    misapplying the chat template. Anvil's manifest tags every task
    with the mode it ran under so a reviewer can see which path produced
    the published-baseline number.
    """

    request_type: ClassVar[Literal["LogLikelihood"]] = "LogLikelihood"
    chat_templated: ClassVar[bool] = False
    """When True, the engine applies the chat template before scoring.

    Default is False because **single-turn-fewshot + chat-template
    scoring is empirically worse than raw-prompt scoring** on
    MMLU-shaped tasks for instruct-tuned models. Verified live:
    Qwen2.5-7B MMLU went 0.615 → 0.44 with ``chat_templated=True``
    under single-turn fewshot. The reason: the model was trained to
    start its assistant turn conversationally ("Sure, the answer
    is...") — so ``P("A" | <|im_start|>assistant\\n)`` is low even
    when "A" is correct.

    The correct fix for instruct-tuned models is :class:`MultiTurnFewshot`:
    each exemplar becomes its own user/assistant message pair, so the model
    sees the full conversation history and scores the final answer token
    correctly. Use ``MultiTurnFewshot`` instead of setting this flag alone.
    """

    @abstractmethod
    def doc_to_text(self, doc: dict[str, Any]) -> str:
        """Render the shared context (question + numbered options)."""

    def doc_to_choices(self, doc: dict[str, Any]) -> Sequence[str]:
        """Per-option continuation strings.

        Default: ``["A", "B", "C", "D"]`` when chat-templated (no
        leading space — the chat template ends with a newline that
        separates the assistant turn from the prompt) or
        ``[" A", " B", " C", " D"]`` for base-model scoring.

        Subclasses override when the task has more or fewer than four
        options, or when continuations are full strings rather than
        letters.
        """
        del doc
        letters = ("A", "B", "C", "D")
        if type(self).chat_templated:
            return list(letters)
        return [f" {ll}" for ll in letters]

    @abstractmethod
    def doc_to_target(self, doc: dict[str, Any]) -> int:
        """Index of the correct option in :meth:`doc_to_choices`."""

    def doc_to_request(self, doc: dict[str, Any]) -> Sequence[Request]:
        # Imported inside the method to keep the module-top import graph
        # honest under TYPE_CHECKING.
        from anvil.primitives.request import LogLikelihood

        ctx = self.doc_to_text(doc)
        templated = type(self).chat_templated
        return [
            LogLikelihood(context=ctx, continuation=c, chat_templated=templated)
            for c in self.doc_to_choices(doc)
        ]

    def request_to_prediction(
        self, response: Sequence[tuple[float, bool]], doc: dict[str, Any]
    ) -> int:
        # ``response`` is the parallel list of (logprob, is_greedy) per choice.
        if not response:
            raise TaskError(f"task {self.name!r}: empty multiple-choice response")
        scores = [r[0] for r in response]
        return scores.index(max(scores))

    def aggregate(self, predictions: list[Any], docs: list[dict[str, Any]]) -> dict[str, float]:
        if not predictions:
            return {self.metric_name: 0.0}
        if len(predictions) != len(docs):
            raise TaskError(
                f"task {self.name!r}: aggregate got {len(predictions)} preds vs {len(docs)} docs"
            )
        correct = 0
        for pred, doc in zip(predictions, docs, strict=True):
            target = self.doc_to_target(doc)
            if pred == target:
                correct += 1
        return {self.metric_name: correct / len(predictions)}


class MultiTurnFewshot(MultipleChoice):
    """Mixin that packs few-shot exemplars as multi-turn user/assistant messages.

    This is the correct fewshot strategy for instruct-tuned models. Instead of
    concatenating all exemplars into a single long prompt (single-turn), each
    exemplar becomes its own user/assistant exchange:

    .. code-block::

        user:      "What is 2+2?\\nA. 3\\nB. 4\\nC. 5\\nD. 6\\nAnswer:"
        assistant: "B"
        user:      "What is 3+3?\\nA. 5\\nB. 6\\nC. 7\\nD. 8\\nAnswer:"
        assistant: "B"
        user:      <current question stem>   ← scored against "A"/"B"/"C"/"D"

    Why this fixes the 15pp gap: instruct models are trained to start assistant
    turns with a direct answer when the conversation context makes it obvious.
    Single-turn fewshot collapses all that context into one user message, so
    the model never sees the pattern of short-answer assistant turns and falls
    back to its "sure, the answer is…" conversational preamble, tanking the
    logprob of the bare letter.

    Subclasses must implement:

    * ``doc_to_text(doc)`` — renders the question stem (question + options +
      "Answer:" cue) for a single doc, without any fewshot prefix.
    * ``doc_to_exemplars()`` — returns the ordered list of few-shot exemplar
      dicts. Called once and cached.
    * ``exemplar_to_answer(doc)`` — returns the gold answer string (e.g. ``"B"``)
      for an exemplar doc.

    The ``chat_templated`` flag is forced True: multi-turn fewshot is only
    meaningful when the chat template is applied.
    """

    chat_templated: ClassVar[bool] = True

    def doc_to_exemplars(self) -> list[dict[str, Any]]:
        """Return the ordered fewshot exemplar pool. Default: empty (zero-shot)."""
        return []

    def exemplar_to_answer(self, doc: dict[str, Any]) -> str:
        """Return the gold answer letter for an exemplar doc."""
        raise NotImplementedError(
            f"{type(self).__name__} uses MultiTurnFewshot but doesn't implement "
            "exemplar_to_answer(). Return the gold letter string (e.g. 'B')."
        )

    def doc_to_messages(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the full multi-turn message list for ``doc``.

        Each fewshot exemplar becomes a user/assistant pair. The final user
        message is the current question stem (without the answer).
        """
        messages: list[dict[str, Any]] = []
        for ex in self.doc_to_exemplars()[: self.n_fewshot]:
            messages.append({"role": "user", "content": self.doc_to_text(ex)})
            messages.append({"role": "assistant", "content": self.exemplar_to_answer(ex)})
        messages.append({"role": "user", "content": self.doc_to_text(doc)})
        return messages

    def doc_to_request(self, doc: dict[str, Any]) -> Sequence[Request]:
        from anvil.primitives.request import LogLikelihood

        messages = self.doc_to_messages(doc)
        msg_tuple = tuple(messages)
        return [
            LogLikelihood(messages=msg_tuple, continuation=c, chat_templated=True)
            for c in self.doc_to_choices(doc)
        ]


@dataclass
class _FewshotPool:
    """A fixed pool of fewshot examples, sliced deterministically per-request."""

    docs: list[dict[str, Any]] = field(default_factory=list)

    def take(self, n: int) -> list[dict[str, Any]]:
        return list(self.docs[:n])


def materialize_dataset(spec: DatasetSpec, *, split: str = "test") -> Iterator[dict[str, Any]]:
    """Yield rows from a dataset spec (HF id, path, or callable).

    HF datasets are loaded with ``datasets.load_dataset``; local files are
    routed by extension (``.jsonl``, ``.parquet``, ``.csv``, ``.arrow``).
    """
    if callable(spec):
        for row in spec():
            yield dict(row)
        return
    if isinstance(spec, Path) or (isinstance(spec, str) and Path(spec).exists()):
        path = Path(spec)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            import json as _json

            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    yield _json.loads(line)
            return
        from datasets import Dataset, load_dataset

        if suffix == ".parquet":
            ds = Dataset.from_parquet(str(path))
        elif suffix == ".csv":
            ds = Dataset.from_csv(str(path))
        else:
            # Fall back to HF loader for arrow / json / etc.
            ds = load_dataset(str(path), split=split)
        for row in ds:
            yield dict(row)
        return
    # HF dataset id
    from datasets import load_dataset

    ds = load_dataset(str(spec), split=split)
    for row in ds:
        yield dict(row)


__all__ = [
    "Task",
    "MultipleChoice",
    "MultiTurnFewshot",
    "DatasetSpec",
    "Tier",
    "materialize_dataset",
]
