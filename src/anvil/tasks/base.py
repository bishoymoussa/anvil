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
from collections.abc import Callable, Iterable, Iterator
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
    def doc_to_request(self, doc: dict[str, Any]) -> Request:
        """Materialize a document into an engine request."""

    @abstractmethod
    def request_to_prediction(self, response: Any, doc: dict[str, Any]) -> Any:
        """Extract a prediction from the engine's response.

        ``response`` is one element of the engine's batched output (e.g. a
        :class:`~anvil.primitives.response.Generation` for ``Generate``
        requests, a ``(logprob, is_greedy)`` tuple for log-likelihood, etc.).
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


__all__ = ["Task", "DatasetSpec", "Tier", "materialize_dataset"]
