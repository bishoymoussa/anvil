"""Public ``anvil.eval`` — the user-facing entrypoint (design §6.1).

The simplest invocation is one line:

    result = anvil.eval(model="...", tasks=["gsm8k"], n_fewshot=5)

Everything else (samplers, hidden-state capture, custom logits processors,
chat-template overrides) is added incrementally without rebuilding the
caller. The defaults are designed to be the right defaults: greedy sampler,
no penalties, 5-shot for tasks that default to 5-shot.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.engine import build_engine
from anvil.exceptions import ConfigError
from anvil.logging import get_logger
from anvil.models.registry import LoadedModel
from anvil.tasks.base import Task
from anvil.tasks.registry import get_task
from anvil.tasks.runner import EvalRunResult, run_eval

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anvil.engine.public import Engine

_log = get_logger(__name__)


def _instantiate(
    name_or_cls: str | type[Task] | Task,
    *,
    n_fewshot: int | None,
    limit: int | None,
) -> Task:
    """Resolve ``name | class | instance`` to a configured :class:`Task` instance."""
    if isinstance(name_or_cls, Task):
        # Caller already configured fewshot/limit on the instance.
        return name_or_cls
    if isinstance(name_or_cls, str):
        cls = get_task(name_or_cls)
    elif isinstance(name_or_cls, type) and issubclass(name_or_cls, Task):
        cls = name_or_cls
    else:
        raise ConfigError(
            f"tasks must be names, Task subclasses, or Task instances; got "
            f"{type(name_or_cls).__name__}"
        )
    return cls(n_fewshot=n_fewshot, limit=limit)


def eval(  # noqa: A001 - shadowing builtins is intentional (matches design §10.1)
    *,
    model: str | LoadedModel | Engine,
    tasks: Iterable[str | type[Task] | Task],
    n_fewshot: int | None = None,
    limit: int | None = None,
    engine: str = "auto",
    revision: str | None = None,
    dtype: str | None = None,
    device_map: str | None = None,
    engine_args: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> EvalRunResult:
    """Run an evaluation. Returns scores + manifest.

    Args:
        model: HF model id, an :class:`anvil.LoadedModel` from
            :func:`anvil.load`, or a raw :class:`Engine`.
        tasks: an iterable of task names (strings) or ``Task`` subclasses /
            instances.
        n_fewshot: number of few-shot exemplars; defaults to each task's
            ``n_fewshot_default``.
        limit: cap docs per task. ``None`` means run the full split.
        engine: backend choice, propagated to :func:`anvil.engine.build_engine`.
        revision, dtype, device_map, engine_args: forwarded to
            ``build_engine`` when ``model`` is a string.
        output_dir: if given, the manifest is written to
            ``output_dir/manifest.json``.

    Example:
        >>> import anvil  # doctest: +SKIP
        >>> result = anvil.eval(model="...", tasks=["gsm8k"], limit=10)  # doctest: +SKIP
        >>> result.scores  # doctest: +SKIP
    """
    task_list = [_instantiate(t, n_fewshot=n_fewshot, limit=limit) for t in tasks]
    if not task_list:
        raise ConfigError("anvil.eval: tasks=[] — nothing to do")

    own_engine = False
    if isinstance(model, str):
        eng: Engine = build_engine(
            model_id=model,
            engine=engine,  # type: ignore[arg-type]
            revision=revision,
            dtype=dtype,
            device_map=device_map,
            engine_args=engine_args,
        )
        own_engine = True
    elif isinstance(model, LoadedModel):
        eng = model.engine
    else:
        eng = model

    try:
        return run_eval(
            engine=eng,
            tasks=task_list,
            output_dir=Path(output_dir) if output_dir is not None else None,
        )
    finally:
        if own_engine:
            eng.shutdown()


__all__ = ["eval"]
