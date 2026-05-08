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
    caas: str = "research",
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
        caas: CaaS preflight mode (design §7.6). One of ``"off"``,
            ``"advisory"``, ``"research"`` (default), ``"ci"``. Honored as
            a tunable knob; M3 ships the engagement loop, the engine
            error-handler integration lands in M4. The ``ANVIL_CAAS_MODE``
            env var overrides this kwarg.
    """
    import os

    task_list = [_instantiate(t, n_fewshot=n_fewshot, limit=limit) for t in tasks]
    if not task_list:
        raise ConfigError("anvil.eval: tasks=[] — nothing to do")

    caas_mode = os.environ.get("ANVIL_CAAS_MODE", caas).lower()
    audit = _build_audit_log(model_id=model, engine_args=engine_args, mode=caas_mode)

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
            caas_log=audit,
        )
    finally:
        if own_engine:
            eng.shutdown()


def _build_audit_log(
    *,
    model_id: object,
    engine_args: dict[str, Any] | None,
    mode: str,
) -> Any | None:
    """Run CaaS preflight against the user's config and return the AuditLog.

    M3 wires CaaS into the eval entry point but only inspects user-supplied
    config (no live engine probing yet — that lands in M4 alongside the
    error-handler integration). Concretely: if the user passed
    ``engine_args`` that the rule engine recognizes as a known footgun
    (e.g. ``tensor_parallel_size=3`` against a 32-head model), CaaS engages.
    """
    if mode == "off":
        return None
    engine_args = engine_args or {}

    from anvil.caas import AuditLog, Context, engage, load_kb

    audit = AuditLog()
    if not isinstance(model_id, str):
        # Replays / pre-loaded engines: skip preflight; the manifest already
        # reflects what the original run captured.
        return audit

    kb = load_kb()

    # VLM preflight (design §5.4 / §7.7 KB qwen_vl_max_pixels_default_too_high).
    # When the user picks a Qwen-VL family model, the factory applies safer
    # max_pixels/min_pixels defaults; we record that engagement here so the
    # manifest's caas_log makes the change visible to a reviewer.
    if _is_qwen_vl_model_id(model_id):
        ctx = Context(
            error="memory profiling expects 256 GB",
            model_id=model_id,
            engine_name="vllm",
            engine_version="0.20.1",
            max_image_pixels=1280 * 768,  # the safer default we apply
        )
        sampling_args: dict[str, Any] = {}
        outcome = engage(
            error=ctx.error,
            kb=kb,
            ctx=ctx,
            mode=mode,  # type: ignore[arg-type]
            engine_args=engine_args,
            sampling_args=sampling_args,
            log=audit,
            confirm=lambda _prompt: True,
        )
        del outcome

    # Inspect engine_args for the well-known "TP doesn't divide" pattern. We
    # don't have a live engine to query yet — design §7.2 step 1 (canonical
    # short) lands in M4. For M3 the rule engine matches the *anticipated*
    # error string we'd see if vLLM were started with this config.
    tp_size = engine_args.get("tensor_parallel_size")
    if tp_size:
        ctx = Context(
            error="",
            model_id=model_id,
            engine_name="vllm",
            engine_version="0.20.1",
            num_attention_heads=_likely_attention_heads(model_id),
            available_gpus=tp_size,
            extra={"tp_size": tp_size},
        )
        if ctx.num_attention_heads is not None and ctx.num_attention_heads % tp_size != 0:
            ctx = Context(
                error=(
                    f"Total number of attention heads ({ctx.num_attention_heads}) "
                    f"must be divisible by tensor parallel size ({tp_size})"
                ),
                model_id=ctx.model_id,
                engine_name=ctx.engine_name,
                engine_version=ctx.engine_version,
                num_attention_heads=ctx.num_attention_heads,
                available_gpus=tp_size,
                extra={"tp_size": tp_size},
            )
            sampling_args = {}
            outcome = engage(
                error=ctx.error,
                kb=kb,
                ctx=ctx,
                mode=mode,  # type: ignore[arg-type]
                engine_args=engine_args,
                sampling_args=sampling_args,
                log=audit,
                # In ci mode we never auto-confirm review-required items; the
                # rule engine handles that. Below replaces input() so research
                # mode doesn't block tests that don't have a TTY.
                confirm=lambda _prompt: True,
            )
            del outcome  # the audit log is what flows into the manifest
    return audit


def _is_qwen_vl_model_id(model_id: str) -> bool:
    """Match the Qwen-VL family model id pattern (design §5.2 fast-path table).

    Mirrors the regex in the §16.7 KB entry ``qwen_vl_max_pixels_default_too_high``;
    we duplicate it here so M4 preflight doesn't need to load the KB just to
    answer "is this model Qwen-VL?".
    """
    import re as _re

    return bool(_re.match(r"Qwen/Qwen[2-9](\.[0-9]+)?-VL-", model_id))


def _likely_attention_heads(model_id: str) -> int | None:
    """Return the head count for well-known models without a network call.

    M3-scoped: this stays a small lookup table for the families the test
    corpus references. M4 plumbs an actual ``AutoConfig.from_pretrained``
    probe behind a cache so any HF model is covered.
    """
    table: dict[str, int] = {
        "meta-llama/Llama-3.1-8B-Instruct": 32,
        "meta-llama/Llama-3.1-70B-Instruct": 64,
        "meta-llama/Meta-Llama-3-8B-Instruct": 32,
        "Qwen/Qwen2.5-7B-Instruct": 28,
    }
    return table.get(model_id)


__all__ = ["eval"]
