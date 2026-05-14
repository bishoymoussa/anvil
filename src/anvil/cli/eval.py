"""``anvil eval`` subcommand — drives :func:`anvil.eval`."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — typer reads runtime annotations
from typing import TYPE_CHECKING

import typer

from anvil.exceptions import AnvilError
from anvil.logging import get_logger
from anvil.tasks.public import eval as anvil_eval

if TYPE_CHECKING:
    from anvil.tasks.base import Task as _Task

_log = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def eval_cmd(
    model: str = typer.Option(..., "--model", help="HF model id or local path."),
    tasks: str = typer.Option(
        "",
        "--tasks",
        help="Comma-separated Anvil-curated task names (e.g. 'mmlu,gsm8k').",
    ),
    lm_eval_tasks: str = typer.Option(
        "",
        "--lm-eval-tasks",
        help="Comma-separated lm-evaluation-harness YAML paths or task names "
        "(see anvil.tasks.lm_eval_shim).",
    ),
    n_fewshot: int | None = typer.Option(
        None, "--n-fewshot", help="Number of few-shot examples (default: per-task)."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Cap docs per task (default: full split)."
    ),
    engine: str = typer.Option("auto", "--engine", help="Engine: auto | hf | vllm."),
    dtype: str | None = typer.Option(
        None, "--dtype", help="bfloat16 | float16 | float32 | (auto)."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write manifest.json to this directory (created if missing).",
    ),
    compare_with_lm_eval: bool = typer.Option(
        False,
        "--compare-with-lm-eval",
        help="Also run lm-evaluation-harness against the same tasks and emit a "
        "per-metric delta report. Requires lm_eval to be installed.",
    ),
) -> None:
    """Run an evaluation. Writes ``manifest.json`` if ``--output`` is given."""
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    yaml_list = [t.strip() for t in lm_eval_tasks.split(",") if t.strip()]
    if not task_list and not yaml_list:
        typer.echo(
            "error: pass at least one of --tasks or --lm-eval-tasks",
            err=True,
        )
        raise typer.Exit(code=2)

    # Compile any --lm-eval-tasks entries up-front and merge into the run.
    # Accepts either a YAML file path or a bare lm-eval task name (resolved
    # from the installed lm-eval catalog via lm_eval.tasks.get_task_dict).
    if yaml_list:
        from anvil.tasks.lm_eval_shim import compile_yaml

        for spec in yaml_list:
            spec_path = Path(spec)
            if spec_path.exists():
                compiled = compile_yaml(spec_path)
                task_list.append(compiled.name)
            else:
                # Try the lm-eval task catalog by name.
                resolved = _resolve_lm_eval_task_by_name(spec)
                if resolved is None:
                    typer.echo(
                        f"error: {spec!r} is not a YAML path and was not found in the "
                        "lm-evaluation-harness task catalog. Install lm_eval and check "
                        "the task name with `lm_eval --tasks list`.",
                        err=True,
                    )
                    raise typer.Exit(code=2)
                task_list.append(resolved.name)

    try:
        result = anvil_eval(
            model=model,
            tasks=task_list,
            n_fewshot=n_fewshot,
            limit=limit,
            engine=engine,
            dtype=dtype,
            output_dir=output,
        )
    except AnvilError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for task_name, metrics in result.scores.items():
        for metric_name, value in metrics.items():
            typer.echo(f"{task_name}.{metric_name}: {value:.4f}")

    if output:
        typer.echo(f"manifest written to {output / 'manifest.json'}")

    if compare_with_lm_eval:
        from anvil.tasks.lm_eval_shim import compare_with_lm_eval as _compare

        try:
            results = _compare(
                model=model,
                tasks=task_list,
                limit=limit,
                n_fewshot=n_fewshot or 0,
            )
        except AnvilError as exc:
            typer.echo(f"compare error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        typer.echo("\n--compare-with-lm-eval delta report:")
        for r in results:
            typer.echo(f"  {r.render()}")


def _resolve_lm_eval_task_by_name(name: str) -> type[_Task] | None:
    """Compile an lm-eval task by catalog name into an Anvil Task class.

    Tries ``lm_eval.tasks.get_task_dict`` (the standard programmatic API for
    lm-evaluation-harness ≥ 0.4). Returns ``None`` if lm_eval is not
    installed or the task name is unknown.
    """
    try:
        import lm_eval.tasks as _lm_tasks
    except ImportError:
        return None

    try:
        task_dict = _lm_tasks.get_task_dict([name])
    except Exception:  # noqa: BLE001 — lm_eval raises various errors for unknown tasks
        return None

    if not task_dict:
        return None

    # ``get_task_dict`` returns ``{name: TaskGroup | Task}``. Retrieve the
    # underlying config (a dict-like object) and compile it via our shim.
    task_obj = task_dict.get(name)
    if task_obj is None:
        return None

    from anvil.tasks.lm_eval_shim import compile_yaml_dict

    # lm_eval Task objects expose their config as a dict via .config or
    # .task_config depending on the version. Build a minimal YAML-shaped
    # dict that the compiler understands.
    cfg: dict[str, object] = {}
    for attr in ("config", "task_config", "_config"):
        raw = getattr(task_obj, attr, None)
        if raw is not None:
            cfg = dict(raw) if not isinstance(raw, dict) else dict(raw)
            break

    if not cfg:
        return None

    cfg.setdefault("task", name)
    try:
        return compile_yaml_dict(cfg)
    except Exception:  # noqa: BLE001
        return None


__all__ = ["app", "eval_cmd"]
