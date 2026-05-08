"""``anvil eval`` subcommand — drives :func:`anvil.eval`."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — typer reads runtime annotations

import typer

from anvil.exceptions import AnvilError
from anvil.logging import get_logger
from anvil.tasks.public import eval as anvil_eval

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

    # Compile any --lm-eval-tasks YAML paths up-front and merge into the run.
    if yaml_list:
        from anvil.tasks.lm_eval_shim import compile_yaml

        for spec in yaml_list:
            spec_path = Path(spec)
            if not spec_path.exists():
                typer.echo(
                    f"error: --lm-eval-tasks expects YAML paths; {spec!r} does not exist. "
                    "(Looking up by lm-eval task name lands in v0.5.)",
                    err=True,
                )
                raise typer.Exit(code=2)
            compiled = compile_yaml(spec_path)
            task_list.append(compiled.name)

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


__all__ = ["app", "eval_cmd"]
