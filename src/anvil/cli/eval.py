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
    tasks: str = typer.Option(..., "--tasks", help="Comma-separated task names (e.g. 'gsm8k')."),
    n_fewshot: int | None = typer.Option(
        None, "--n-fewshot", help="Number of few-shot examples (default: per-task)."
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Cap docs per task (default: full split)."
    ),
    engine: str = typer.Option(
        "auto", "--engine", help="Engine: auto | hf | vllm (vllm lands in M1)."
    ),
    dtype: str | None = typer.Option(
        None, "--dtype", help="bfloat16 | float16 | float32 | (auto)."
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Write manifest.json to this directory (created if missing).",
    ),
) -> None:
    """Run an evaluation. Writes ``manifest.json`` if ``--output`` is given."""
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    if not task_list:
        typer.echo("error: --tasks must be a non-empty comma-separated list", err=True)
        raise typer.Exit(code=2)

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


__all__ = ["app", "eval_cmd"]
