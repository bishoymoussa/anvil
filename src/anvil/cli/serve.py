"""``anvil serve`` CLI subcommand (design §10.3)."""

from __future__ import annotations

import typer

from anvil.exceptions import AnvilError

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback(invoke_without_command=True)
def serve_cmd(
    model: str = typer.Option(..., "--model", help="HF model id or local path."),
    port: int = typer.Option(8000, "--port"),
    host: str = typer.Option("0.0.0.0", "--host"),  # noqa: S104
    engine: str = typer.Option("auto", "--engine"),
    dtype: str | None = typer.Option(None, "--dtype"),
) -> None:
    """Run the OpenAI-compatible HTTP server."""
    from anvil.server import serve as _serve

    try:
        _serve(model=model, port=port, host=host, engine=engine, dtype=dtype)
    except AnvilError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


__all__ = ["app"]
