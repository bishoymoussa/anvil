"""``anvil manifest`` subcommand. M0 ships ``verify`` only; ``diff`` and
``replay`` are full features in M2 (design §16.10)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — typer reads runtime annotations

import typer

from anvil.exceptions import AnvilError, ManifestError
from anvil.manifest.schema import Manifest

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def verify(path: Path = typer.Argument(..., help="Path to manifest.json.")) -> None:
    """Verify a manifest's signature."""
    try:
        m = Manifest.load(path)
    except AnvilError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    if not m.manifest_signature:
        typer.echo("UNSIGNED")
        raise typer.Exit(code=2)
    if m.verify():
        typer.echo("OK")
    else:
        typer.echo("MISMATCH")
        raise typer.Exit(code=2)


@app.command()
def diff(
    a: Path = typer.Argument(..., help="First manifest."),
    b: Path = typer.Argument(..., help="Second manifest."),
) -> None:
    """Diff two manifests; print every score-affecting delta."""
    ma = Manifest.load(a)
    mb = Manifest.load(b)
    delta = Manifest.diff(ma, mb)
    if not delta:
        typer.echo("(identical, modulo timestamps and signature)")
        return
    for path, (av, bv) in delta.items():
        typer.echo(f"{path}: {av!r} → {bv!r}")


@app.command()
def replay(path: Path = typer.Argument(..., help="Path to manifest.json.")) -> None:
    """Replay an evaluation from its manifest. (M2 work — surfaces the gap.)"""
    del path
    raise ManifestError(
        "anvil manifest replay is M2 work (design §16.10). It will reconstruct "
        "the run from the manifest's pinned model revision, sampler, and tasks."
    )


__all__ = ["app"]
