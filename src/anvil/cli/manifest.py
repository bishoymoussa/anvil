"""``anvil manifest`` subcommand. M0 ships ``verify`` only; ``diff`` and
``replay`` are full features in M2 (design §16.10)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — typer reads runtime annotations

import typer

from anvil.exceptions import AnvilError
from anvil.manifest import (
    Manifest,
    diff_entries,
    strip_caas,
)
from anvil.manifest import (
    replay as _replay,
)

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
    """Diff two manifests; severity-tagged so reviewers eyeball ``critical`` first."""
    ma = Manifest.load(a)
    mb = Manifest.load(b)
    entries = diff_entries(ma, mb)
    if not entries:
        typer.echo("(identical, modulo timestamps and signature)")
        return
    for entry in entries:
        typer.echo(entry.render())


@app.command()
def replay(
    path: Path = typer.Argument(..., help="Path to manifest.json."),
    output: Path | None = typer.Option(
        None, "--output", help="Write the replayed manifest to this directory."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if scores differ from the original."
    ),
) -> None:
    """Re-run an evaluation from its manifest.

    Reconstructs the run config (model, sampler, tasks, fewshot count) and
    re-executes against the same engine. With ``--strict``, any score
    divergence exits non-zero — useful for CI checking that a manifest
    still reproduces.
    """
    try:
        result = _replay(path, output_dir=output, strict=strict)
    except AnvilError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    for task_name, metrics in result.scores.items():
        for metric_name, value in metrics.items():
            typer.echo(f"{task_name}.{metric_name}: {value:.4f}")
    if output:
        typer.echo(f"manifest written to {output / 'manifest.json'}")


@app.command("strip-caas")
def strip_caas_cmd(
    path: Path = typer.Argument(..., help="Path to manifest.json."),
    output: Path | None = typer.Option(
        None, "--output", help="Where to write the stripped manifest (default: stdout)."
    ),
) -> None:
    """Produce a frozen-config rerun spec by clearing the CaaS log (§8.3).

    The output is **unsigned** — the user is expected to re-run with the
    original config and let signing happen on the new run, not rebadge the
    stripped manifest as authoritative.
    """
    m = Manifest.load(path)
    stripped = strip_caas(m)
    if output is None:
        typer.echo(stripped.canonical_json())
    else:
        output.write_text(stripped.canonical_json(), encoding="utf-8")
        typer.echo(f"stripped manifest written to {output}")


__all__ = ["app"]
