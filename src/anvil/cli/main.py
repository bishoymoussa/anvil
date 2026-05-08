"""``anvil`` CLI entrypoint (typer).

Subcommands:

* ``anvil eval`` — run an evaluation (M0+).
* ``anvil manifest verify | diff | replay`` — manifest tooling (M2+; stubs raise).
* ``anvil caas test | list-known-issues`` — CaaS tooling (M3+; stubs raise).
* ``anvil doctor`` — diagnose env / install (M6+; stubs raise).

Every flag has a Python-API equivalent (design §10.3).
"""

from __future__ import annotations

import typer

from anvil._version import __version__
from anvil.cli.eval import app as eval_app
from anvil.cli.manifest import app as manifest_app

app = typer.Typer(
    name="anvil",
    help="Anvil — research-first, evaluation-first inference (see docs/design.md).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(eval_app, name="eval", help="Run an evaluation.")
app.add_typer(manifest_app, name="manifest", help="Manifest tooling (verify/diff/replay).")


@app.command()
def version() -> None:
    """Print the installed Anvil version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Diagnose install / env / GPU issues (M6 work)."""
    raise typer.Exit(
        code=2,
    )


__all__ = ["app"]
