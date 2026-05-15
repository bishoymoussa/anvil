"""``anvil`` CLI entrypoint (typer).

Subcommands:

* ``anvil eval`` — run an evaluation.
* ``anvil manifest verify | diff | replay | strip-caas`` — manifest tooling.
* ``anvil doctor`` — diagnose install / env / GPU.
* ``anvil version`` — print the installed version.

Every flag has a Python-API equivalent (design §10.3).
"""

from __future__ import annotations

import typer

from anvil._version import __version__
from anvil.cli.eval import app as eval_app
from anvil.cli.manifest import app as manifest_app
from anvil.cli.mcp import app as mcp_app
from anvil.cli.serve import app as serve_app

app = typer.Typer(
    name="anvil",
    help="Anvil — research-first, evaluation-first inference (see docs/design.md).",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(eval_app, name="eval", help="Run an evaluation.")
app.add_typer(
    manifest_app,
    name="manifest",
    help="Manifest tooling (verify/diff/replay/strip-caas).",
)
app.add_typer(
    serve_app,
    name="serve",
    help="Run the OpenAI-compatible HTTP server.",
)
app.add_typer(
    mcp_app,
    name="mcp",
    help="Start the Anvil MCP server (stdio or HTTP).",
)


@app.command()
def version() -> None:
    """Print the installed Anvil version."""
    typer.echo(__version__)


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", help="Emit results as JSON (machine-readable)."
    ),
) -> None:
    """Diagnose install / env / GPU issues.

    Exits 0 if every check is ``ok``, 1 if any check is ``warn``, 2 if any
    is ``fail``. Use ``--json`` to consume the report from CI or other tools.
    """
    from anvil.cli.doctor import overall_status, render_table, run_all_checks, to_json

    checks = run_all_checks()
    if json_output:
        typer.echo(to_json(checks))
    else:
        typer.echo(render_table(checks))

    status = overall_status(checks)
    if status == "fail":
        raise typer.Exit(code=2)
    if status == "warn":
        raise typer.Exit(code=1)


__all__ = ["app"]
