"""``anvil mcp`` — start the Anvil MCP server."""

from __future__ import annotations

import typer

app = typer.Typer(help="Start the Anvil MCP server.")


@app.callback(invoke_without_command=True)
def run(
    http: bool = typer.Option(
        False,
        "--http",
        help="Use streamable-HTTP transport instead of stdio. Listens on localhost:8765.",
    ),
) -> None:
    """Start the Anvil MCP server.

    Default transport is stdio (for Claude Desktop / Claude Code).
    Pass --http to expose a streamable-HTTP endpoint instead.

    \b
    Add to claude_desktop_config.json:
        {
          "mcpServers": {
            "anvil": { "command": "anvil", "args": ["mcp"] }
          }
        }
    """
    try:
        from anvil.mcp_server import mcp
    except ImportError as exc:
        typer.echo(
            f"Error: MCP extra not installed. Run: pip install 'anvil-eval[mcp]'\n({exc})",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    if http:
        typer.echo("Starting Anvil MCP server (HTTP) on localhost:8765 …")
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


__all__ = ["app"]
