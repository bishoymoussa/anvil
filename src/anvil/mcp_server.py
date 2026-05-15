"""Anvil MCP server (design §10.4).

Exposes Anvil's core capabilities as MCP tools so any MCP-compatible
research agent (Claude Desktop, Claude Code, Cursor, etc.) can drive
evaluations, inspect manifests, and query the environment through natural
language.

Usage:
    anvil mcp          # stdio transport (default — for Claude Desktop / Code)
    anvil mcp --http   # streamable-HTTP transport on localhost:8765

Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "anvil": { "command": "anvil", "args": ["mcp"] }
      }
    }

Tools exposed:
    anvil_list_tasks     — list registered benchmark tasks with metadata.
    anvil_eval           — run a model on one or more tasks; returns scores.
    anvil_manifest_diff  — diff two saved manifests; explains score gaps.
    anvil_manifest_verify — verify a manifest's content hash.
    anvil_doctor         — environment diagnosis (CUDA, HF token, vLLM …).
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="anvil",
    instructions=(
        "Anvil is a research-first LLM evaluation library. "
        "Use anvil_list_tasks to discover benchmarks, anvil_eval to run them, "
        "anvil_manifest_diff to compare runs, and anvil_doctor to diagnose issues."
    ),
    port=8765,
)


# ------------------------------------------------------------------ tools


@mcp.tool()
def anvil_list_tasks() -> str:
    """List all registered Anvil benchmark tasks with their metadata.

    Returns a JSON array of objects with keys: name, dataset, tier,
    request_type, n_fewshot_default, metric_name.
    """
    from anvil.tasks.registry import _REGISTRY, list_tasks

    names = list_tasks()
    out: list[dict[str, Any]] = []
    for name in names:
        cls = _REGISTRY[name]
        out.append(
            {
                "name": name,
                "dataset": str(getattr(cls, "dataset", "unknown")),
                "tier": getattr(cls, "tier", "custom"),
                "request_type": getattr(cls, "request_type", "Generate"),
                "n_fewshot_default": getattr(cls, "n_fewshot_default", 0),
                "metric_name": getattr(cls, "metric_name", "accuracy"),
            }
        )
    return json.dumps(out, indent=2)


@mcp.tool()
def anvil_eval(
    model: str,
    tasks: str,
    n_fewshot: int | None = None,
    limit: int | None = None,
    output: str | None = None,
) -> str:
    """Run an Anvil evaluation and return scores.

    Args:
        model: HuggingFace model ID or local path (e.g. 'meta-llama/Llama-3.1-8B-Instruct').
        tasks: comma-separated task names (e.g. 'mmlu,gsm8k').
        n_fewshot: number of few-shot examples (overrides each task's default).
        limit: max docs per task (useful for quick smoke-tests).
        output: optional path to save the manifest JSON.

    Returns:
        JSON object with keys 'scores' (dict per task) and 'manifest_path' (if saved).
    """
    import anvil

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    try:
        result = anvil.eval(
            model=model,
            tasks=task_list,
            n_fewshot=n_fewshot,
            limit=limit,
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})

    manifest_path: str | None = None
    if output:
        try:
            result.manifest.save(output)
            manifest_path = output
        except Exception as exc:  # noqa: BLE001
            manifest_path = f"(save failed: {exc})"

    return json.dumps(
        {
            "scores": result.scores,
            "manifest_path": manifest_path,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def anvil_manifest_diff(path_a: str, path_b: str) -> str:
    """Diff two Anvil manifests and explain what changed between runs.

    Args:
        path_a: path to the first (baseline) manifest JSON.
        path_b: path to the second (comparison) manifest JSON.

    Returns:
        Human-readable diff including score deltas, config changes, and
        any fields that differ between the two runs.
    """
    from anvil.manifest.diff import diff_entries
    from anvil.manifest.schema import Manifest

    try:
        a = Manifest.load(path_a)
        b = Manifest.load(path_b)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"could not load manifests: {exc}"})

    try:
        entries = diff_entries(a, b)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"diff failed: {exc}"})

    if not entries:
        return json.dumps({"identical": True, "message": "No differences found."})

    return json.dumps(
        [{"path": e.path, "severity": e.severity, "a": e.a, "b": e.b} for e in entries],
        indent=2,
        default=str,
    )


@mcp.tool()
def anvil_manifest_verify(path: str) -> str:
    """Verify that an Anvil manifest's content hash is intact.

    Args:
        path: path to the manifest JSON file.

    Returns:
        JSON with keys 'valid' (bool), 'manifest_id', and 'message'.
    """
    from anvil.manifest.schema import Manifest

    try:
        manifest = Manifest.load(path)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"valid": False, "message": f"could not load: {exc}"})

    try:
        manifest.verify()
        return json.dumps(
            {
                "valid": True,
                "model": manifest.model.id
                if hasattr(manifest.model, "id")
                else str(manifest.model),
                "message": "Manifest is intact — content hash verified.",
            }
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {
                "valid": False,
                "message": str(exc),
            }
        )


@mcp.tool()
def anvil_doctor() -> str:
    """Diagnose the Anvil environment: CUDA, HF token, vLLM, disk space, etc.

    Returns:
        JSON array of check results, each with 'name', 'status' (ok/warn/fail),
        and 'message'. Overall status is the worst across all checks.
    """
    from anvil.cli.doctor import overall_status, run_all_checks

    checks = run_all_checks()
    return json.dumps(
        {
            "overall": overall_status(checks),
            "checks": [c.to_dict() for c in checks],
        },
        indent=2,
    )


__all__ = ["mcp"]
