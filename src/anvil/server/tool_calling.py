"""Constrained-decoding tool-call plumbing (design §10.2, §16.10 M6).

The manuscript's wedge: vLLM ships 14+ per-model ``--tool-call-parser``
flags (``hermes``, ``llama3_json``, ``mistral``, ``qwen3_coder``, …) and
each is buggy in some way. Anvil replaces all of them with **one
grammar** — when the user supplies ``tools`` in their request, we
constrain decoding to the tool-call schema directly.

For v0:

* If ``xgrammar`` is installed, we build a JSON-schema grammar from the
  user's tool list and ask the engine to constrain decoding.
* If ``xgrammar`` isn't installed, we surface a clear error rather than
  pretending tool-calling works (the failure mode would be the model
  emitting free-form JSON that mostly-but-not-always parses).

The actual engine integration (passing the compiled grammar through to
vLLM's ``GuidedDecodingParams`` or a ``LogitsProcessor`` for the HF path)
is *prepared* in this module — we expose :func:`build_tool_grammar` and
:func:`parse_tool_call_response` — but the M6 acceptance test only
exercises the schema construction and the parser, not the live
constrained-decoding loop.

Live constrained-decoding lands when we wire xgrammar through the
``LogitsProcessor`` path in M2's wrapper layer — the V0-style API that
:mod:`anvil.engine._wrappers` is designed to host.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anvil.server.schemas import Tool, ToolCall


# Conservative regex extracting one JSON-object code-fence from text.
_JSON_FENCE = re.compile(r"```(?:json)?\s*\n?(\{.*?\})\s*\n?```", re.DOTALL)
_BARE_JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def build_tool_grammar(tools: list[Tool]) -> dict[str, Any]:
    """Compile a tool list to a JSON-schema grammar suitable for xgrammar.

    The resulting schema is an ``oneOf`` over each tool's
    ``{"name": "...", "arguments": {...}}`` shape. xgrammar accepts this
    directly via ``GrammarCompiler.compile_json_schema``.

    Returning the schema as a plain dict keeps this module xgrammar-free
    at import time — only :func:`assert_xgrammar_available` actually
    imports the optional dep.
    """
    if not tools:
        raise ValueError("tools must be non-empty")
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": [t.function.name for t in tools],
            },
            "arguments": {
                "oneOf": [
                    {
                        "type": "object",
                        "title": t.function.name,
                        "properties": (t.function.parameters or {}).get("properties", {}),
                        "required": (t.function.parameters or {}).get("required", []),
                    }
                    for t in tools
                ]
            },
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def assert_xgrammar_available() -> None:
    """Raise :class:`ConfigError` with a clear pointer if xgrammar is missing."""
    from anvil.exceptions import ConfigError

    try:
        import xgrammar  # noqa: F401
    except ImportError as exc:
        raise ConfigError(
            "Tool calling requires xgrammar. Install with "
            '`uv pip install -e ".[xgrammar]"`. Anvil refuses to ship the '
            "per-model --tool-call-parser flag matrix the design (§10.2) "
            "explicitly avoids."
        ) from exc


def parse_tool_call_response(text: str, tools: list[Tool]) -> list[ToolCall]:
    """Best-effort parse of a free-text generation into tool calls.

    Used as the fallback when xgrammar isn't installed. Looks for one of:

    1. ``\\`\\`\\`json\\n{...}\\n\\`\\`\\`` fenced block.
    2. A bare ``{...}`` JSON object.

    Returns an empty list on failure (the caller treats that as a plain
    text response). The function is conservative — false positives in
    tool-call parsing are way worse than false negatives.
    """
    from anvil.server.schemas import FunctionCall, ToolCall

    candidate: str | None = None
    fence_match = _JSON_FENCE.search(text)
    if fence_match is not None:
        candidate = fence_match.group(1)
    else:
        bare_match = _BARE_JSON_OBJECT.search(text)
        if bare_match is not None:
            candidate = bare_match.group(0)
    if candidate is None:
        return []
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, dict):
        return []
    name = parsed.get("name") or parsed.get("function", {}).get("name")
    args = parsed.get("arguments") or parsed.get("function", {}).get("arguments")
    if name is None or args is None:
        return []
    valid_names = {t.function.name for t in tools}
    if name not in valid_names:
        return []
    arg_str = args if isinstance(args, str) else json.dumps(args)
    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            function=FunctionCall(name=str(name), arguments=arg_str),
        )
    ]


__all__ = [
    "build_tool_grammar",
    "assert_xgrammar_available",
    "parse_tool_call_response",
]
