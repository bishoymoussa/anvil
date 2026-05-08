"""OpenAI-compatible HTTP server (design §10.2, §16.10 M6).

Drop-in replacement for the OpenAI Python SDK base URL: point your
client at ``http://localhost:8000`` and the existing
``client.chat.completions.create(model="...", messages=[...])`` works.

Constrained-decoding tool calls (§10.2): when the request supplies
``tools`` and ``tool_choice="auto"``, Anvil routes through xgrammar
(if installed) to enforce the tool-call schema by construction —
removing the need for the per-model ``--tool-call-parser`` flag the
manuscript catalogues as a vLLM pain point. xgrammar isn't a hard
dependency; without it the server still answers chat completions but
rejects ``tools`` arguments with a clear error.

For development and small-scale eval. Production-scale serving — autoscaling,
P/D disaggregation, expert parallelism — is explicitly out of scope (§12).
"""

from __future__ import annotations

from anvil.server.app import build_app
from anvil.server.serve import serve

__all__ = ["serve", "build_app"]
