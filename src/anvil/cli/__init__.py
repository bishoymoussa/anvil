"""``anvil`` CLI — thin wrapper over the Python API (design §10.3, §16.9).

Every flag has a Python-API equivalent. The CLI is deliberately small;
configuration is a YAML file or env vars when things get complex.
"""

from __future__ import annotations

from anvil.cli.main import app

__all__ = ["app"]
