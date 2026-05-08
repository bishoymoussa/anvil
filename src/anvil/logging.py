"""Structured logging configuration (design manuscript §16.9).

Two modes:

* **TTY**: human-readable, rich-formatted, colored.
* **Non-TTY** (CI, files, pipes): one JSON object per line on stderr, suitable for
  log aggregation.

The choice is made by ``ANVIL_LOG_FORMAT`` (``human`` | ``json`` | ``auto``,
default ``auto``). Inside Anvil code, never use ``print``: get a logger via
``anvil.logging.get_logger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

_INITIALIZED = False


class _JsonFormatter(logging.Formatter):
    """Each record becomes one JSON object on a line on stderr."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup(level: int | str | None = None, *, force: bool = False) -> None:
    """Configure Anvil logging exactly once.

    Args:
        level: log level (string or stdlib int). Falls back to
            ``ANVIL_LOG_LEVEL`` env var, then ``INFO``.
        force: re-initialize even if ``setup`` already ran (useful in tests).
    """
    global _INITIALIZED
    if _INITIALIZED and not force:
        return

    resolved_level = level if level is not None else os.getenv("ANVIL_LOG_LEVEL", "INFO")
    if isinstance(resolved_level, str):
        resolved_level = logging.getLevelNamesMapping().get(resolved_level.upper(), logging.INFO)

    fmt = os.getenv("ANVIL_LOG_FORMAT", "auto").lower()
    use_json = fmt == "json" or (fmt == "auto" and not sys.stderr.isatty())

    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    root = logging.getLogger("anvil")
    root.setLevel(resolved_level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``anvil`` namespace.

    ``setup`` is invoked lazily on first use.
    """
    setup()
    if name.startswith("anvil"):
        return logging.getLogger(name)
    return logging.getLogger(f"anvil.{name}")


__all__ = ["setup", "get_logger"]
