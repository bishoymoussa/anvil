"""KB YAML loader (design §7.3).

Walks ``src/anvil/caas/kb/entries/*.yaml``, parses each top-level list,
validates against :class:`KBEntry`, and returns the flat list. Duplicate
``id`` across files raises ``ConfigError`` — the registry is process-wide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from anvil.caas.kb.schema import KBEntry
from anvil.exceptions import ConfigError


def _entries_dir() -> Path:
    """The directory shipping the YAML KB. Resolved relative to this file."""
    return Path(__file__).resolve().parent / "entries"


def load_all(*, root: Path | None = None) -> list[KBEntry]:
    """Load every shipped KB entry, validated against the schema.

    Args:
        root: optional override of the entries directory (used by tests).

    Returns:
        Flat ``list[KBEntry]``, one per entry across all YAML files,
        sorted by ``id`` for deterministic ordering.

    Raises:
        ConfigError: on duplicate IDs, malformed YAML, or schema violations.
    """
    base = root if root is not None else _entries_dir()
    if not base.exists():
        raise ConfigError(f"KB entries directory not found: {base}")

    entries: list[KBEntry] = []
    seen: dict[str, str] = {}  # id → first source filename
    for yaml_path in sorted(base.glob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text())
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise ConfigError(
                f"{yaml_path.name}: top-level YAML must be a list of entries, "
                f"got {type(raw).__name__}"
            )
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ConfigError(f"{yaml_path.name}[{i}]: entry must be a mapping")
            try:
                entry = KBEntry.model_validate(item)
            except Exception as exc:  # pydantic ValidationError, etc.
                raise ConfigError(f"{yaml_path.name}[{i}]: invalid KB entry: {exc}") from exc
            if entry.id in seen:
                raise ConfigError(
                    f"duplicate KB id {entry.id!r}: "
                    f"first defined in {seen[entry.id]}, redefined in {yaml_path.name}"
                )
            seen[entry.id] = yaml_path.name
            entries.append(entry)
    entries.sort(key=lambda e: e.id)
    return entries


def load_by_id(entry_id: str) -> KBEntry:
    """Convenience: load all entries and return the one matching ``entry_id``."""
    for entry in load_all():
        if entry.id == entry_id:
            return entry
    raise ConfigError(f"no KB entry with id {entry_id!r}")


def _serialize_entry_for_round_trip(entry: KBEntry) -> dict[str, Any]:
    """Internal helper used by tests to round-trip an entry through YAML."""
    return entry.model_dump(mode="json", exclude_none=True)


__all__ = ["load_all", "load_by_id"]
