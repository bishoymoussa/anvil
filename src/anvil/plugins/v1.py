"""Plugin protocol v1 — discovery via entry points (design §5.3).

The agent's §16.12 decision was: namespaced entry points. Plugins ship a
``[project.entry-points."anvil.plugins.v1.<kind>"]`` block in their own
``pyproject.toml``; we discover them with :func:`importlib.metadata.entry_points`.

Plugins should NOT import any ``anvil.*`` path other than this module.
This is a *convention* (third-party packages live outside Anvil's repo so
import-linter can't enforce it from here); we honor it by re-exporting the
small set of registration helpers plugins legitimately need below.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any, Literal

from anvil.exceptions import PluginError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PluginKind = Literal["models", "tasks", "metrics", "engines", "kb_entries"]
_GROUP_PREFIX = "anvil.plugins.v1."


def discover(kind: PluginKind) -> Iterator[tuple[str, Any]]:
    """Yield ``(name, loaded_object)`` for every registered plugin of ``kind``.

    Errors during entry-point loading are wrapped in :class:`PluginError`
    with the plugin name so the user can identify the offender.
    """
    eps = entry_points(group=_GROUP_PREFIX + kind)
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as exc:  # noqa: BLE001
            raise PluginError(f"plugin {ep.name!r} (kind={kind}) failed to load: {exc}") from exc
        yield ep.name, obj


def register_model_impl(architecture: str) -> Callable[[type[Any]], type[Any]]:
    """Re-export of :func:`anvil.models.registry.register_model_impl`.

    Living on the plugin protocol module makes it importable from a plugin
    without crossing into Anvil's internals.
    """
    from anvil.models.registry import register_model_impl as _impl

    return _impl(architecture)


def register_request_type(name: str) -> Callable[[type[Any]], type[Any]]:
    """Reserved hook: future request types (e.g. ``Generate3D``) register here."""
    del name

    def _noop(cls: type[Any]) -> type[Any]:  # pragma: no cover - placeholder
        raise PluginError("register_request_type is reserved for v1; not callable in v0.")

    return _noop


__all__ = [
    "PluginKind",
    "discover",
    "register_model_impl",
    "register_request_type",
]
