"""Task registry — name → Task class (design §6).

Tasks register at import time with a class decorator or by passing a
:class:`anvil.tasks.base.Task` subclass to :func:`register_task`. The
built-in tasks register themselves on first import of
:mod:`anvil.tasks.builtin`.
"""

from __future__ import annotations

from typing import TypeVar

from anvil.exceptions import ConfigError
from anvil.tasks.base import Task

T = TypeVar("T", bound=type[Task])

_REGISTRY: dict[str, type[Task]] = {}


def register_task(task_cls: type[Task]) -> type[Task]:
    """Register a Task subclass under its :attr:`Task.name`.

    Idempotent: re-registering an identical class is a no-op; re-registering
    a *different* class under the same name raises ``ConfigError``.
    """
    name = getattr(task_cls, "name", None)
    if not name:
        raise ConfigError(f"register_task: {task_cls.__qualname__} has no `name` class attribute")
    if name in _REGISTRY and _REGISTRY[name] is not task_cls:
        raise ConfigError(
            f"task {name!r} already registered as "
            f"{_REGISTRY[name].__qualname__}; cannot replace with {task_cls.__qualname__}"
        )
    _REGISTRY[name] = task_cls
    return task_cls


def get_task(name: str) -> type[Task]:
    """Return the registered Task class for ``name``.

    Raises:
        ConfigError: if no task with that name is registered. The error lists
            the available names so the user can spot a typo.
    """
    if name not in _REGISTRY:
        # Trigger import of built-in tasks so a fresh process can resolve them.
        import contextlib

        with contextlib.suppress(ImportError):
            import anvil.tasks.builtin  # noqa: F401
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ConfigError(f"unknown task {name!r}. Registered tasks: {available}")
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    """Return the names of all registered tasks, sorted."""
    import contextlib

    with contextlib.suppress(ImportError):
        import anvil.tasks.builtin  # noqa: F401
    return sorted(_REGISTRY)


__all__ = ["register_task", "get_task", "list_tasks"]
