"""Tests for the task registry (design §6)."""

from __future__ import annotations

import pytest

from anvil.exceptions import ConfigError
from anvil.primitives.request import Generate
from anvil.tasks import register_task
from anvil.tasks.base import Task
from anvil.tasks.registry import _REGISTRY, get_task, list_tasks


class _FakeTask(Task):
    name = "_fake_unique_task"
    dataset = "fake/ds"

    def doc_to_request(self, doc: dict) -> Generate:  # type: ignore[type-arg]
        return Generate(prompt=str(doc.get("q", "")))

    def request_to_prediction(self, response: object, doc: dict) -> str:  # type: ignore[type-arg]
        return getattr(response, "text", "")


def teardown_function() -> None:
    """Cleanup after each test so registration is local."""
    _REGISTRY.pop("_fake_unique_task", None)
    _REGISTRY.pop("_fake_unique_task_2", None)


class TestRegistry:
    def test_register_and_get(self) -> None:
        register_task(_FakeTask)
        assert get_task("_fake_unique_task") is _FakeTask

    def test_register_idempotent(self) -> None:
        register_task(_FakeTask)
        register_task(_FakeTask)  # second call is a no-op
        assert get_task("_fake_unique_task") is _FakeTask

    def test_register_conflict(self) -> None:
        register_task(_FakeTask)

        class _Other(_FakeTask):
            name = "_fake_unique_task"

        with pytest.raises(ConfigError, match="already registered"):
            register_task(_Other)

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(ConfigError, match="unknown task"):
            get_task("definitely_not_a_real_task")

    def test_list_tasks_includes_builtins(self) -> None:
        names = list_tasks()
        # M0 builtins: at least gsm8k.
        assert "gsm8k" in names

    def test_register_task_without_name(self) -> None:
        class _Bad(_FakeTask):
            name = ""

        with pytest.raises(ConfigError, match="no `name`"):
            register_task(_Bad)
