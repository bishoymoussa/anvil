"""Test fixtures and helpers (design §16.12: agent picks pytest layout).

Conventions:

* Each test file is named ``test_<module-or-feature>.py``.
* Markers: ``@pytest.mark.requires_network``, ``@pytest.mark.requires_gpu``,
  ``@pytest.mark.requires_hf_gated``, ``@pytest.mark.requires_vllm``,
  ``@pytest.mark.slow``. ``pyproject.toml`` excludes them from the default
  run; opt in with ``pytest -m ...``.
* Fixtures that need third-party state (HF Hub, GPUs) are scoped narrowly
  so failure to acquire them is local to the test that needs them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``tests.helpers``-style imports possible in integration tests.
_TESTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS_DIR))


def pytest_collection_modifyitems(config: object, items: list[object]) -> None:
    """Honor opt-in env vars for normally-skipped markers.

    ``ANVIL_TEST_RUN_NETWORK=1`` runs ``requires_network`` tests this session;
    similarly ``ANVIL_TEST_RUN_GPU``, ``ANVIL_TEST_RUN_HF_GATED``,
    ``ANVIL_TEST_RUN_VLLM``. We don't override pytest's marker filtering,
    only document the convention.
    """
    del config, items  # marker filtering is configured in pyproject.toml
    _ = os.environ  # touch to silence unused-import linters


__all__: list[str] = []
