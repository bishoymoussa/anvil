"""M0 acceptance test (design §16.10).

The manuscript's literal acceptance test runs:

    anvil.eval(model="meta-llama/Llama-3.1-8B-Instruct",
               tasks=["gsm8k"], n_fewshot=5, limit=50, output_dir=tmp_path)

That requires (a) HF_TOKEN with Llama gating, (b) ~16GB to download, (c)
~10 min on a 4090. We split it into two:

* :func:`test_milestone_0_end_to_end_mocked` — the **mock** variant. Uses
  a :class:`StubEngine` and a programmatic dataset whose last numbers are
  the gold answers. Runs in <100 ms with no model download. Asserts:
  scores in (0.4, 0.95), manifest exists, manifest validates, model id and
  fewshot count round-trip through the manifest.

* :func:`test_milestone_0_end_to_end_real_llama` — the **real** variant
  from the manuscript. Marked ``requires_hf_gated`` so it's skipped by
  default. Run with ``pytest -m requires_hf_gated`` and an authenticated
  HF token to validate the milestone live.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

import anvil
from anvil.tasks.builtin.gsm8k import GSM8K
from anvil.tasks.runner import run_eval
from helpers import StubEngine
from helpers.datasets import tiny_gsm8k

if TYPE_CHECKING:
    from pathlib import Path


def test_milestone_0_end_to_end_mocked(tmp_path: Path) -> None:
    """M0 mocked: pipeline runs end-to-end without a real model."""

    # Subclass GSM8K to swap its dataset for our programmatic fixture.
    class _TinyGSM8K(GSM8K):
        name = "_tiny_gsm8k_for_m0"
        dataset = staticmethod(tiny_gsm8k)

    task = _TinyGSM8K(n_fewshot=0, limit=5)
    engine = StubEngine(model_id="stub/llama-like")

    result = run_eval(engine=engine, tasks=[task], output_dir=tmp_path)

    # Score is in the manuscript's accepted range (0.40, 0.95).
    score = result.scores[task.name]["accuracy"]
    assert 0.40 < score < 0.95 + 1e-9, f"score {score} outside accepted range"

    # Manifest exists, parses, verifies.
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists(), "manifest.json was not written"
    m = anvil.Manifest.model_validate_json(manifest_path.read_text())
    assert m.verify(), "manifest signature failed to verify"

    # Model id, task name, fewshot count round-trip through the manifest.
    assert m.model.id == "stub/llama-like"
    assert m.tasks[0].name == task.name
    assert m.tasks[0].n_fewshot == 0
    assert m.tasks[0].metric == "accuracy"
    assert m.tasks[0].request_type == "Generate"
    assert m.anvil_version == anvil.__version__


@pytest.mark.requires_hf_gated
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_milestone_0_end_to_end_real_llama(tmp_path: Path) -> None:
    """The literal §16.10 test. Runs only with HF_TOKEN + GPU.

    Run with: ``pytest -m requires_hf_gated tests/integration/test_milestone_0.py``
    """
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set; cannot fetch gated Llama-3 weights")

    result = anvil.eval(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tasks=["gsm8k"],
        n_fewshot=5,
        limit=50,
        output_dir=tmp_path,
    )

    # Per §16.10:
    #   assert "gsm8k" in result.scores
    #   assert 0.40 < result.scores["gsm8k"]["accuracy"] < 0.95
    assert "gsm8k" in result.scores
    score = result.scores["gsm8k"]["accuracy"]
    assert 0.40 < score < 0.95, f"GSM8K score {score} outside published-baseline range"

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    m = anvil.Manifest.model_validate_json(manifest_path.read_text())
    assert m.model.id == "meta-llama/Llama-3.1-8B-Instruct"
    assert m.tasks[0].name == "gsm8k"
    assert m.tasks[0].n_fewshot == 5
