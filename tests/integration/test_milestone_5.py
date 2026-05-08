"""M5 acceptance tests (design §16.10).

Three literal manuscript tests:

1. ``test_milestone_5_lm_eval_yaml_imports`` — import an existing
   lm-eval task YAML and run it. Mocked to use the StubEngine + a
   programmatic dataset; the live variant against real Llama is marked
   ``requires_hf_gated``.
2. ``test_milestone_5_compare_with_lm_eval`` — Anvil score within 1pp of
   lm-evaluation-harness on the same task. Marked ``requires_lm_eval``
   (the harness itself isn't a hard dep) + ``requires_hf_gated``.
3. ``test_milestone_5_custom_modality_rna`` — the §6.7 RNA example.
   Mocked to use the StubEngine's deterministic embed; live variant
   against ``multimolecule/rnafm`` marked ``requires_network``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from anvil.exceptions import ConfigError
from anvil.primitives.request import Embed
from anvil.tasks.base import MultipleChoice, Task
from anvil.tasks.lm_eval_shim import compile_yaml_dict
from anvil.tasks.registry import _REGISTRY
from anvil.tasks.runner import run_eval
from helpers import StubEngine

if TYPE_CHECKING:
    from pathlib import Path

    pass


def teardown_function() -> None:
    """Drop test-only registrations."""
    for name in list(_REGISTRY):
        if name.startswith("_"):
            _REGISTRY.pop(name, None)


# ---------------------------------------------------------------- offline


def _arc_like_dataset() -> list[dict[str, Any]]:
    """A 4-doc programmatic ARC-Challenge-shaped dataset."""
    return [
        {
            "question": "Q1?",
            "choices": ["W", "X", "Y", "Z"],
            "answerKey": "A",
            "subject": "elementary",
        },
        {
            "question": "Q2?",
            "choices": ["W", "X", "Y", "Z"],
            "answerKey": "B",
            "subject": "elementary",
        },
        {
            "question": "Q3?",
            "choices": ["W", "X", "Y", "Z"],
            "answerKey": "C",
            "subject": "elementary",
        },
        {
            "question": "Q4?",
            "choices": ["W", "X", "Y", "Z"],
            "answerKey": "D",
            "subject": "elementary",
        },
    ]


def test_milestone_5_lm_eval_yaml_imports_offline(tmp_path: Path) -> None:
    """Compile an lm-eval-shaped spec, run it through Anvil's runner with
    the StubEngine, assert tier='imported' lands in the manifest."""
    spec: dict[str, Any] = {
        "task": "_arc_challenge_offline",
        "dataset_path": "ai2_arc",
        "dataset_name": "ARC-Challenge",
        "output_type": "multiple_choice",
        "test_split": "test",
        "doc_to_text": "Question: {question}\nAnswer: {answerKey}",
        "doc_to_target": "answerKey",
        "doc_to_choice": "choices",
    }
    compiled = compile_yaml_dict(spec)
    assert issubclass(compiled, MultipleChoice)
    assert compiled.tier == "imported"

    # Swap the dataset for the programmatic fixture so we don't need the Hub.
    class _OfflineCompiled(compiled):
        name = "_arc_challenge_offline_runtime"
        dataset = staticmethod(lambda: iter(_arc_like_dataset()))

    task = _OfflineCompiled(n_fewshot=0, limit=4)
    engine = StubEngine(model_id="stub/lm-eval", miss_every=4)
    result = run_eval(engine=engine, tasks=[task], output_dir=tmp_path)

    # Compiled lm-eval tasks report ``acc`` (the harness's convention),
    # not ``accuracy`` — the Anvil-curated MMLU/MMMU tasks use the longer
    # name. The manifest's per-task ``metric`` field encodes which name
    # this run used.
    score = result.scores[task.name]["acc"]
    assert 0.0 <= score <= 1.0
    assert result.manifest.tasks[0].metric == "acc"
    # The manifest tags the imported task.
    assert result.manifest.tasks[0].tier == "imported"
    assert result.manifest.tasks[0].name == task.name


def test_milestone_5_custom_modality_rna_offline() -> None:
    """The §6.7 RNA example, materialized.

    Uses :func:`anvil.load_custom`-shape (we instantiate a StubEngine
    directly to avoid loading rnafm), a programmatic dataset of
    (sequence, activity) pairs, and a regression head ``Task`` that
    aggregates Spearman correlation. Confirms the modality-agnostic
    Task abstraction works end-to-end.
    """
    import torch

    from anvil.primitives.response import EmbedResult

    try:
        from scipy.stats import spearmanr  # type: ignore[import-not-found]
    except ImportError:
        # scipy is heavyweight; for the offline test we compute a manual
        # rank correlation. The manuscript's actual RNA example uses scipy.
        def spearmanr(x: list[float], y: list[float]) -> Any:  # type: ignore[no-redef]
            class _Result:
                statistic: float

            n = len(x)
            ranks_x = sorted(range(n), key=lambda i: x[i])
            ranks_y = sorted(range(n), key=lambda i: y[i])
            rx = [0] * n
            ry = [0] * n
            for r, idx in enumerate(ranks_x):
                rx[idx] = r
            for r, idx in enumerate(ranks_y):
                ry[idx] = r
            mean_rx = sum(rx) / n
            mean_ry = sum(ry) / n
            num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
            denom_x = sum((rx[i] - mean_rx) ** 2 for i in range(n)) ** 0.5
            denom_y = sum((ry[i] - mean_ry) ** 2 for i in range(n)) ** 0.5
            r = _Result()
            r.statistic = (num / (denom_x * denom_y)) if (denom_x * denom_y) else 0.0
            return r

    # Programmatic dataset — sequences + scalar targets where higher activity
    # correlates with longer sequences (artificial, but stable for the test).
    docs = [
        {"sequence": "AAAA", "activity": 0.10},
        {"sequence": "AAAAA", "activity": 0.20},
        {"sequence": "AAAAAA", "activity": 0.40},
        {"sequence": "AAAAAAA", "activity": 0.55},
        {"sequence": "AAAAAAAA", "activity": 0.70},
        {"sequence": "AAAAAAAAA", "activity": 0.85},
    ]

    class _RNAFunctionRegression(Task):
        name = "_rna_function_offline"
        dataset = staticmethod(lambda: iter(docs))
        request_type = "Embed"
        metric_name = "spearman"

        def doc_to_request(self, doc: dict[str, Any]) -> Embed:
            return Embed(input=doc["sequence"], pool="mean", layer=-1)

        def request_to_prediction(self, response: Any, doc: dict[str, Any]) -> Any:
            del doc
            assert isinstance(response, EmbedResult)
            return response.embedding

        def aggregate(
            self, predictions: list[Any], in_docs: list[dict[str, Any]]
        ) -> dict[str, float]:
            # Project each embedding onto a fixed direction (a "regression
            # head") to get a scalar prediction. Real RNA work fits a Ridge
            # over a held-out split; for the unit test, just a stable map.
            head = torch.tensor(
                [1.0, -0.5, 0.25, 0.1, -0.1, 0.0, 0.0, 0.0]
            )  # 8-dim to match StubEngine
            # Score = activity ↔ sequence-length correlation, which the
            # stub embedding faithfully captures because longer inputs
            # produce different hashes.
            preds = [float(p[: len(head)] @ head[: len(p)]) for p in predictions]
            targets = [float(d["activity"]) for d in in_docs]
            stat = spearmanr(preds, targets).statistic
            # spearmanr returns NaN if everything ties; coerce.
            value = 0.0 if stat != stat else float(stat)  # noqa: PLR0124
            return {"spearman": value}

    engine = StubEngine(model_id="stub/rnafm")
    result = run_eval(engine=engine, tasks=[_RNAFunctionRegression()])
    assert "_rna_function_offline" in result.scores
    assert "spearman" in result.scores["_rna_function_offline"]
    # The manifest's task entry records the Embed request type.
    assert result.manifest.tasks[0].request_type == "Embed"
    # No sampler block — embed requests don't carry one.
    assert result.manifest.sampler is None


# ---------------------------------------------------------------- live


@pytest.mark.requires_hf_gated
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_milestone_5_lm_eval_yaml_imports_live() -> None:
    """The §16.10 live test: import arc_challenge, run against Llama-3.1-8B."""
    import os

    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    spec = {
        "task": "arc_challenge_live",
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Challenge",
        "output_type": "multiple_choice",
        "test_split": "test",
        "doc_to_text": "Question: {question}\nAnswer:",
        "doc_to_target": "answerKey",
        "doc_to_choice": "choices.text",
    }
    compile_yaml_dict(spec)

    import anvil

    result = anvil.eval(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tasks=["arc_challenge_live"],
        limit=50,
    )
    assert "arc_challenge_live" in result.scores
    assert result.manifest.tasks[0].tier == "imported"


@pytest.mark.requires_hf_gated
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_milestone_5_compare_with_lm_eval_live() -> None:
    """Anvil within 1pp of lm-evaluation-harness on the same task. Needs the
    harness installed and Llama-3 access."""
    pytest.importorskip("lm_eval")
    import os

    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")

    from anvil.tasks.lm_eval_shim import compare_with_lm_eval

    results = compare_with_lm_eval(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tasks=["arc_challenge"],
        limit=200,
        n_fewshot=0,
    )
    for r in results:
        assert abs(r.delta) < 0.01, f"{r.task}.{r.metric}: Δ={r.delta:.4f} > 0.01"


def test_milestone_5_compare_with_lm_eval_surfaces_clearly_when_missing() -> None:
    """When lm_eval isn't installed, compare_with_lm_eval should raise a
    clear ConfigError pointing at the install. We can't reliably uninstall
    in the test env, so we monkeypatch the import check."""
    from anvil.tasks.lm_eval_shim import compare as compare_module
    from anvil.tasks.lm_eval_shim import compare_with_lm_eval

    real_check = compare_module._have_lm_eval
    compare_module._have_lm_eval = lambda: False
    try:
        with pytest.raises(ConfigError, match="lm-evaluation-harness is not installed"):
            compare_with_lm_eval(model="any/model", tasks=["x"], limit=1)
    finally:
        compare_module._have_lm_eval = real_check
