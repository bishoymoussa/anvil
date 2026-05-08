"""``anvil eval --compare-with-lm-eval``: run both engines, diff the scores
(design §6.4, §17 A.6).

Returns a structured delta report so the user sees exactly which fields
account for any score difference. If lm-evaluation-harness isn't
installed, the function raises with a clear error pointing at the
optional install.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from anvil.exceptions import ConfigError
from anvil.logging import get_logger

_log = get_logger(__name__)


@dataclass
class CompareResult:
    """One side-by-side comparison of Anvil vs lm-eval-harness scores."""

    task: str
    metric: str
    anvil: float
    lm_eval: float
    delta: float

    def render(self) -> str:
        return (
            f"{self.task}.{self.metric}: anvil={self.anvil:.4f} "
            f"lm_eval={self.lm_eval:.4f} Δ={self.delta:+.4f}"
        )


def _have_lm_eval() -> bool:
    """True iff lm-evaluation-harness is importable in the current env."""
    try:
        importlib.import_module("lm_eval")
    except ImportError:
        return False
    return True


def compare_with_lm_eval(
    *,
    model: str,
    tasks: list[str],
    limit: int | None = None,
    n_fewshot: int = 0,
) -> list[CompareResult]:
    """Run Anvil and lm-evaluation-harness against the same task; diff scores.

    Args:
        model: HF model id; passed to both engines.
        tasks: lm-eval task names. Compiled to Anvil tasks via
            :func:`anvil.tasks.lm_eval_shim.compile_yaml`-equivalent
            spec lookup.
        limit: cap docs per task (passed to both).
        n_fewshot: few-shot count.

    Returns:
        list of :class:`CompareResult` entries — one per ``(task, metric)``.

    Raises:
        ConfigError: if lm-evaluation-harness isn't installed.
    """
    if not _have_lm_eval():
        raise ConfigError(
            "lm-evaluation-harness is not installed; --compare-with-lm-eval "
            "requires it. Install with `uv pip install lm_eval`."
        )

    # Import via tasks.public, NOT the top-level ``anvil`` package — the
    # latter pulls in ``anvil.server``, which the layered architecture
    # contract forbids ``anvil.tasks`` from depending on.
    from anvil.tasks.public import eval as _anvil_eval

    anvil_result = _anvil_eval(
        model=model,
        tasks=tasks,
        n_fewshot=n_fewshot,
        limit=limit,
    )

    lm_eval_scores = _run_lm_eval(model=model, tasks=tasks, limit=limit, n_fewshot=n_fewshot)

    out: list[CompareResult] = []
    for task in tasks:
        anvil_metrics = anvil_result.scores.get(task, {})
        lm_metrics = lm_eval_scores.get(task, {})
        # Surface every metric we have a number for on either side.
        all_metric_names = set(anvil_metrics) | set(lm_metrics)
        for metric in sorted(all_metric_names):
            a = float(anvil_metrics.get(metric, float("nan")))
            b = float(lm_metrics.get(metric, float("nan")))
            out.append(
                CompareResult(
                    task=task,
                    metric=metric,
                    anvil=a,
                    lm_eval=b,
                    delta=a - b,
                )
            )
    return out


def _run_lm_eval(
    *,
    model: str,
    tasks: list[str],
    limit: int | None,
    n_fewshot: int,
) -> dict[str, dict[str, float]]:  # pragma: no cover - exercised only when lm_eval installed
    """Drive lm-evaluation-harness's Python API.

    Branches on the harness API surface so this stays robust across the
    lm-eval-harness versions Anvil's gap analysis cites.
    """
    lm_eval = importlib.import_module("lm_eval")

    simple_evaluate = getattr(lm_eval, "simple_evaluate", None)
    if simple_evaluate is None:
        raise ConfigError(
            "lm_eval.simple_evaluate not found; this Anvil shim was tested "
            "against lm-evaluation-harness ≥0.4. Upgrade or pin compatibly."
        )

    _log.info("running lm-evaluation-harness for comparison: model=%s tasks=%s", model, tasks)
    raw = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model}",
        tasks=tasks,
        num_fewshot=n_fewshot,
        limit=limit,
    )
    results: dict[str, dict[str, float]] = {}
    for task_name, metrics in raw.get("results", {}).items():
        # lm-eval's metric keys look like "acc,none" / "acc_norm,none"; we
        # collapse the "filter" suffix for the comparison output.
        clean: dict[str, float] = {}
        for raw_metric, value in metrics.items():
            if isinstance(value, int | float):
                stem = raw_metric.split(",")[0]
                clean[stem] = float(value)
        results[task_name] = clean
    return results


# Used by the test_milestone_5_compare_with_lm_eval offline check.
def render_report(results: list[CompareResult]) -> str:
    """Format a comparison list as a single multi-line string."""
    if not results:
        return "(no results)"
    return "\n".join(r.render() for r in results)


__all__ = ["compare_with_lm_eval", "CompareResult", "render_report"]
