"""The batched evaluation loop (design §6.2).

Reads docs from the task's dataset, materializes :class:`Generate` requests
in batches, drives the engine at full throughput, decodes per-doc
predictions, and aggregates scores. The full :class:`anvil.Manifest` is
emitted at end-of-run.

For M0 this only handles ``Generate``-shaped tasks (``GSM8K``); other
request types raise. M1 extends to ``LogLikelihood``; M5 extends to
``Embed`` / ``Custom``.
"""

from __future__ import annotations

import datetime as _dt
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil._version import __version__ as _anvil_version
from anvil.exceptions import TaskError
from anvil.logging import get_logger
from anvil.manifest.schema import Manifest, ModelInfo, TaskInfo
from anvil.primitives.request import Generate
from anvil.tasks.base import Task, materialize_dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from anvil.engine.public import Engine

_log = get_logger(__name__)


@dataclass
class EvalRunResult:
    """The outcome of an evaluation run.

    Attributes:
        scores: ``{task_name: {metric_name: value}}``.
        manifest: the (signed) :class:`Manifest`.
        outputs: per-task list of raw engine responses, exposed for research
            workflows (not serialized into the manifest).
    """

    scores: dict[str, dict[str, float]]
    manifest: Manifest
    outputs: dict[str, list[Any]]


def _split_for(task: Task) -> str:
    """Return the dataset split this task expects.

    Hard-coded per task because M0 doesn't yet have a metadata layer.
    """
    if task.name == "gsm8k":
        return "test"
    return "test"


def _config_name_for(task: Task) -> str | None:
    if task.name == "gsm8k":
        return "main"
    return None


def _iter_docs(task: Task) -> Iterator[dict[str, Any]]:
    """Stream docs from the task's dataset, applying ``limit`` if set."""
    # Access via type(...) to bypass mypy's bound-method binding for
    # ClassVar[... | Callable]; runtime semantics identical.
    spec = type(task).dataset
    if isinstance(spec, str) and "/" in spec and not Path(spec).exists():
        # HF id with optional config name. Use the dataset library directly so we
        # can pass the config (gsm8k has 'main' / 'socratic').
        from datasets import load_dataset

        cfg = _config_name_for(task)
        ds = (
            load_dataset(spec, cfg, split=_split_for(task))
            if cfg
            else load_dataset(spec, split=_split_for(task))
        )
        rows: Iterable[dict[str, Any]] = (dict(r) for r in ds)
    else:
        rows = materialize_dataset(spec, split=_split_for(task))
    for yielded, row in enumerate(rows):
        if task.limit is not None and yielded >= task.limit:
            return
        yield row


def _hardware_info() -> dict[str, Any]:
    """Best-effort host description for the manifest.

    All torch probes are wrapped: a torch wheel mismatched against the host
    CUDA driver (design §1.2 / §9.1) raises a UserWarning during
    ``torch.cuda.is_available()`` that we don't want to escalate. The
    manifest's ``hardware`` field is informational only — if we can't get
    GPU info, we record what we know.
    """
    import warnings

    info: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                cuda_ok = torch.cuda.is_available()
            except (RuntimeError, OSError) as exc:
                info["cuda_probe_error"] = str(exc)
                cuda_ok = False
        if cuda_ok:
            try:
                info["gpus"] = [
                    torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                ]
                info["cuda"] = torch.version.cuda
            except (RuntimeError, OSError) as exc:  # pragma: no cover - hardware-dependent
                info["cuda_probe_error"] = str(exc)
    except ImportError:  # pragma: no cover - torch is a hard dep
        pass
    return info


def run_eval(
    *,
    engine: Engine,
    tasks: list[Task],
    output_dir: Path | None = None,
    extra_manifest: dict[str, Any] | None = None,
    caas_log: Any = None,
) -> EvalRunResult:
    """Run all ``tasks`` against ``engine`` and emit a manifest.

    Args:
        engine: the constructed engine.
        tasks: instantiated :class:`Task`s.
        output_dir: if given, the manifest is written to
            ``output_dir/manifest.json``. The directory is created if needed.
        extra_manifest: optional additional fields merged into the manifest's
            ``smoke_test`` block — used by CaaS in M3+ to record sentinel data.
        caas_log: an optional :class:`anvil.caas.AuditLog` whose records
            flow into the manifest's ``caas_log`` field. The runner does
            not engage CaaS itself — that happens before the call (typically
            in :func:`anvil.tasks.public.eval`); the runner only embeds the
            already-recorded actions in the signed manifest.

    Returns:
        :class:`EvalRunResult` with scores, the signed manifest, and per-task
        raw outputs (the latter is **not** persisted; it's a research hook).
    """
    started_at = _dt.datetime.now(_dt.UTC).isoformat()
    scores: dict[str, dict[str, float]] = {}
    raw_outputs: dict[str, list[Any]] = {}
    task_infos: list[TaskInfo] = []

    sampler_field: dict[str, Any] | None = None

    for task in tasks:
        if task.request_type not in ("Generate", "LogLikelihood", "Embed", "Custom"):
            raise TaskError(
                f"task {task.name!r}: request_type={task.request_type!r} not yet "
                "supported. M5 ships Generate / LogLikelihood / Embed / Custom; "
                "Classify lands in v0.5 (design §16.10)."
            )
        _log.info(
            "running task %s (n_fewshot=%d, limit=%s, type=%s)",
            task.name,
            task.n_fewshot,
            task.limit,
            task.request_type,
        )
        docs: list[dict[str, Any]] = list(_iter_docs(task))
        if not docs:
            raise TaskError(f"task {task.name!r}: dataset yielded no rows")

        # ``doc_to_request`` may return a single Request or a Sequence (MCQ).
        # Flatten with per-doc counts so we can re-group responses afterward.
        flat_requests: list[Any] = []
        per_doc_counts: list[int] = []
        for doc in docs:
            r = task.doc_to_request(doc)
            if isinstance(r, list | tuple):
                flat_requests.extend(r)
                per_doc_counts.append(len(r))
            else:
                flat_requests.append(r)
                per_doc_counts.append(1)

        # Dispatch.
        flat_responses: list[Any]
        if task.request_type == "Generate":
            for fr in flat_requests:
                if not isinstance(fr, Generate):
                    raise TaskError(
                        f"task {task.name!r}: request_type='Generate' but a non-Generate "
                        f"request appeared: {type(fr).__name__}"
                    )
            generate_requests: list[Generate] = list(flat_requests)
            # Capture the sampler for the manifest (first non-None wins).
            for req in generate_requests:
                if req.sampler is not None and sampler_field is None:
                    sampler_field = req.sampler.to_manifest_field()
                    break
            flat_responses = _batched_generate(engine, generate_requests, batch_size=2)
        elif task.request_type == "LogLikelihood":
            from anvil.primitives.request import LogLikelihood

            ll_requests: list[LogLikelihood] = list(flat_requests)
            for fr in ll_requests:
                if not isinstance(fr, LogLikelihood):
                    raise TaskError(
                        f"task {task.name!r}: request_type='LogLikelihood' but a "
                        f"non-LogLikelihood request appeared: {type(fr).__name__}"
                    )
            flat_responses = list(engine.loglikelihood(ll_requests))
        elif task.request_type == "Embed":
            from anvil.primitives.request import Embed as _Embed

            embed_requests: list[_Embed] = list(flat_requests)
            for fr in embed_requests:
                if not isinstance(fr, _Embed):
                    raise TaskError(
                        f"task {task.name!r}: request_type='Embed' but a non-Embed "
                        f"request appeared: {type(fr).__name__}"
                    )
            flat_responses = list(engine.embed(embed_requests))
        else:  # Custom
            from anvil.primitives.request import Custom as _Custom

            custom_requests: list[_Custom] = list(flat_requests)
            flat_responses = []
            for fr in custom_requests:
                if not isinstance(fr, _Custom):
                    raise TaskError(
                        f"task {task.name!r}: request_type='Custom' but a non-Custom "
                        f"request appeared: {type(fr).__name__}"
                    )
                inputs = list(fr.inputs or [])
                flat_responses.extend(engine.custom(fr.fn, inputs))

        # Re-group flat responses per-doc.
        per_doc_responses: list[Any] = []
        cursor = 0
        for count in per_doc_counts:
            chunk = flat_responses[cursor : cursor + count]
            per_doc_responses.append(chunk if count != 1 else chunk[0])
            cursor += count

        predictions = [
            task.request_to_prediction(r, d) for r, d in zip(per_doc_responses, docs, strict=True)
        ]
        scores[task.name] = task.aggregate(predictions, docs)
        raw_outputs[task.name] = list(per_doc_responses)

        task_infos.append(
            TaskInfo(
                name=task.name,
                tier=task.tier,
                version=f"{task.name}@anvil-v0",
                dataset_revision=_dataset_revision(task),
                n_fewshot=task.n_fewshot,
                metric=task.metric_name,
                request_type=task.request_type,
            )
        )

    ended_at = _dt.datetime.now(_dt.UTC).isoformat()

    # Pull CaaS records into the manifest's caas_log if a log was passed in.
    caas_records = caas_log.to_list() if caas_log is not None else []

    manifest = Manifest(
        anvil_version=_anvil_version,
        engine=_engine_dict(engine),
        model=_model_info(engine),
        tokenization=_tokenization_field(engine),
        chat_template=None,  # M1 plumbs ChatTemplate through the runner
        sampler=sampler_field,
        tasks=task_infos,
        scores=scores,
        smoke_test={"samples": 0, "outcome": "skipped"} | (extra_manifest or {}),
        caas_log=caas_records,
        hardware=_hardware_info(),
        started_at=started_at,
        ended_at=ended_at,
    ).sign()

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest.save(out / "manifest.json")

    return EvalRunResult(scores=scores, manifest=manifest, outputs=raw_outputs)


def _batched_generate(engine: Engine, requests: list[Generate], *, batch_size: int) -> list[Any]:
    """Dispatch ``requests`` to the engine in batches of ``batch_size``."""
    out: list[Any] = []
    for start in range(0, len(requests), batch_size):
        chunk = requests[start : start + batch_size]
        out.extend(engine.generate_logprobs(chunk))
    return out


def _engine_dict(engine: Engine) -> dict[str, Any]:
    """Pull the engine's identifying info into a manifest-shaped dict."""
    info = getattr(engine, "backend_info", None)
    if info is not None:
        return dict(info)
    return {"name": "unknown", "version": "?", "backend_hash": engine.backend_hash}


def _model_info(engine: Engine) -> ModelInfo:
    info = engine.model_info
    return ModelInfo(
        id=str(info["id"]),
        revision=str(info["revision"]),
        dtype=str(info["dtype"]),
        quantization=info.get("quantization"),
        config_hash=str(info["config_hash"]),
        architecture=str(info["architecture"]),
    )


def _tokenization_field(engine: Engine) -> dict[str, Any]:
    """Synthesize the manifest's tokenization block.

    For M0 the engine doesn't expose a Tokenization primitive directly; we
    record the EOS ids and padding side from the underlying tokenizer where
    available. M1+ promotes this to a full :class:`Tokenization` projection.
    """
    out: dict[str, Any] = {"hash": "sha256:unset", "padding_side": "left"}
    tk = getattr(engine, "tokenizer", None)
    if tk is not None:
        eos = []
        if tk.eos_token_id is not None:
            eos.append(int(tk.eos_token_id))
        out["eos_token_ids"] = eos
        out["bos_handling"] = "from-template"
    return out


def _dataset_revision(task: Task) -> str:
    """Best-effort dataset SHA. Real implementation lands in M2.

    For HF ids we record the spec verbatim with a ``hf:`` prefix; for
    callables/paths we hash the path string. The manifest schema only requires
    a stable string here.
    """
    spec = type(task).dataset
    if callable(spec):
        return "callable:" + getattr(spec, "__qualname__", "anonymous")
    return f"hf:{spec}"


__all__ = ["EvalRunResult", "run_eval"]
