"""Manifest replay (design §8.3, §16.10).

Reconstruct an ``anvil.eval(...)`` invocation from a saved manifest and
re-execute it. If the manifest is well-formed and the run was deterministic,
the replay's scores are byte-identical to the original.

What gets reconstructed
-----------------------

* ``model.id`` and ``model.revision`` → the engine is rebuilt against the
  same weights.
* ``engine.name`` → the same backend (hf | vllm) is preferred.
* ``sampler`` → an :class:`anvil.Sampler` is reconstructed field-by-field.
* ``tasks`` → each task is looked up in the registry by name, with
  ``n_fewshot`` and ``limit`` re-applied.

What is NOT reconstructed
-------------------------

* The CaaS log. CaaS preflight runs again on replay; if the original run
  had auto-fixes, replay either (a) reproduces them deterministically or
  (b) raises ``ReplayMismatch`` if the environment differs. (CaaS itself
  lands in M3, so v0 replay never has CaaS deltas to worry about.)
* Hardware info — irrelevant for reproducibility of scores; recorded but
  not consulted.
* Custom :class:`Task` subclasses must be re-registered by the caller
  before replay (e.g. by importing the module that defines them).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.exceptions import ManifestError
from anvil.logging import get_logger

if TYPE_CHECKING:
    from anvil.manifest.schema import Manifest
    from anvil.tasks.runner import EvalRunResult

_log = get_logger(__name__)


def replay(
    source: Manifest | str | Path,
    *,
    output_dir: str | Path | None = None,
    strict: bool = False,
) -> EvalRunResult:
    """Re-run an evaluation from a saved manifest.

    Args:
        source: an in-memory :class:`Manifest`, a path to a manifest JSON
            file, or a string holding canonical-JSON manifest text.
        output_dir: if given, the new run's manifest is written there.
        strict: if True, raise ``ManifestError`` when the new manifest's
            scores differ from the source. Default False — replay returns
            the new result and the caller compares manually.

    Raises:
        ManifestError: if the source isn't a valid manifest, or if any
            referenced task is not registered, or (when ``strict``) if
            scores don't match.
    """
    manifest = _coerce_manifest(source)

    # Late imports to keep the manifest layer free of any tasks / engine deps.
    from anvil.primitives.sampler import Sampler
    from anvil.tasks.public import eval as anvil_eval
    from anvil.tasks.registry import get_task

    task_instances = []
    for task_info in manifest.tasks:
        try:
            cls = get_task(task_info.name)
        except Exception as exc:
            raise ManifestError(
                f"replay: task {task_info.name!r} not registered. Custom "
                "tasks must be imported before replay."
            ) from exc
        task_instances.append(cls(n_fewshot=task_info.n_fewshot))

    sampler = _sampler_from_manifest(manifest, Sampler)
    engine = manifest.engine.get("name", "auto")

    _log.info(
        "replaying manifest: model=%s engine=%s tasks=%s",
        manifest.model.id,
        engine,
        [t.name for t in manifest.tasks],
    )

    result = anvil_eval(
        model=manifest.model.id,
        tasks=task_instances,
        revision=manifest.model.revision if manifest.model.revision != "main" else None,
        engine=engine,
        dtype=manifest.model.dtype,
        output_dir=output_dir,
    )
    del sampler  # currently unused: anvil.eval doesn't take a sampler kwarg in M0/M1.

    if strict:
        for task_name, original in manifest.scores.items():
            replayed = result.scores.get(task_name)
            if replayed != original:
                raise ManifestError(
                    f"replay mismatch for task {task_name!r}: original {original} vs "
                    f"replayed {replayed}"
                )

    return result


def _coerce_manifest(source: Manifest | str | Path) -> Manifest:
    """Accept Manifest | path | JSON-text and return a Manifest."""
    from anvil.manifest.schema import Manifest

    if isinstance(source, Manifest):
        return source
    if isinstance(source, Path) or (
        isinstance(source, str) and (len(source) < 4096 and Path(source).exists())
    ):
        return Manifest.load(Path(source))
    if isinstance(source, str):
        # Treat as raw canonical-JSON text.
        try:
            return Manifest.model_validate_json(source)
        except Exception as exc:
            raise ManifestError(f"replay: source is not a valid manifest: {exc}") from exc
    raise TypeError(f"replay: unexpected source type {type(source).__name__}")


def _sampler_from_manifest(manifest: Manifest, sampler_cls: type[Any]) -> Any | None:
    """Reconstruct a :class:`Sampler` from the manifest's ``sampler`` block.

    Returns ``None`` for embed/classify-only manifests where the sampler
    field is absent.
    """
    if manifest.sampler is None:
        return None
    s = manifest.sampler
    return sampler_cls(
        temperature=float(s.get("temperature", 0.0)),
        top_p=float(s.get("top_p", 1.0)),
        top_k=int(s.get("top_k", -1)),
        min_p=float(s.get("min_p", 0.0)),
        repetition_penalty=float(s.get("repetition_penalty", 1.0)),
        presence_penalty=float(s.get("presence_penalty", 0.0)),
        frequency_penalty=float(s.get("frequency_penalty", 0.0)),
        max_tokens=int(s.get("max_tokens", 2048)),
        seed=s.get("seed"),
        stop=tuple(s.get("stop", ()) or ()),
        stop_token_ids=tuple(s.get("stop_token_ids", ()) or ()),
        n=int(s.get("n", 1)),
        source=str(s.get("source", "explicit")),
    )


__all__ = ["replay"]
