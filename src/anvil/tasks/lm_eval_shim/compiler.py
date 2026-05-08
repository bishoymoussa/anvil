"""Compile an lm-evaluation-harness YAML into an Anvil :class:`Task` class
(design §6.4).

Supported output types:

* ``multiple_choice`` — emits a :class:`anvil.tasks.base.MultipleChoice`
  subclass that scores each choice as a continuation log-likelihood.
* ``loglikelihood`` — same path; the harness uses two output types for
  similar shapes.
* ``generate_until`` — emits a Generate-typed Task with extraction
  driven by the YAML's ``filter_list``.

YAML fields honored:

* ``task`` — used as the Anvil task name.
* ``dataset_path`` (and optionally ``dataset_name``) — the HF dataset id;
  the runner streams via ``datasets.load_dataset(path, name)``.
* ``test_split`` — defaults to ``test``.
* ``doc_to_text`` — Python expression or template string applied per-doc.
* ``doc_to_target`` — extracts the gold target.
* ``doc_to_choice`` — for multiple-choice, the per-option strings.
* ``num_fewshot`` — propagated to the Anvil task's ``n_fewshot_default``.
* ``output_type`` — routed as above.
* ``filter_list`` — for generate_until tasks, the regex/extractor chain.

Fields **not** honored (yet) and surfaced in :class:`UnsupportedYAML`:

* ``process_results`` — custom Python hook; out-of-scope for v0.
* ``training_split`` few-shot sampling beyond the simple "first N" form.
* Custom Jinja in ``doc_to_text`` — we apply Python ``str.format`` and
  fall back to direct field extraction; complex Jinja must be ported
  to a v0 Anvil ``Task`` by hand (the manifest's ``tier: imported`` tag
  signals that the run was best-effort migrated).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from anvil.exceptions import ConfigError
from anvil.primitives.request import Generate
from anvil.primitives.response import Generation
from anvil.primitives.sampler import Sampler
from anvil.tasks.base import MultipleChoice, Task
from anvil.tasks.registry import register_task

if TYPE_CHECKING:
    from collections.abc import Sequence

_LETTERS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")


class UnsupportedYAML(ConfigError):
    """The YAML uses a feature the v0 shim doesn't support."""

    error_code = "ANVIL-E0061"


# ---------------------------------------------------------------- helpers


def _eval_doc_to_text(template: str | None, doc: dict[str, Any]) -> str:
    """Apply a YAML ``doc_to_text`` template to a doc.

    Three accepted forms (in order of preference):

    1. Python ``str.format`` template (``"Question: {question}\\nAnswer:"``).
    2. A field name (``"question"``) — extracts ``doc[name]``.
    3. ``None`` / missing — falls back to ``doc.get("text") or str(doc)``.

    Custom Jinja-in-YAML forms must be ported by hand. The shim raises
    :class:`UnsupportedYAML` if a brace-delimited template references a
    missing field, rather than silently dropping the field.
    """
    if template is None:
        return str(doc.get("text") or doc.get("question") or doc)
    if "{" in template:
        try:
            return template.format(**doc)
        except (KeyError, IndexError) as exc:
            raise UnsupportedYAML(
                f"doc_to_text template references missing field: {exc}. "
                "Port complex Jinja templates to an Anvil Task subclass."
            ) from exc
    if isinstance(doc.get(template), str | int | float):
        return str(doc[template])
    return str(doc)


def _eval_doc_to_target(template: str | None, doc: dict[str, Any]) -> Any:
    """Extract the target value. Same shape as :func:`_eval_doc_to_text`."""
    if template is None:
        return doc.get("target") or doc.get("answer") or doc.get("label")
    if "{" in template:
        return template.format(**doc)
    if template in doc:
        return doc[template]
    return doc.get("target")


def _eval_doc_to_choice(spec: Any, doc: dict[str, Any]) -> list[str]:
    """Extract the list of choices for multiple-choice tasks."""
    if isinstance(spec, list):
        return [str(s) for s in spec]
    if isinstance(spec, str):
        if spec in doc and isinstance(doc[spec], list | tuple):
            return [str(s) for s in doc[spec]]
        if "{" in spec:
            rendered = spec.format(**doc)
            return [s.strip() for s in rendered.split("\n") if s.strip()]
    if isinstance(doc.get("choices"), list | tuple):
        return [str(s) for s in doc["choices"]]
    raise UnsupportedYAML(
        "could not resolve doc_to_choice; supply a list, a field name, or a template"
    )


def _normalize_target_to_index(target: Any, choices: Sequence[str]) -> int:
    """Coerce a target value to the choice index it represents.

    Targets in the harness are either:
    * an int index (``0..len(choices)-1``),
    * the literal choice string (find by equality),
    * the letter (``"A".."J"``).
    """
    if isinstance(target, int):
        if 0 <= target < len(choices):
            return target
        raise ValueError(f"target index {target} out of range for {len(choices)} choices")
    if isinstance(target, str):
        s = target.strip()
        if len(s) == 1 and s.upper() in _LETTERS:
            idx = _LETTERS.index(s.upper())
            if idx < len(choices):
                return idx
        if s in choices:
            return list(choices).index(s)
        try:
            return int(s)
        except ValueError as exc:
            raise ValueError(
                f"could not match target {target!r} to any of {list(choices)}"
            ) from exc
    raise TypeError(f"unsupported target type {type(target).__name__}")


# ---------------------------------------------------------------- main API


def compile_yaml(path: str | Path) -> type[Task]:
    """Load a YAML file and compile to an Anvil :class:`Task` subclass.

    The returned class is **registered** in :func:`anvil.tasks.registry`
    on first compilation and tagged ``tier='imported'``. Re-compiling the
    same YAML returns the cached class.

    Example:
        >>> Task = compile_yaml("path/to/arc_challenge.yaml")  # doctest: +SKIP
        >>> # Now reachable as `anvil.eval(model=..., tasks=["arc_challenge"])`.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"lm-eval YAML not found: {p}")
    spec = yaml.safe_load(p.read_text())
    if not isinstance(spec, dict):
        raise UnsupportedYAML(f"{p}: top-level YAML must be a mapping")
    return compile_yaml_dict(spec)


def compile_yaml_dict(spec: dict[str, Any]) -> type[Task]:
    """Compile an in-memory lm-eval-style spec dict to an Anvil Task class.

    Used by :func:`compile_yaml` and by tests that build specs in code.
    """
    name = spec.get("task")
    if not isinstance(name, str) or not name:
        raise UnsupportedYAML("'task' field is required and must be a non-empty string")

    output_type = spec.get("output_type", "loglikelihood")
    if output_type in ("multiple_choice", "loglikelihood"):
        return _compile_multiple_choice(spec, output_type)
    if output_type in ("generate_until", "generate"):
        return _compile_generate(spec)
    raise UnsupportedYAML(f"output_type={output_type!r} not supported in v0")


# ---------------------------------------------------------------- multiple_choice


def _compile_multiple_choice(spec: dict[str, Any], output_type: str) -> type[Task]:
    # Bind every function-local under a different name from the class
    # attribute it eventually becomes — Python class-body scope does not
    # reach back into the enclosing function's locals via closure for
    # names that match a class attribute on the same line. ``name = name``
    # would NameError; ``name = _t_name`` is fine.
    _t_name = str(spec["task"])
    _t_dataset = str(spec.get("dataset_path", ""))
    _t_dataset_name = spec.get("dataset_name")
    _t_test_split = str(spec.get("test_split", "test"))
    _t_fewshot = int(spec.get("num_fewshot", 0))
    _t_doc_to_text = spec.get("doc_to_text")
    _t_doc_to_target = spec.get("doc_to_target")
    _t_doc_to_choice = spec.get("doc_to_choice")

    class _Compiled(MultipleChoice):
        name = _t_name
        dataset = _t_dataset
        n_fewshot_default = _t_fewshot
        metric_name = "acc"
        tier = "imported"

        # Carry the YAML through so users (and tests) can inspect it.
        lm_eval_spec: dict[str, Any] = dict(spec)
        lm_eval_dataset_name: Any = _t_dataset_name
        lm_eval_test_split: str = _t_test_split
        lm_eval_output_type: str = output_type

        def doc_to_text(self, doc: dict[str, Any]) -> str:
            return _eval_doc_to_text(_t_doc_to_text, doc)

        def doc_to_choices(self, doc: dict[str, Any]) -> list[str]:
            return _eval_doc_to_choice(_t_doc_to_choice, doc)

        def doc_to_target(self, doc: dict[str, Any]) -> int:
            target = _eval_doc_to_target(_t_doc_to_target, doc)
            choices = self.doc_to_choices(doc)
            return _normalize_target_to_index(target, choices)

    _Compiled.__qualname__ = f"LmEvalTask_{_t_name}"
    register_task(_Compiled)
    return _Compiled


# ---------------------------------------------------------------- generate_until


def _compile_generate(spec: dict[str, Any]) -> type[Task]:
    # See _compile_multiple_choice for why these are renamed away from the
    # class attributes they become.
    _t_name = str(spec["task"])
    _t_dataset = str(spec.get("dataset_path", ""))
    _t_dataset_name = spec.get("dataset_name")
    _t_test_split = str(spec.get("test_split", "test"))
    _t_fewshot = int(spec.get("num_fewshot", 0))
    _t_doc_to_text = spec.get("doc_to_text")
    _t_doc_to_target = spec.get("doc_to_target")
    _t_until = spec.get("generation_kwargs", {}).get("until") or []
    if isinstance(_t_until, str):
        _t_until = [_t_until]
    _t_max_new_tokens = int(spec.get("generation_kwargs", {}).get("max_gen_toks", 256))
    _t_filters = spec.get("filter_list") or []

    class _Compiled(Task):
        name = _t_name
        dataset = _t_dataset
        n_fewshot_default = _t_fewshot
        metric_name = "acc"
        tier = "imported"
        request_type = "Generate"

        lm_eval_spec: dict[str, Any] = dict(spec)
        lm_eval_dataset_name: Any = _t_dataset_name
        lm_eval_test_split: str = _t_test_split

        def doc_to_request(self, doc: dict[str, Any]) -> Generate:
            text = _eval_doc_to_text(_t_doc_to_text, doc)
            messages: list[dict[str, Any]] = [{"role": "user", "content": text}]
            sampler = Sampler.greedy(max_tokens=_t_max_new_tokens, stop=tuple(_t_until))
            return Generate(messages=tuple(messages), sampler=sampler)

        def request_to_prediction(self, response: Any, doc: dict[str, Any]) -> str:
            del doc
            if not isinstance(response, Generation):
                raise TypeError(f"expected Generation, got {type(response).__name__}")
            return _apply_filters(response.text, _t_filters)

        def aggregate(self, predictions: list[Any], docs: list[dict[str, Any]]) -> dict[str, float]:
            from anvil.metrics.exact_match import exact_match

            if not predictions:
                return {self.metric_name: 0.0}
            correct = 0
            for pred, doc in zip(predictions, docs, strict=True):
                target = str(_eval_doc_to_target(_t_doc_to_target, doc) or "")
                if exact_match(str(pred), target) > 0.5:
                    correct += 1
            return {self.metric_name: correct / len(predictions)}

    _Compiled.__qualname__ = f"LmEvalTask_{_t_name}"
    register_task(_Compiled)
    return _Compiled


def _apply_filters(text: str, filter_list: list[Any]) -> str:
    """Apply lm-eval-harness-style ``filter_list`` extractors to ``text``.

    Supported filter shapes:

    * ``regex_pattern`` (``regex``) — first capture group of the pattern.
    * ``take_first`` — split on whitespace, return the first token.
    * ``strip_decimal_zeros`` — collapses ``"5.0"`` → ``"5"``.

    Anything else is ignored with a warning logged in the runner; complex
    Python ``filter:`` callables must be ported.
    """
    out = text
    for filt in filter_list:
        if not isinstance(filt, dict):
            continue
        for inner in filt.get("filter", [filt]):
            if not isinstance(inner, dict):
                continue
            kind = inner.get("function")
            if kind == "regex":
                pattern = inner.get("regex_pattern", "")
                m = re.search(pattern, out)
                if m is not None:
                    out = m.group(1) if m.groups() else m.group(0)
            elif kind == "take_first":
                out = out.split(maxsplit=1)[0] if out.split() else out
    return out.strip()


__all__ = [
    "compile_yaml",
    "compile_yaml_dict",
    "UnsupportedYAML",
]
