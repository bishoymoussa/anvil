"""Model loading + fast-path registry (design §5.1, §5.3).

For M0 :func:`load` constructs an ``HFEngine`` directly and returns a
:class:`LoadedModel` wrapper. The fast-path registry is in place so that M1+
can decorate architectures without changing the public surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anvil.engine import build_engine
from anvil.exceptions import ConfigError
from anvil.logging import get_logger
from anvil.primitives.request import Generate
from anvil.primitives.sampler import Sampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from anvil.engine.public import Engine
    from anvil.primitives.response import Generation

_log = get_logger(__name__)

_FAST_PATHS: dict[str, type[Any]] = {}


def register_model_impl(architecture: str) -> Callable[[type[Any]], type[Any]]:
    """Decorator: register a fast-path implementation for ``architecture``.

    ``architecture`` is matched against ``config.architectures[0]``.

    Example:
        >>> @register_model_impl("LlamaForCausalLM")
        ... class _Llama:
        ...     pass
        >>> "LlamaForCausalLM" in _FAST_PATHS
        True
    """

    def _decorate(cls: type[Any]) -> type[Any]:
        if architecture in _FAST_PATHS:
            raise ConfigError(
                f"fast-path for {architecture!r} already registered "
                f"({_FAST_PATHS[architecture].__name__})"
            )
        _FAST_PATHS[architecture] = cls
        return cls

    return _decorate


@dataclass
class LoadedModel:
    """A loaded model + a thin generate facade for ad-hoc use.

    For evaluation flows users go through :func:`anvil.eval`; for tutorials
    and notebook exploration ``model.generate(...)`` is the obvious shape.
    """

    engine: Engine
    model_id: str

    def generate(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        sampler: Sampler | None = None,
        max_tokens: int | None = None,
    ) -> Generation:
        """Single-prompt convenience wrapper around :meth:`Engine.generate_logprobs`."""
        s = sampler or Sampler.greedy(max_tokens=max_tokens or 256)
        msgs_t: tuple[dict[str, Any], ...] | None = (
            tuple(messages) if messages is not None else None
        )
        req = Generate(messages=msgs_t, prompt=prompt, sampler=s)
        outs = self.engine.generate_logprobs([req])
        return outs[0]

    def shutdown(self) -> None:
        self.engine.shutdown()

    @property
    def info(self) -> dict[str, Any]:
        return self.engine.model_info


def load(
    model_id: str,
    *,
    engine: str = "auto",
    revision: str | None = None,
    dtype: str | None = None,
    device_map: str | None = None,
    engine_args: dict[str, Any] | None = None,
) -> LoadedModel:
    """Load a model.

    Args:
        model_id: HF Hub id or local path.
        engine: ``"auto"``, ``"hf"``, or ``"vllm"`` (M1+).
        revision: optional model revision pin.
        dtype: ``"bfloat16"``, ``"float16"``, ``"float32"``, or ``None``.
        device_map: passed to transformers; default ``"auto"``.
        engine_args: backend-specific extras.

    Example:
        >>> # m = load("Qwen/Qwen2.5-1.5B-Instruct")  # doctest: +SKIP
        >>> # m.generate(prompt="Hello").text  # doctest: +SKIP
    """
    if engine not in ("auto", "hf", "vllm"):
        raise ConfigError(f"engine must be one of {{auto, hf, vllm}}, got {engine!r}")
    eng = build_engine(
        model_id=model_id,
        engine=engine,  # type: ignore[arg-type]
        revision=revision,
        dtype=dtype,
        device_map=device_map,
        engine_args=engine_args,
    )
    _log.info("loaded model %s via %s", model_id, eng.__class__.__name__)
    return LoadedModel(engine=eng, model_id=model_id)


def load_custom(
    *,
    model_id: str,
    model_class: type[Any] | None = None,
    tokenizer_class: type[Any] | None = None,
    revision: str | None = None,
    dtype: str | None = None,
    device_map: str | None = None,
    engine_args: dict[str, Any] | None = None,
) -> LoadedModel:
    """Load a non-causal model via the slow path (design §6.7).

    The ``model_class`` is whatever transformers (or third-party) class
    is appropriate for the architecture: ``AutoModel`` for generic
    encoders, ``AutoModelForSequenceClassification`` for classifiers,
    or — for non-text-domain models like RNA-FM — the project's own
    class import. ``trust_remote_code=True`` is enabled because most
    domain-specific models ship custom modeling code; v0.5 will gate
    this behind explicit user consent (per §7.7 KB
    ``trust_remote_code_required``).

    Args:
        model_id: HF model id or local path.
        model_class: a class with ``.from_pretrained``. Defaults to
            ``transformers.AutoModel``.
        tokenizer_class: a class with ``.from_pretrained``. Defaults to
            ``transformers.AutoTokenizer``.
        revision: optional model revision pin.
        dtype: ``"bfloat16"``, ``"float16"``, ``"float32"``, or ``None``.
        device_map: passed through to ``from_pretrained`` (default
            ``"auto"``).
        engine_args: backend-specific extras; reserved for v0.5.

    Example:
        >>> from transformers import AutoModel  # doctest: +SKIP
        >>> m = anvil.load_custom(  # doctest: +SKIP
        ...     model_id="multimolecule/rnafm",
        ...     model_class=AutoModel,
        ... )
    """
    from anvil.engine._hf.embed_runner import HFEmbedEngine

    eng = HFEmbedEngine(
        model_id=model_id,
        revision=revision,
        dtype=dtype,
        device_map=device_map or "auto",
        model_class=model_class,
        tokenizer_class=tokenizer_class,
        engine_args=engine_args,
    )
    _log.info("loaded custom model %s via HFEmbedEngine", model_id)
    return LoadedModel(engine=eng, model_id=model_id)


__all__ = ["load", "load_custom", "register_model_impl", "LoadedModel"]
