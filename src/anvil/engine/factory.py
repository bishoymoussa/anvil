"""Engine factory: pick a backend per (model, hardware, user choice).

For M0 the only available backend is the HuggingFace slow path. M1 introduces
vLLM and the auto-selection logic that prefers the fast path when both the
architecture is on the fast list and vLLM is installed and the hardware
supports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from anvil.exceptions import ConfigError
from anvil.logging import get_logger

if TYPE_CHECKING:
    from anvil.engine.public import Engine

EngineChoice = Literal["auto", "hf", "vllm"]

_log = get_logger(__name__)


def build_engine(
    model_id: str,
    *,
    engine: EngineChoice = "auto",
    revision: str | None = None,
    dtype: str | None = None,
    device_map: str | None = None,
    engine_args: dict[str, Any] | None = None,
) -> Engine:
    """Construct an engine for ``model_id``.

    Args:
        model_id: a HF Hub identifier (``"meta-llama/Llama-3.1-8B-Instruct"``)
            or a local path.
        engine: ``"auto"``, ``"hf"``, or ``"vllm"``. ``"auto"`` falls back to
            ``"hf"`` in M0; in M1+ it prefers vLLM when available.
        revision: optional model revision pin.
        dtype: ``"bfloat16"``, ``"float16"``, ``"float32"``, or ``None`` for auto.
        device_map: passed through to ``AutoModelForCausalLM.from_pretrained``;
            most users want ``"auto"``.
        engine_args: backend-specific extras (e.g. ``tensor_parallel_size``
            for vLLM).

    Raises:
        ConfigError: if ``engine="vllm"`` is requested in M0 (not yet
            implemented).
    """
    engine_args = engine_args or {}

    if engine == "vllm":
        from anvil.engine._vllm.adapter import VLLMEngine

        return VLLMEngine(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            engine_args=engine_args,
        )

    if engine == "auto":
        # M1 default: prefer vLLM if it is installed and CUDA is available;
        # otherwise fall back to the HF slow path. We do not gate on the
        # architecture being on the fast list — that's M6's call (the slow
        # path is correct on every architecture, just slower).
        engine = _auto_select(engine_args)
        _log.info("engine='auto' resolved to %r", engine)

    if engine == "hf":
        from anvil.engine._hf.runner import HFEngine

        return HFEngine(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            device_map=device_map or "auto",
            engine_args=engine_args,
        )

    raise ConfigError(f"unknown engine choice: {engine!r}")


def _auto_select(engine_args: dict[str, Any] | None) -> EngineChoice:
    """Pick a backend when ``engine='auto'``.

    Rules:

    * If vLLM is importable AND CUDA is available, choose ``vllm``.
    * Otherwise choose ``hf`` (the slow path always works).

    The choice is logged so the manifest's ``engine`` field documents what
    actually ran.
    """
    del engine_args
    try:
        import vllm  # noqa: F401
    except ImportError:
        return "hf"
    try:
        import warnings

        import torch

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuda_ok = torch.cuda.is_available()
    except (ImportError, RuntimeError, OSError):
        return "hf"
    return "vllm" if cuda_ok else "hf"


__all__ = ["build_engine", "EngineChoice"]
