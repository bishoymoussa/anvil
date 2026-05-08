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
        # M1 introduces this. Surface explicitly rather than silently falling
        # back, so the user knows their request was not honored.
        raise ConfigError(
            "engine='vllm' is not implemented in M0; use engine='hf' or "
            "engine='auto' (which falls back to hf in M0). vLLM lands in M1 "
            "(design §16.10)."
        )

    if engine == "auto":
        _log.info("engine='auto' → 'hf' (M0 only ships the HF slow path).")
        engine = "hf"

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


__all__ = ["build_engine", "EngineChoice"]
