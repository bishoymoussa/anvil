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
        if _is_multimodal(model_id, revision=revision):
            from anvil.engine._hf.mm_runner import (
                HFVLMEngine,
                apply_qwen_vl_defaults,
                is_qwen_vl,
            )

            arch = _peek_architecture(model_id, revision=revision)
            effective_engine_args = dict(engine_args)
            if is_qwen_vl(arch):
                effective_engine_args = apply_qwen_vl_defaults(effective_engine_args)
                _log.info(
                    "Qwen-VL fast-path detected (%s); applying max_pixels/min_pixels defaults",
                    arch,
                )
            return HFVLMEngine(
                model_id=model_id,
                revision=revision,
                dtype=dtype,
                device_map=device_map or "auto",
                engine_args=effective_engine_args,
            )
        from anvil.engine._hf.runner import HFEngine

        return HFEngine(
            model_id=model_id,
            revision=revision,
            dtype=dtype,
            device_map=device_map or "auto",
            engine_args=engine_args,
        )

    raise ConfigError(f"unknown engine choice: {engine!r}")


def _peek_architecture(model_id: str, *, revision: str | None = None) -> str | None:
    """Cheap lookup of ``config.architectures[0]`` without weight download.

    Used by the factory to route VLMs vs causal LMs and by the Qwen-VL
    fast-path detector. Returns ``None`` on lookup failure (treat as
    not-a-VLM and fall through).
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, revision=revision)
        archs = getattr(cfg, "architectures", None) or []
        if archs:
            return str(archs[0])
        # Some VLM configs use ``model_type`` exclusively.
        return str(getattr(cfg, "model_type", "")) or None
    except Exception:  # noqa: BLE001
        return None


_VLM_ARCH_PATTERNS: tuple[str, ...] = (
    "Vision",
    "VL",
    "MultiModal",
    "ImageText",
    "Idefics",
    "LlavaForConditionalGeneration",
    "Llava",
    "Pixtral",
    "Phi3V",
    "Phi4Multi",
    "MiniCPMV",
    "InternVL",
    "Molmo",
    "CogVLM",
)


def _is_multimodal(model_id: str, *, revision: str | None = None) -> bool:
    """Best-effort: does this checkpoint declare a vision-language architecture?

    The check is **structural** — we read ``config.architectures[0]`` and
    look for known VLM markers. Substring matching is intentionally
    permissive; new VLM architectures usually carry one of the markers above.
    Day-zero coverage with the slow path is the design's promise (§3.3);
    misclassifying a text-only model as VLM is the worse failure (it would
    try to load a processor that doesn't exist), so the heuristic is
    conservative.
    """
    arch = _peek_architecture(model_id, revision=revision)
    if not arch:
        return False
    return any(marker in arch for marker in _VLM_ARCH_PATTERNS)


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
