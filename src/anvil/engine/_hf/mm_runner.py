"""HuggingFace transformers slow-path engine for vision-language models (design §5.4).

VLMs route here when the user passes ``messages`` whose ``content`` is a
list of typed parts (``{"type": "image" | "text", …}``). The pipeline:

1. **Extract images** from each request.
2. **Preprocess** every image (RGBA→RGB, EXIF, sweet-spot resize) — see
   :mod:`anvil.primitives.multimodal`.
3. **Render** the chat template via the tokenizer's
   ``apply_chat_template``, but the processor (not the tokenizer) is
   responsible for building image-aware token sequences.
4. **Forward** through ``AutoProcessor`` → ``AutoModelForVision2Seq`` /
   ``AutoModelForImageTextToText``.
5. **Track image-token spans** so the manifest can record per-image
   token counts.

For v0 we rely on the model's ``Processor`` (the HF abstraction that
unifies tokenizer + image preprocessor) — every modern VLM ships one.
The actual fast-path with custom kernels is M6 (per design §5.2);
this slow path's job is to be correct for any architecture that loads in
``transformers.AutoModelForVision2Seq`` the day it ships.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import torch

from anvil.exceptions import EngineError, ModelLoadError, TaskError
from anvil.logging import get_logger
from anvil.primitives.multimodal import (
    ProcessedImage,
    collect_images,
    is_multimodal_message,
    preprocess_image,
    replace_images_with_processed,
)
from anvil.primitives.response import Generation
from anvil.primitives.sampler import Sampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from anvil.primitives.request import Classify, Embed, Generate, LogLikelihood
    from anvil.primitives.response import ClassifyResult, EmbedResult

_log = get_logger(__name__)


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(name: str | None) -> torch.dtype:
    if name is None:
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown dtype {name!r}; expected one of {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


class HFVLMEngine:
    """VLM-aware HF slow path. Implements :class:`anvil.engine.public.Engine`."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None = None,
        dtype: str | None = None,
        device_map: str = "auto",
        engine_args: dict[str, Any] | None = None,
        max_pixels: int | None = None,
        min_pixels: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._dtype_name = dtype
        self._engine_args = dict(engine_args or {})
        self._dtype = _resolve_dtype(dtype)

        # Image budget: respects --mm-processor-kwargs if the user pinned it
        # (the CaaS Qwen-VL fix sets these), else falls back to the
        # multimodal module's defaults.
        mm_kwargs = self._engine_args.get("mm_processor_kwargs", {}) or {}
        self.max_pixels = max_pixels or mm_kwargs.get("max_pixels")
        self.min_pixels = min_pixels or mm_kwargs.get("min_pixels")

        from transformers import AutoConfig, AutoProcessor

        try:
            self.config = AutoConfig.from_pretrained(model_id, revision=revision)
            self._processor = AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
                model_id, revision=revision
            )
            self._model = _load_vlm_model(
                model_id, revision=revision, dtype=self._dtype, device_map=device_map
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"failed to load VLM {model_id!r}: {exc}") from exc

        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._architecture = type(self._model).__name__

        # Best-effort EOS detection — same logic as the text path.
        tk = getattr(self._processor, "tokenizer", None)
        self._eos_ids: tuple[int, ...] = _eos_ids_for(tk) if tk is not None else ()

    # ----------------------------------------------------------- bookkeeping
    @property
    def model_info(self) -> dict[str, Any]:
        cfg_str = str(self.config.to_dict())
        cfg_hash = "sha256:" + hashlib.sha256(cfg_str.encode()).hexdigest()
        return {
            "id": self.model_id,
            "revision": self.revision or "main",
            "dtype": str(self._dtype).replace("torch.", ""),
            "quantization": None,
            "config_hash": cfg_hash,
            "architecture": self._architecture,
        }

    @property
    def backend_hash(self) -> str:
        import transformers

        payload = (
            f"hf_vlm:{transformers.__version__}:dtype={self._dtype}"
            f":max_pixels={self.max_pixels}:min_pixels={self.min_pixels}"
        ).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def backend_info(self) -> dict[str, str]:
        import transformers

        return {
            "name": "hf_vlm",
            "version": transformers.__version__,
            "backend_hash": self.backend_hash,
        }

    @property
    def tokenizer(self) -> Any:
        return getattr(self._processor, "tokenizer", None)

    def shutdown(self) -> None:
        del self._model
        if torch.cuda.is_available():  # pragma: no cover - hardware-dependent
            torch.cuda.empty_cache()

    # --------------------------------------------------------- generate path
    @torch.inference_mode()
    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]:
        del top_k
        if not requests:
            return []

        # 1) Preprocess every image once. Record (count, original_size,
        #    processed_size, hash) per image for the manifest.
        processed_per_request: list[list[ProcessedImage]] = []
        for req in requests:
            messages = list(req.messages or [])
            raws = collect_images(messages)
            processed_per_request.append(
                [
                    preprocess_image(
                        r,
                        max_pixels=self.max_pixels or 1280 * 768,
                        min_pixels=self.min_pixels or 56 * 56,
                    )
                    for r in raws
                ]
            )

        sampler = self._common_sampler(requests)

        # 2) Render each request via the processor (which handles both text
        #    and image-token expansion).
        rendered: list[dict[str, Any]] = []
        for req, processed in zip(requests, processed_per_request, strict=True):
            messages = list(req.messages or [])
            messages_with_imgs = replace_images_with_processed(messages, processed)
            text_prompt = self._render_text(messages_with_imgs)
            images_only = [p.image for p in processed]
            try:
                inputs = self._processor(
                    text=[text_prompt],
                    images=images_only or None,
                    return_tensors="pt",
                    padding=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise TaskError(f"VLM processor failed for {self.model_id!r}: {exc}") from exc
            rendered.append(inputs)

        # 3) Forward each request individually. Batched VLM forward with
        #    ragged image grids requires per-architecture stacking; the slow
        #    path runs them one-at-a-time for correctness — design §3.3
        #    explicitly trades throughput for "loads anything HF loads".
        outputs: list[Generation] = []
        for req, inputs, processed in zip(requests, rendered, processed_per_request, strict=True):
            del req
            inputs_on_dev = {
                k: (v.to(self._device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }
            stop_token_ids = list(self._eos_ids) + [
                t for t in sampler.stop_token_ids if t not in self._eos_ids
            ]
            gen_kwargs = _sampler_to_hf_kwargs(sampler, stop_token_ids)
            try:
                out_ids = self._model.generate(
                    **inputs_on_dev,
                    **gen_kwargs,
                )
            except RuntimeError as exc:
                raise EngineError(f"VLM generate failed: {exc}") from exc

            # The processor builds inputs whose ``input_ids`` includes the
            # prompt tokens. Slice off the prompt to get just the new tokens.
            prompt_len = inputs_on_dev["input_ids"].shape[1] if "input_ids" in inputs_on_dev else 0
            new_ids = out_ids[0, prompt_len:].tolist()
            new_ids = _strip_trailing(new_ids, stop_token_ids)
            text = self._processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            finish = "stop" if new_ids and new_ids[-1] in stop_token_ids else "length"

            image_token_counts = self._extract_image_token_counts(inputs_on_dev, processed)

            outputs.append(
                Generation(
                    text=text,
                    token_ids=tuple(new_ids),
                    finish_reason=finish,
                    prompt_token_count=prompt_len,
                    image_token_counts=tuple(image_token_counts),
                    extra={
                        "image_hashes": [p.hash for p in processed],
                        "image_original_sizes": [p.original_size for p in processed],
                        "image_processed_sizes": [p.processed_size for p in processed],
                    },
                )
            )
        return outputs

    def generate_until(self, requests: list[tuple[str, list[str]]]) -> list[Generation]:
        # VLM "generate-until" with text-only prompts is well-defined.
        # Defer to generate_logprobs and post-trim.
        if not requests:
            return []
        from anvil.primitives.request import Generate as _Generate

        gen_reqs = [_Generate(prompt=p) for p, _ in requests]
        outs = self.generate_logprobs(gen_reqs)
        truncated: list[Generation] = []
        for gen, (_, untils) in zip(outs, requests, strict=True):
            text = gen.text
            cut = len(text)
            for marker in untils:
                idx = text.find(marker)
                if idx >= 0 and idx < cut:
                    cut = idx
            truncated.append(
                Generation(
                    text=text[:cut],
                    token_ids=gen.token_ids,
                    finish_reason="stop" if cut < len(text) else gen.finish_reason,
                    prompt_token_count=gen.prompt_token_count,
                    image_token_counts=gen.image_token_counts,
                )
            )
        return truncated

    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]:
        # MMMU + similar use loglikelihood for letter-MCQ scoring even when
        # the prompt contains an image. We delegate to a single-pass forward
        # via the processor; see test_mmmu for end-to-end coverage.
        del requests
        raise EngineError(
            "VLM loglikelihood is implemented in the runner via Generate "
            "with explicit option-ranking; the engine method is not used "
            "directly in v0 (lands in M5 alongside lm-eval-shim)."
        )

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        del requests
        raise EngineError("loglikelihood_rolling is text-only; not supported by VLM engine")

    def embed(self, requests: list[Embed]) -> list[EmbedResult]:
        del requests
        raise EngineError("Embed not yet supported by VLM slow path (M5)")

    def classify(self, requests: list[Classify]) -> list[ClassifyResult]:
        del requests
        raise EngineError("Classify not yet supported by VLM slow path")

    def custom(self, fn: Callable[[list[Any]], list[Any]], inputs: list[Any]) -> list[Any]:
        return fn(inputs)

    # -------------------------------------------------------------- helpers
    def _render_text(self, messages: list[dict[str, Any]]) -> str:
        """Apply the model's chat template, preserving multimodal content parts."""
        try:
            return str(
                self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except AttributeError:
            # Older processors only expose tokenizer.apply_chat_template.
            tk = self._processor.tokenizer
            try:
                return str(
                    tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                )
            except Exception as exc:  # noqa: BLE001
                raise TaskError(f"apply_chat_template failed for {self.model_id!r}: {exc}") from exc

    def _common_sampler(self, requests: list[Generate]) -> Sampler:
        first = requests[0].sampler or Sampler.greedy()
        for r in requests[1:]:
            other = r.sampler or Sampler.greedy()
            if other.hash != first.hash:
                raise TaskError(
                    "VLM batch requires all requests to share a Sampler "
                    "(per-request samplers land in M2)"
                )
        return first

    def _extract_image_token_counts(
        self, inputs: dict[str, Any], processed: list[ProcessedImage]
    ) -> list[int]:
        """Best-effort per-image token count.

        Different VLM processors expose different keys: Qwen-VL uses
        ``image_grid_thw``; LLaVA exposes ``pixel_values`` plus inferred
        token counts from ``input_ids``. For v0 we look up the standard
        Qwen-VL key, then fall back to dividing the total vision-token
        count evenly. The number lands in :class:`Generation.image_token_counts`.
        """
        if not processed:
            return []
        grid = inputs.get("image_grid_thw")
        if grid is not None:
            try:
                tokens = grid.prod(dim=-1).tolist()
                return [int(t) for t in tokens]
            except (AttributeError, RuntimeError):  # pragma: no cover
                pass
        # Fallback: rough estimate based on processed pixel count.
        return [max(1, p.pixel_count // (28 * 28)) for p in processed]


def _load_vlm_model(
    model_id: str,
    *,
    revision: str | None,
    dtype: torch.dtype,
    device_map: str,
) -> Any:
    """Try Vision2Seq first; fall back to ImageTextToText (newer API).

    The two transformers classes overlap in coverage; the right one for a
    given checkpoint is whichever ``config.architectures[0]`` is registered
    against. We try both, emitting a clear error if neither works.
    """
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq

    last_err: Exception | None = None
    for cls in (AutoModelForVision2Seq, AutoModelForImageTextToText):
        try:
            return cls.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=dtype,
                device_map=device_map,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    raise EngineError(
        f"could not load {model_id!r} as a vision-language model. Last error: {last_err}"
    )


def _eos_ids_for(tk: Any) -> tuple[int, ...]:
    """Same EOS-collection logic as the text engine, copy-localized to
    avoid an upward import."""
    out: list[int] = []
    if getattr(tk, "eos_token_id", None) is not None:
        out.append(int(tk.eos_token_id))
    for marker in ("<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>"):
        try:
            tid = tk.convert_tokens_to_ids(marker)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(tid, int) and tid != tk.unk_token_id and tid not in out:
            out.append(tid)
    return tuple(out)


def _strip_trailing(ids: list[int], stop_ids: list[int]) -> list[int]:
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[: i + 1]
    return ids


def _sampler_to_hf_kwargs(sampler: Sampler, stop_token_ids: list[int]) -> dict[str, Any]:
    if sampler.temperature == 0.0:
        kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": sampler.max_tokens,
        }
    else:
        kwargs = {
            "do_sample": True,
            "temperature": sampler.temperature,
            "top_p": sampler.top_p,
            "top_k": sampler.top_k if sampler.top_k > 0 else 0,
            "repetition_penalty": sampler.repetition_penalty,
            "max_new_tokens": sampler.max_tokens,
        }
        if sampler.seed is not None:
            torch.manual_seed(sampler.seed)
    if stop_token_ids:
        kwargs["eos_token_id"] = stop_token_ids
    return kwargs


def is_vlm_request(messages: list[dict[str, Any]]) -> bool:
    """Quick predicate: are any of these messages multimodal?"""
    return any(is_multimodal_message(m) for m in messages)


# ---------------------- Qwen-VL family routing (engine-local helpers) ----------------------

QWEN_VL_ARCHITECTURES: tuple[str, ...] = (
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
)


def is_qwen_vl(architecture: str | None) -> bool:
    """True iff ``architecture`` is in the Qwen-VL family fast-path roster.

    Lives in the engine layer so the factory can query without violating
    the import hierarchy (engine cannot reach into ``anvil.models``).
    The ``models/_fast/qwen_vl.py`` registration is still the §5.2 source
    of truth for the public fast-path roster; this helper duplicates the
    architecture list deliberately.
    """
    return architecture in QWEN_VL_ARCHITECTURES


def apply_qwen_vl_defaults(engine_args: dict[str, Any]) -> dict[str, Any]:
    """Set Anvil's safer defaults for Qwen-VL models (design §5.4 / §7.7 KB).

    Caps ``max_pixels`` at 1280×768 instead of the upstream 12.8M-pixel
    default that triggers OOMs on consumer hardware. The user can still
    override via ``mm_processor_kwargs`` in ``engine_args`` — we only
    apply when the field is unset.
    """
    from anvil.primitives.multimodal import DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS

    out = dict(engine_args)
    mm_kwargs: dict[str, Any] = dict(out.get("mm_processor_kwargs") or {})
    mm_kwargs.setdefault("max_pixels", DEFAULT_MAX_PIXELS)
    mm_kwargs.setdefault("min_pixels", DEFAULT_MIN_PIXELS)
    out["mm_processor_kwargs"] = mm_kwargs
    return out


__all__ = ["HFVLMEngine", "is_vlm_request"]
