"""HuggingFace transformers slow-path engine.

Implements :class:`anvil.engine.public.Engine` over
``transformers.AutoModelForCausalLM``. Correctness focus: padding,
attention-mask, and KV-cache handling are the things that go silently wrong
in batched HF generation, so we test them at batch sizes 1, 2, 8 and at
least one ragged batch (see :mod:`tests.unit.test_hf_runner_padding`).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from anvil.exceptions import EngineError, ModelLoadError, TaskError
from anvil.logging import get_logger
from anvil.primitives.request import Classify, Embed, Generate, LogLikelihood
from anvil.primitives.response import (
    ClassifyResult,
    EmbedResult,
    Generation,
)
from anvil.primitives.sampler import Sampler

if TYPE_CHECKING:
    from collections.abc import Callable

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

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


class HFEngine:
    """Engine backed by a single ``AutoModelForCausalLM`` instance.

    Single-process, single-device or single-host multi-GPU via
    ``device_map="auto"``. Multi-host parallelism is out of scope for the
    slow path (use vLLM in M1+ for that).
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None = None,
        dtype: str | None = None,
        device_map: str = "auto",
        engine_args: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._dtype_name = dtype
        self._engine_args = dict(engine_args or {})
        self._dtype = _resolve_dtype(dtype)

        _log.info(
            "loading %s via HF slow path (dtype=%s, device_map=%s)",
            model_id,
            self._dtype,
            device_map,
        )
        try:
            self.config = AutoConfig.from_pretrained(model_id, revision=revision)
            self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                model_id, revision=revision
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=self._dtype,
                device_map=device_map,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"failed to load {model_id!r}: {exc}") from exc

        # Decoder-only generation requires left-padding so that EOS detection
        # and stop-string matching see contiguous suffixes per row.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Use EOS as PAD when the tokenizer doesn't define one (Llama pattern).
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()  # type: ignore[no-untyped-call]
        self._device = next(self.model.parameters()).device
        self._architecture = type(self.model).__name__

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

        payload = f"hf:{transformers.__version__}:dtype={self._dtype}".encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def backend_info(self) -> dict[str, str]:
        import transformers

        return {
            "name": "hf",
            "version": transformers.__version__,
            "backend_hash": self.backend_hash,
        }

    def shutdown(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ tokenizing
    def _render_generate(self, req: Generate) -> str:
        """Resolve a Generate request to a single prompt string.

        For ``messages``, applies the tokenizer's chat_template. For
        ``prompt``, returns it as-is.
        """
        if req.prompt is not None:
            return req.prompt
        assert req.messages is not None  # guaranteed by Generate.__post_init__
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    list(req.messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise TaskError(
                f"apply_chat_template failed for {self.model_id!r}: {exc}. "
                "If this model is an instruct/chat checkpoint without a chat "
                "template, see the M0 README note on Llama-3 EOT handling."
            ) from exc

    def _resolve_sampler(self, req: Generate) -> Sampler:
        return req.sampler if req.sampler is not None else Sampler.greedy()

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]:
        """Batched open-ended generation. Returns one :class:`Generation` per request.

        Padding is left-side; attention mask is the standard 0/1 inverse of
        pad positions; we use ``model.generate`` so HF handles the KV cache
        for us — what we own is *making sure the inputs are right*.
        """
        if not requests:
            return []
        prompts = [self._render_generate(r) for r in requests]
        sampler = self._resolve_sampler(requests[0])
        # M0 invariant: all requests in a batch share a sampler. Multi-sampler
        # batching is M2 work (per-request logits processors).
        for r in requests[1:]:
            other = self._resolve_sampler(r)
            if other.hash != sampler.hash:
                raise TaskError(
                    "M0 HFEngine.generate_logprobs requires all requests in a "
                    "batch to share a Sampler; got different hashes. "
                    "Per-request samplers land with the wrapper layer in M2."
                )

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.config, "max_position_embeddings", 8192),
            add_special_tokens=False,  # the chat template emits BOS already
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        eos_ids = _eos_ids_for(self.tokenizer)
        stop_token_ids = list(eos_ids)
        for sid in sampler.stop_token_ids:
            if sid not in stop_token_ids:
                stop_token_ids.append(sid)

        gen_kwargs = _sampler_to_hf_kwargs(sampler, stop_token_ids)
        try:
            out_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        except RuntimeError as exc:
            raise EngineError(f"HF generate failed: {exc}") from exc

        results: list[Generation] = []
        for i, full in enumerate(out_ids):
            # The first ``input_ids.shape[1]`` positions are the (padded) prompt.
            new_ids = full[input_ids.shape[1] :].tolist()
            new_ids = _strip_trailing(new_ids, stop_token_ids)
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            finish = "stop" if (new_ids and new_ids[-1] in stop_token_ids) else "length"
            results.append(
                Generation(
                    text=text,
                    token_ids=tuple(new_ids),
                    finish_reason=finish,
                    prompt_token_count=prompt_lengths[i],
                )
            )
        return results

    def generate_until(self, requests: list[tuple[str, list[str]]]) -> list[Generation]:
        """Generate until any of ``until_strings`` appears.

        Implementation: defer to :meth:`generate_logprobs`, then truncate at
        the first occurrence of any stop string. Real streaming-stop is M1+.
        """
        if not requests:
            return []
        gen_reqs = [Generate(prompt=p) for p, _ in requests]
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
                )
            )
        return truncated

    # -------------------------------------------------------- log-likelihood
    @torch.inference_mode()
    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]:
        """Batched (context, continuation) log-likelihood.

        For each pair we encode ``context + continuation`` and read the
        model's logits at the continuation positions. ``is_greedy`` is True
        if ``argmax(logits) == continuation_token`` at every step.
        """
        if not requests:
            return []
        results: list[tuple[float, bool]] = []
        # Encode pairs individually because continuation lengths vary per pair.
        for req in requests:
            ctx_ids = self.tokenizer.encode(req.context, add_special_tokens=False)
            full_ids = self.tokenizer.encode(
                req.context + req.continuation, add_special_tokens=False
            )
            cont_ids = full_ids[len(ctx_ids) :]
            if not cont_ids:
                results.append((0.0, True))
                continue
            inp = torch.tensor([full_ids], device=self._device)
            logits = self.model(input_ids=inp).logits[0]  # [seq_len, vocab]
            log_probs = torch.log_softmax(logits, dim=-1)
            total = 0.0
            greedy = True
            for offset, tok in enumerate(cont_ids):
                pos = len(ctx_ids) + offset - 1
                if pos < 0:
                    continue  # degenerate empty-context case
                step_logp = log_probs[pos]
                total += float(step_logp[tok].item())
                if int(step_logp.argmax().item()) != tok:
                    greedy = False
            results.append((total, greedy))
        return results

    @torch.inference_mode()
    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        out: list[float] = []
        for s in requests:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if len(ids) < 2:
                out.append(0.0)
                continue
            inp = torch.tensor([ids], device=self._device)
            logits = self.model(input_ids=inp).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            total = 0.0
            for i in range(1, len(ids)):
                total += float(log_probs[i - 1, ids[i]].item())
            out.append(total)
        return out

    # ----------------------------------------------- embed/classify/custom
    def embed(self, requests: list[Embed]) -> list[EmbedResult]:
        # M5 lands the proper non-causal slow path. M0 surfaces the gap.
        del requests
        raise EngineError(
            "Embed requests are not supported by the HF causal-LM slow path. "
            "Use anvil.load_custom(..., model_class=AutoModel) once M5 lands."
        )

    def classify(self, requests: list[Classify]) -> list[ClassifyResult]:
        del requests
        raise EngineError(
            "Classify requests are not supported by the HF causal-LM slow path "
            "in M0. Score via loglikelihood over each label as a continuation."
        )

    def custom(self, fn: Callable[[list[Any]], list[Any]], inputs: list[Any]) -> list[Any]:
        return fn(inputs)


# ---------------------------------------------------------------- helpers


def _eos_ids_for(tk: PreTrainedTokenizerBase) -> tuple[int, ...]:
    """Collect EOS-equivalent ids for a tokenizer.

    Mirrors :func:`anvil.primitives.tokenization._collect_eos_ids` but lives
    here because the engine layer can't import from primitives' helpers
    (only their public dataclasses).
    """
    out: list[int] = []
    if tk.eos_token_id is not None:
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
    """Cut ``ids`` at the first stop-token occurrence (inclusive of the stop)."""
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[: i + 1]
    return ids


def _sampler_to_hf_kwargs(sampler: Sampler, stop_token_ids: list[int]) -> dict[str, Any]:
    """Translate :class:`Sampler` to ``model.generate`` kwargs."""
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
    if sampler.n != 1:
        kwargs["num_return_sequences"] = sampler.n
    return kwargs


__all__ = ["HFEngine"]
