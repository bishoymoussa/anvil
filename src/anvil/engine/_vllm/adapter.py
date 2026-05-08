"""vLLM backend adapter (design §3.1).

Implements :class:`anvil.engine.public.Engine` over ``vllm.LLM``. Pinned to
vllm==0.20.1 (design §16.8); see ``version_compat.py`` for the
flag-translation tables.

This module imports vLLM **lazily** inside :class:`VLLMEngine.__init__` so
that ``import anvil`` does not require vLLM to be installed. If the user
asks for ``engine="vllm"`` and the import fails, we surface a clear
``EngineError`` pointing at the optional extra (``pip install anvil[vllm]``).

Three engine methods are first-class on this backend:

* :meth:`generate_logprobs` — wraps ``llm.generate`` with translated sampling.
* :meth:`generate_until` — same path; stop strings are honored via
  ``SamplingParams.stop``.
* :meth:`loglikelihood` — uses ``prompt_logprobs=1`` to score continuations
  in a single batched prefill, without any actual generation. This is what
  drives MMLU at engine throughput.

Embed/Classify and per-request logits processors raise — they belong on
the HF backend (M0) or the M2 wrapper layer respectively.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from anvil.engine._vllm import version_compat
from anvil.exceptions import EngineError, ModelLoadError, TaskError
from anvil.logging import get_logger
from anvil.primitives.response import Generation
from anvil.primitives.sampler import Sampler

if TYPE_CHECKING:  # pragma: no cover - import-time-only
    from collections.abc import Callable

    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from anvil.primitives.request import Classify, Embed, Generate, LogLikelihood
    from anvil.primitives.response import ClassifyResult, EmbedResult

_log = get_logger(__name__)


@dataclass
class _BatchedPromptLogprobs:
    """One row's slice of vLLM ``prompt_logprobs`` output."""

    token_ids: list[int]
    chosen_logprobs: list[float]
    argmax_token_ids: list[int]


class VLLMEngine:
    """vLLM-backed engine.

    Pinned to ``vllm==anvil.engine._vllm.version_compat.PINNED_VLLM_VERSION``;
    runtime version is checked against the pin in :meth:`__init__`.
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None = None,
        dtype: str | None = None,
        engine_args: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._dtype_name = dtype or "auto"
        self._engine_args = dict(engine_args or {})

        try:
            import vllm
        except ImportError as exc:  # pragma: no cover - extras-dependent
            raise EngineError(
                "vLLM is not installed. Install the optional extra: "
                '`uv pip install -e ".[vllm]"` (pinned to '
                f"vllm=={version_compat.PINNED_VLLM_VERSION})."
            ) from exc

        # Sanity-check the version pin so users see a clear error rather than
        # an obscure attribute error if they have an off-pin vLLM.
        actual = getattr(vllm, "__version__", "unknown")
        if actual != version_compat.PINNED_VLLM_VERSION:
            _log.warning(
                "vllm version %s differs from pinned %s; flag-translation "
                "may misbehave if the API drifted",
                actual,
                version_compat.PINNED_VLLM_VERSION,
            )
        self._vllm_version = actual

        kwargs = version_compat.llm_kwargs(
            {
                "model_id": model_id,
                **({"revision": revision} if revision else {}),
                "dtype": self._dtype_name,
                # Always force generation_config="vllm" — the design's neutral
                # sampler default; never let generation_config.json sneak in.
                # See design §1.2 / KB entry generation_config_overrides_sampler_v0_8.
                "generation_config": "vllm",
                **self._engine_args,
            }
        )

        try:
            self._llm = vllm.LLM(**kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"vllm.LLM(**{kwargs!r}) failed: {exc}") from exc

        # Tokenizer is exposed by vllm.LLM via .get_tokenizer() in 0.20.x.
        try:
            self._tokenizer: PreTrainedTokenizerBase = self._llm.get_tokenizer()
        except AttributeError as exc:  # pragma: no cover - api-drift safety
            raise EngineError(
                f"vllm.LLM.get_tokenizer() not available in vllm=={actual}; "
                "update anvil/engine/_vllm/version_compat.py."
            ) from exc

    # ------------------------------------------------------------ bookkeeping
    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def model_info(self) -> dict[str, Any]:
        # vLLM stores the HF config on llm_engine.model_config; the exact path
        # has historically varied. Use a conservative dict-stringify path.
        try:
            cfg = self._llm.llm_engine.model_config.hf_config
            cfg_str = str(cfg.to_dict() if hasattr(cfg, "to_dict") else cfg)
        except (AttributeError, RuntimeError):  # pragma: no cover
            cfg_str = self.model_id
        cfg_hash = "sha256:" + hashlib.sha256(cfg_str.encode()).hexdigest()
        try:
            arch = type(self._llm.llm_engine.model_config.hf_config).__name__
        except AttributeError:  # pragma: no cover
            arch = "Unknown"
        return {
            "id": self.model_id,
            "revision": self.revision or "main",
            "dtype": self._dtype_name,
            "quantization": self._engine_args.get("quantization"),
            "config_hash": cfg_hash,
            "architecture": arch,
        }

    @property
    def backend_hash(self) -> str:
        payload = f"vllm:{self._vllm_version}:dtype={self._dtype_name}".encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def backend_info(self) -> dict[str, str]:
        return {
            "name": "vllm",
            "version": self._vllm_version,
            "backend_hash": self.backend_hash,
        }

    def shutdown(self) -> None:
        # vLLM's LLM class doesn't expose an explicit close; dropping the ref
        # plus a CUDA cache flush is the documented pattern.
        del self._llm
        try:
            import torch

            if torch.cuda.is_available():  # pragma: no cover - hardware-dependent
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    # --------------------------------------------------------- generate paths
    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]:
        if not requests:
            return []
        sampler = self._common_sampler(requests)
        sp = self._make_sampling_params(sampler, top_k=top_k)
        prompts = [self._render_generate(req) for req in requests]

        try:
            outs = self._llm.generate(prompts, sp, use_tqdm=False)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"vllm.LLM.generate failed: {exc}") from exc

        return [self._to_generation(o) for o in outs]

    def generate_until(self, requests: list[tuple[str, list[str]]]) -> list[Generation]:
        if not requests:
            return []
        prompts = [p for p, _ in requests]
        # Build per-request sampling: stops vary, max_tokens shared.
        # vLLM 0.20 supports per-prompt SamplingParams via a list aligned with prompts.
        per_request_sps = [
            self._make_sampling_params(Sampler.greedy(), stop=untils, top_k=0)
            for _, untils in requests
        ]
        try:
            outs = self._llm.generate(prompts, per_request_sps, use_tqdm=False)
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"vllm.LLM.generate (until) failed: {exc}") from exc
        return [self._to_generation(o) for o in outs]

    # ------------------------------------------------------- log-likelihood
    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]:
        if not requests:
            return []
        # Tokenize each (context, continuation) pair into a single sequence and
        # remember the offset where the continuation starts. Then we send all
        # sequences to vLLM with prompt_logprobs=1 and max_tokens=1 — vLLM
        # returns logprobs for every prompt token, from which we sum the
        # continuation positions. One batched prefill, no generation.
        encoded: list[tuple[list[int], int]] = []
        for req in requests:
            ctx_ids = self._tokenizer.encode(req.context, add_special_tokens=False)
            full_ids = self._tokenizer.encode(
                req.context + req.continuation, add_special_tokens=False
            )
            encoded.append((full_ids, len(ctx_ids)))

        sp = self._make_sampling_params(Sampler.greedy(max_tokens=1), prompt_logprobs=1)
        prompt_token_ids = [ids for ids, _ in encoded]

        try:
            outs = self._llm.generate(
                {"prompt_token_ids": prompt_token_ids},
                sp,
                use_tqdm=False,
            )
        except TypeError:
            # Older 0.20.x signature: pass token_ids via prompts list.
            outs = self._llm.generate(
                prompts=prompt_token_ids,
                sampling_params=sp,
                use_tqdm=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise EngineError(f"vllm.LLM.generate (loglikelihood) failed: {exc}") from exc

        results: list[tuple[float, bool]] = []
        for (full_ids, ctx_len), out in zip(encoded, outs, strict=True):
            prompt_logprobs = getattr(out, "prompt_logprobs", None)
            if prompt_logprobs is None:
                raise EngineError(
                    "vLLM did not return prompt_logprobs; check vllm version pin "
                    f"({self._vllm_version} vs {version_compat.PINNED_VLLM_VERSION})."
                )
            sliced = self._slice_continuation(prompt_logprobs, full_ids, ctx_len)
            total = sum(sliced.chosen_logprobs)
            greedy = sliced.chosen_logprobs and sliced.chosen_logprobs == [
                lp
                for lp, tok, argmax in zip(
                    sliced.chosen_logprobs,
                    sliced.token_ids,
                    sliced.argmax_token_ids,
                    strict=True,
                )
                if tok == argmax
            ]
            results.append((float(total), bool(greedy)))
        return results

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        if not requests:
            return []
        encoded = [self._tokenizer.encode(s, add_special_tokens=False) for s in requests]
        sp = self._make_sampling_params(Sampler.greedy(max_tokens=1), prompt_logprobs=1)
        outs = self._llm.generate(
            {"prompt_token_ids": encoded},
            sp,
            use_tqdm=False,
        )
        out_scores: list[float] = []
        for ids, out in zip(encoded, outs, strict=True):
            prompt_logprobs = getattr(out, "prompt_logprobs", None) or []
            total = 0.0
            for offset, tok in enumerate(ids):
                if offset == 0 or prompt_logprobs[offset] is None:
                    continue
                token_lp = prompt_logprobs[offset].get(tok)
                if token_lp is not None:
                    total += float(getattr(token_lp, "logprob", token_lp))
            out_scores.append(total)
        return out_scores

    # ------------------------------------------- non-causal / escape hatches
    def embed(self, requests: list[Embed]) -> list[EmbedResult]:
        del requests
        raise EngineError(
            "Embed requests are not supported by the vLLM backend in v0. "
            "Use anvil.load_custom(...) with engine='hf' (M5)."
        )

    def classify(self, requests: list[Classify]) -> list[ClassifyResult]:
        del requests
        raise EngineError(
            "Classify requests are not supported by the vLLM backend in v0. "
            "Score via loglikelihood over each label as a continuation."
        )

    def custom(self, fn: Callable[[list[Any]], list[Any]], inputs: list[Any]) -> list[Any]:
        return fn(inputs)

    # -------------------------------------------------------------- helpers
    def _make_sampling_params(
        self,
        sampler: Sampler,
        *,
        stop: list[str] | None = None,
        top_k: int = 0,
        prompt_logprobs: int | None = None,
        logprobs: int | None = None,
    ) -> Any:
        del top_k  # vLLM uses sampler.top_k directly
        import vllm

        kwargs: dict[str, Any] = {
            "temperature": sampler.temperature,
            "top_p": sampler.top_p,
            "top_k": sampler.top_k,
            "max_tokens": sampler.max_tokens,
            "n": sampler.n,
        }
        if sampler.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = sampler.repetition_penalty
        if sampler.presence_penalty != 0.0:
            kwargs["presence_penalty"] = sampler.presence_penalty
        if sampler.frequency_penalty != 0.0:
            kwargs["frequency_penalty"] = sampler.frequency_penalty
        if sampler.seed is not None:
            kwargs["seed"] = sampler.seed
        if sampler.min_p > 0.0:
            kwargs["min_p"] = sampler.min_p
        if sampler.stop:
            kwargs["stop"] = list(sampler.stop)
        if stop:
            kwargs["stop"] = list(kwargs.get("stop") or []) + list(stop)
        if sampler.stop_token_ids:
            kwargs["stop_token_ids"] = list(sampler.stop_token_ids)
        if prompt_logprobs is not None:
            kwargs["prompt_logprobs"] = prompt_logprobs
        if logprobs is not None:
            kwargs["logprobs"] = logprobs
        translated = version_compat.sampling_kwargs(kwargs)
        return vllm.SamplingParams(**translated)

    def _common_sampler(self, requests: list[Generate]) -> Sampler:
        first = requests[0].sampler or Sampler.greedy()
        for r in requests[1:]:
            other = r.sampler or Sampler.greedy()
            if other.hash != first.hash:
                raise TaskError(
                    "VLLMEngine.generate_logprobs requires all requests in a "
                    "batch to share a Sampler; got different hashes. "
                    "Per-request samplers land with the wrapper layer in M2."
                )
        return first

    def _render_generate(self, req: Generate) -> str:
        if req.prompt is not None:
            return req.prompt
        assert req.messages is not None
        try:
            return str(
                self._tokenizer.apply_chat_template(
                    list(req.messages),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise TaskError(f"apply_chat_template failed for {self.model_id!r}: {exc}") from exc

    @staticmethod
    def _to_generation(out: Any) -> Generation:
        first = out.outputs[0]
        return Generation(
            text=getattr(first, "text", ""),
            token_ids=tuple(getattr(first, "token_ids", ())),
            finish_reason=str(getattr(first, "finish_reason", "stop")),
            prompt_token_count=len(getattr(out, "prompt_token_ids", []) or []),
        )

    @staticmethod
    def _slice_continuation(
        prompt_logprobs: list[Any],
        full_ids: list[int],
        ctx_len: int,
    ) -> _BatchedPromptLogprobs:
        """Extract continuation logprobs from vLLM's per-position prompt_logprobs.

        ``prompt_logprobs[i]`` is a dict ``{token_id: Logprob(logprob, rank)}``
        for position ``i`` of the prompt; entry 0 is None (no preceding context).
        """
        chosen: list[float] = []
        argmax_ids: list[int] = []
        cont_ids: list[int] = []
        for pos in range(ctx_len, len(full_ids)):
            entry = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
            if entry is None:
                continue
            tok = full_ids[pos]
            cont_ids.append(tok)
            tok_lp = entry.get(tok)
            chosen.append(float(getattr(tok_lp, "logprob", tok_lp or 0.0)))
            # Highest-ranked token at this position (rank 1).
            best_id = min(
                entry.keys(),
                key=lambda k: getattr(entry[k], "rank", 0),
                default=tok,
            )
            argmax_ids.append(int(best_id))
        return _BatchedPromptLogprobs(
            token_ids=cont_ids,
            chosen_logprobs=chosen,
            argmax_token_ids=argmax_ids,
        )


__all__ = ["VLLMEngine"]
