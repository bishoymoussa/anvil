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

    from anvil.primitives.hidden_state_spec import HiddenStateSpec

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

    def _render_messages(self, messages: list[dict[str, Any]]) -> str:
        """Apply the chat template to a pre-built multi-turn message list.

        Used by :meth:`loglikelihood` when the request carries a ``messages``
        list (multi-turn fewshot). The template is applied with
        ``add_generation_prompt=True`` so the engine scores the continuation
        as the assistant turn's first tokens — identical semantics to the
        single-turn path but with the full conversation history visible.
        """
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except (AttributeError, ValueError, TypeError):
            # Fall back to concatenating user messages if the tokenizer
            # doesn't support apply_chat_template.
            return "\n".join(
                m["content"] for m in messages if isinstance(m.get("content"), str)
            )

    def _render_chat_context(self, context: str) -> str:
        """Wrap ``context`` as a user turn and apply the model's chat template.

        Used by :meth:`loglikelihood` when the request's ``chat_templated``
        flag is set. Mirrors lm-evaluation-harness's
        ``--apply_chat_template`` shape: one user message containing the
        full prompt (including any few-shot exemplars), followed by the
        ``add_generation_prompt`` separator. The continuation is then
        scored as the assistant turn's first tokens.

        Falls back to the bare context (no template) if the tokenizer
        doesn't expose ``apply_chat_template`` — base models don't ship
        one, and the caller's ``chat_templated`` flag is honored as
        best-effort rather than blocking.
        """
        try:
            return str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": context}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except (AttributeError, ValueError, TypeError):
            return context

    # -------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]:
        """Batched open-ended generation. Returns one :class:`Generation` per request.

        Padding is left-side; attention mask is the standard 0/1 inverse of
        pad positions; we use ``model.generate`` so HF handles the KV cache
        for us — what we own is *making sure the inputs are right*.

        If any request carries a :class:`~anvil.primitives.hidden_state_spec.HiddenStateSpec`,
        the forward pass runs with ``output_hidden_states=True`` and the
        requested layers / positions are extracted and returned in
        :attr:`~anvil.primitives.response.Generation.hidden_states`.
        """
        if not requests:
            return []
        prompts = [self._render_generate(r) for r in requests]
        sampler = self._resolve_sampler(requests[0])
        for r in requests[1:]:
            other = self._resolve_sampler(r)
            if other.hash != sampler.hash:
                raise TaskError(
                    "HFEngine.generate_logprobs requires all requests in a "
                    "batch to share a Sampler; got different hashes."
                )

        capture_spec: HiddenStateSpec | None = next(
            (r.capture for r in requests if r.capture is not None), None
        )

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(self.config, "max_position_embeddings", 8192),
            add_special_tokens=False,
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
        if capture_spec is not None:
            gen_kwargs["output_hidden_states"] = True
            gen_kwargs["return_dict_in_generate"] = True

        try:
            raw_out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        except RuntimeError as exc:
            raise EngineError(f"HF generate failed: {exc}") from exc

        # Unpack: with return_dict_in_generate the output is a ModelOutput.
        if capture_spec is not None:
            out_ids = raw_out.sequences  # type: ignore[union-attr]
            # hidden_states is a tuple of tuples: (step, layer, [B, seq, H])
            raw_hidden: tuple[tuple[torch.Tensor, ...], ...] = getattr(
                raw_out, "hidden_states", ()
            )
        else:
            out_ids = raw_out
            raw_hidden = ()

        results: list[Generation] = []
        for i, full in enumerate(out_ids):
            new_ids = full[input_ids.shape[1] :].tolist()
            new_ids = _strip_trailing(new_ids, stop_token_ids)
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            finish = "stop" if (new_ids and new_ids[-1] in stop_token_ids) else "length"

            hidden: dict[int, torch.Tensor] = {}
            if capture_spec is not None and raw_hidden:
                hidden = _extract_hidden_states(
                    raw_hidden=raw_hidden,
                    spec=capture_spec,
                    batch_idx=i,
                    prompt_len=int(prompt_lengths[i]),
                    gen_len=len(new_ids),
                )

            results.append(
                Generation(
                    text=text,
                    token_ids=tuple(new_ids),
                    finish_reason=finish,
                    prompt_token_count=int(prompt_lengths[i]),
                    hidden_states=hidden,
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
    LOGLIKELIHOOD_BATCH_SIZE: int = 8
    """Per-forward-pass row count. Small enough to fit big-vocab logits in 24GB."""

    @torch.inference_mode()
    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]:
        """Batched (context, continuation) log-likelihood.

        Tokenizes each pair, packs as many as fit into a fixed-size batch,
        runs a single forward, and reads logits at the continuation positions.
        Left-padding so the continuation suffix is contiguous on the right
        (so position offsets are simple).

        ``is_greedy`` is True iff ``argmax(logits) == continuation_token`` at
        every continuation position.
        """
        if not requests:
            return []

        # Pre-tokenize all pairs so we know the shapes upfront.
        # When ``chat_templated`` is set on the request, we wrap the
        # context as a single user message and apply the model's chat
        # template (with ``add_generation_prompt=True``) before encoding.
        # The continuation is then scored as the assistant turn's first
        # tokens — the published-baseline configuration for instruct
        # models, addressing lm-eval-harness #1841 by construction.
        pairs: list[tuple[list[int], list[int]]] = []  # (ctx_ids, full_ids)
        for req in requests:
            if req.messages is not None:
                # Multi-turn fewshot: the messages list is already the full
                # conversation; apply the chat template directly.
                rendered = self._render_messages(list(req.messages))
            elif req.chat_templated:
                rendered = self._render_chat_context(req.context)
            else:
                rendered = req.context
            ctx_ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            full_ids = self.tokenizer.encode(rendered + req.continuation, add_special_tokens=False)
            pairs.append((ctx_ids, full_ids))

        results: list[tuple[float, bool]] = []
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0

        for batch_start in range(0, len(pairs), self.LOGLIKELIHOOD_BATCH_SIZE):
            batch = pairs[batch_start : batch_start + self.LOGLIKELIHOOD_BATCH_SIZE]

            # Skip rows whose continuation is empty (degenerate); they score 0.
            empty_indices = [i for i, (ctx, full) in enumerate(batch) if len(full) <= len(ctx)]
            non_empty = [
                (i, ctx, full) for i, (ctx, full) in enumerate(batch) if len(full) > len(ctx)
            ]
            if not non_empty:
                results.extend((0.0, True) for _ in batch)
                continue

            max_len = max(len(full) for _, _, full in non_empty)
            input_ids = torch.full(
                (len(non_empty), max_len), pad_id, dtype=torch.long, device=self._device
            )
            attention_mask = torch.zeros(
                (len(non_empty), max_len), dtype=torch.long, device=self._device
            )
            for row, (_orig_idx, _ctx, full) in enumerate(non_empty):
                offset = max_len - len(full)
                input_ids[row, offset:] = torch.tensor(full, device=self._device)
                attention_mask[row, offset:] = 1

            try:
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits  # [B, max_len, vocab]
            except RuntimeError as exc:
                raise EngineError(f"HF forward (loglikelihood) failed: {exc}") from exc

            # Per-row scoring.
            row_results: dict[int, tuple[float, bool]] = {}
            for row, (orig_idx, ctx, full) in enumerate(non_empty):
                offset = max_len - len(full)
                cont_start = offset + len(ctx)
                cont_ids = full[len(ctx) :]
                # We need logits[row, cont_start - 1 : cont_start - 1 + len(cont_ids)]
                # — each predicts the token at cont_start + j for j in [0, len(cont_ids)).
                first_pos = cont_start - 1
                last_pos = first_pos + len(cont_ids)  # exclusive
                slab = logits[row, first_pos:last_pos]  # [len(cont_ids), vocab]
                step_log_probs = torch.log_softmax(slab.float(), dim=-1)
                cont_tensor = torch.tensor(cont_ids, device=self._device)
                # Gather logprobs at the chosen tokens.
                chosen = step_log_probs.gather(-1, cont_tensor.unsqueeze(-1)).squeeze(-1)
                total = float(chosen.sum().item())
                argmaxes = step_log_probs.argmax(dim=-1)
                greedy = bool((argmaxes == cont_tensor).all().item())
                row_results[orig_idx] = (total, greedy)

            for idx in range(len(batch)):
                if idx in empty_indices:
                    results.append((0.0, True))
                else:
                    results.append(row_results[idx])

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
        """Score each input against its label set via per-label log-likelihood.

        Each (input, label) pair becomes a LogLikelihood request with the
        input as context and the label as continuation. The label with the
        highest logprob wins.
        """
        if not requests:
            return []

        # Expand: one LogLikelihood per (request, label) combination.
        ll_requests: list[LogLikelihood] = []
        label_counts: list[int] = []
        for req in requests:
            inp = str(req.input)
            for label in req.label_set:
                ll_requests.append(LogLikelihood(context=inp, continuation=label))
            label_counts.append(len(req.label_set))

        ll_responses = self.loglikelihood(ll_requests)

        results: list[ClassifyResult] = []
        cursor = 0
        for req, count in zip(requests, label_counts, strict=True):
            chunk = ll_responses[cursor : cursor + count]
            cursor += count
            logprobs = {label: lp for label, (lp, _) in zip(req.label_set, chunk, strict=True)}
            best_label = max(logprobs, key=lambda k: logprobs[k])
            results.append(ClassifyResult(label=best_label, label_logprobs=logprobs))
        return results

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


def _extract_hidden_states(
    raw_hidden: tuple[tuple[torch.Tensor, ...], ...],
    spec: HiddenStateSpec,
    batch_idx: int,
    prompt_len: int,
    gen_len: int,
) -> dict[int, torch.Tensor]:
    """Slice per-layer hidden states out of HF's generate output format.

    HF returns ``hidden_states`` as a tuple of generation steps; each step
    is a tuple of layer tensors with shape ``[batch, seq_at_step, hidden]``.
    Step 0 covers the full prompt; subsequent steps cover one new token each.

    We concatenate across steps (dim=1) to get the full sequence per layer,
    then apply :attr:`HiddenStateSpec.positions` to select positions.
    """
    num_layers = len(raw_hidden[0]) if raw_hidden else 0
    total_len = prompt_len + gen_len

    # Resolve layer indices (support negatives).
    resolved_layers: list[int] = []
    for layer_idx in spec.layers:
        actual = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if 0 <= actual < num_layers:
            resolved_layers.append(actual)

    captured: dict[int, torch.Tensor] = {}
    for layer_idx in resolved_layers:
        # Collect this layer's tensor slice across all steps.
        step_slices: list[torch.Tensor] = []
        for step_tensors in raw_hidden:
            if layer_idx < len(step_tensors):
                # Shape: [batch, seq_at_step, hidden]
                step_slices.append(step_tensors[layer_idx][batch_idx])  # [seq, hidden]
        if not step_slices:
            continue
        full_seq = torch.cat(step_slices, dim=0)  # [total_len, hidden]

        # Apply position spec.
        positions = spec.positions
        if isinstance(positions, tuple):
            idx = torch.tensor(
                [p % total_len for p in positions], dtype=torch.long, device=full_seq.device
            )
            sliced = full_seq[idx]
        elif positions == "last":
            sliced = full_seq[-1:] if full_seq.shape[0] > 0 else full_seq
        elif positions == "first":
            sliced = full_seq[:1]
        else:  # "all", "image_tokens", "text_tokens" — return all for now
            sliced = full_seq

        # Cast if requested.
        if spec.dtype is not None:
            sliced = sliced.to(spec.dtype)
        if spec.pin_memory and sliced.is_cuda:
            sliced = sliced.cpu().pin_memory()
        elif sliced.is_cuda:
            sliced = sliced.cpu()

        # Store under the original (possibly negative) layer index for stable keys.
        original_idx = spec.layers[resolved_layers.index(layer_idx)]
        captured[original_idx] = sliced

    return captured


__all__ = ["HFEngine"]
