"""HF non-causal slow-path engine for embedding / classification (design §6.7).

Drives a model loaded via ``transformers.AutoModel`` (or any user-supplied
``model_class``) to produce pooled hidden representations. The four pooling
strategies (``mean``, ``cls``, ``last``, ``max``, ``none``) cover the
patterns the non-text modalities in §6.7 use — RNA (mean), protein (cls,
last), audio (mean), embeddings (none).

Generation requests raise — embed engines are not causal LMs. Use
``anvil.load(...)`` for chat/generation models.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import torch

from anvil.exceptions import EngineError, ModelLoadError, TaskError
from anvil.logging import get_logger
from anvil.primitives.response import EmbedResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from anvil.primitives.request import Classify, Embed, Generate, LogLikelihood
    from anvil.primitives.response import ClassifyResult, Generation

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
        raise ValueError(f"unknown dtype {name!r}")
    return _DTYPE_MAP[name]


class HFEmbedEngine:
    """Embed-only engine for non-causal models (design §6.7).

    Constructed via :func:`anvil.load_custom`. Lazily-imports
    ``transformers.AutoModel`` / ``AutoTokenizer`` (or whatever
    ``model_class`` the caller supplied — RNA-FM uses ``AutoModel`` from
    multimolecule, protein models use ``AutoModel`` from transformers,
    etc.).
    """

    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None = None,
        dtype: str | None = None,
        device_map: str = "auto",
        model_class: Any = None,
        tokenizer_class: Any = None,
        engine_args: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self._dtype_name = dtype
        self._dtype = _resolve_dtype(dtype)
        self._engine_args = dict(engine_args or {})

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        mod_cls = model_class or AutoModel
        tok_cls = tokenizer_class or AutoTokenizer

        try:
            self.config = AutoConfig.from_pretrained(model_id, revision=revision)
            self._tokenizer = tok_cls.from_pretrained(
                model_id, revision=revision, trust_remote_code=True
            )
            self._model = mod_cls.from_pretrained(
                model_id,
                revision=revision,
                torch_dtype=self._dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"failed to load custom model {model_id!r}: {exc}") from exc

        self._model.eval()
        self._device = next(self._model.parameters()).device
        self._architecture = type(self._model).__name__

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

        payload = f"hf_embed:{transformers.__version__}:dtype={self._dtype}".encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    @property
    def backend_info(self) -> dict[str, str]:
        import transformers

        return {
            "name": "hf_embed",
            "version": transformers.__version__,
            "backend_hash": self.backend_hash,
        }

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def shutdown(self) -> None:
        del self._model
        if torch.cuda.is_available():  # pragma: no cover - hardware-dependent
            torch.cuda.empty_cache()

    # --------------------------------------------------------------- embed
    @torch.inference_mode()
    def embed(self, requests: list[Embed]) -> list[EmbedResult]:
        """Pooled hidden-states for each request.

        Inputs:
        * Strings — tokenized with the model's tokenizer; ``input_ids`` and
          ``attention_mask`` go to the model.
        * Tensors — passed through directly (caller's responsibility to
          shape correctly).

        Outputs:
        * ``EmbedResult`` carrying ``embedding`` (a ``[hidden_dim]`` 1-D
          tensor on CPU), the requested ``layer``, and the pooling strategy.
        """
        if not requests:
            return []
        results: list[EmbedResult] = []
        for req in requests:
            if isinstance(req.input, str):
                enc = self._tokenizer(
                    req.input,
                    return_tensors="pt",
                    truncation=True,
                    max_length=getattr(self.config, "max_position_embeddings", 4096),
                )
                input_ids = enc["input_ids"].to(self._device)
                attention_mask = enc.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self._device)
            elif isinstance(req.input, torch.Tensor):
                input_ids = req.input.to(self._device)
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                attention_mask = None
            else:
                raise TaskError(
                    f"Embed.input must be str or torch.Tensor; got {type(req.input).__name__}"
                )

            try:
                out = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            except RuntimeError as exc:
                raise EngineError(f"HF embed forward failed: {exc}") from exc

            hidden_states = getattr(out, "hidden_states", None)
            if hidden_states is None:
                # Some models return ``last_hidden_state`` directly when
                # ``output_hidden_states`` isn't honored.
                last_hidden = getattr(out, "last_hidden_state", None)
                if last_hidden is None:
                    raise EngineError(
                        "model returned no hidden states; pass a model that "
                        "exposes hidden_states or last_hidden_state"
                    )
                layer_tensor = last_hidden
            else:
                layer_tensor = hidden_states[req.layer]

            pooled = _pool(layer_tensor, attention_mask, req.pool)
            results.append(
                EmbedResult(
                    embedding=pooled.detach().cpu().float(),
                    layer=req.layer,
                    pool=req.pool,
                    input_token_count=int(input_ids.shape[1]),
                )
            )
        return results

    # -------------------------------------------------- raises for the rest
    def generate_logprobs(self, requests: list[Generate], top_k: int = 5) -> list[Generation]:
        del requests, top_k
        raise EngineError(
            "HFEmbedEngine does not support generation. Use anvil.load(...) for "
            "chat/generation models."
        )

    def generate_until(self, requests: list[tuple[str, list[str]]]) -> list[Generation]:
        del requests
        raise EngineError("HFEmbedEngine does not support generate_until")

    def loglikelihood(self, requests: list[LogLikelihood]) -> list[tuple[float, bool]]:
        del requests
        raise EngineError("HFEmbedEngine does not support loglikelihood")

    def loglikelihood_rolling(self, requests: list[str]) -> list[float]:
        del requests
        raise EngineError("HFEmbedEngine does not support loglikelihood_rolling")

    def classify(self, requests: list[Classify]) -> list[ClassifyResult]:
        del requests
        raise EngineError("HFEmbedEngine does not support classify in v0")

    def custom(self, fn: Callable[[list[Any]], list[Any]], inputs: list[Any]) -> list[Any]:
        return fn(inputs)


def _pool(hidden: torch.Tensor, mask: torch.Tensor | None, strategy: str) -> torch.Tensor:
    """Pool ``[B, T, D]`` hidden states to ``[D]`` per the named strategy."""
    if hidden.ndim == 3 and hidden.shape[0] == 1:
        squeezed = hidden[0]  # [T, D]
    else:
        squeezed = hidden.reshape(-1, hidden.shape[-1])

    if strategy == "none":
        return squeezed
    if strategy == "cls":
        return squeezed[0]
    if strategy == "last":
        return squeezed[-1]
    if strategy == "max":
        return squeezed.max(dim=0).values
    if strategy == "mean":
        if mask is not None:
            m = mask[0].to(squeezed.dtype).unsqueeze(-1)
            return (squeezed * m).sum(dim=0) / m.sum().clamp_min(1.0)
        return squeezed.mean(dim=0)
    raise ValueError(f"unknown pool strategy {strategy!r}")


__all__ = ["HFEmbedEngine"]
