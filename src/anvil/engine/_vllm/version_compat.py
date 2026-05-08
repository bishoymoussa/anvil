"""vLLM CLI/API compatibility (design §3.1 / §16.1).

vLLM's CLI and Python API change every few minor releases (#28669, V0/V1
transition, generation-config default flip in v0.8.0, etc.). The wrapper
layer pins to a single vLLM version per Anvil release and normalizes the
surface against it.

For Anvil v0 the pinned version is **vllm==0.20.1**. Bumping requires:

1. Running ``tests/integration/test_milestone_1_*`` against the new vLLM.
2. Updating the kwarg map below if any constructor arg renamed.
3. Recording the bump in ``docs/design.md`` § (changelog TBD).

This module deliberately has *no* runtime dependency on vLLM — the maps
are pure data. The actual import + construction lives in :mod:`adapter`.
"""

from __future__ import annotations

from typing import Any, Final

PINNED_VLLM_VERSION: Final[str] = "0.20.1"
"""The vLLM version Anvil v0 is verified against."""


# Anvil-internal name → vLLM kwarg name for ``vllm.LLM(**kwargs)``.
# Keep this exhaustive of every flag we use; new flags must be added here
# rather than passed through opaquely.
LLM_CONSTRUCTOR_KWARGS: Final[dict[str, str]] = {
    "model_id": "model",
    "revision": "revision",
    "dtype": "dtype",
    "tensor_parallel_size": "tensor_parallel_size",
    "gpu_memory_utilization": "gpu_memory_utilization",
    "max_model_len": "max_model_len",
    "trust_remote_code": "trust_remote_code",
    "quantization": "quantization",
    "seed": "seed",
    "enforce_eager": "enforce_eager",
    "disable_log_stats": "disable_log_stats",
    "generation_config": "generation_config",
}


# Anvil-internal name → ``vllm.SamplingParams(**kwargs)`` kwarg name.
SAMPLING_PARAMS_KWARGS: Final[dict[str, str]] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "repetition_penalty": "repetition_penalty",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "max_tokens": "max_tokens",
    "seed": "seed",
    "stop": "stop",
    "stop_token_ids": "stop_token_ids",
    "n": "n",
    "logprobs": "logprobs",
    "prompt_logprobs": "prompt_logprobs",
}


def llm_kwargs(internal: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anvil-internal flag dict to vLLM ``LLM`` constructor kwargs.

    Unknown internal keys raise ``KeyError`` rather than silently passing
    through — that is the whole point of pinning.
    """
    out: dict[str, Any] = {}
    for k, v in internal.items():
        if k not in LLM_CONSTRUCTOR_KWARGS:
            raise KeyError(
                f"unknown vLLM LLM constructor flag {k!r}; add it to "
                "anvil/engine/_vllm/version_compat.py:LLM_CONSTRUCTOR_KWARGS "
                "after verifying it exists in vllm=={ver}".format(ver=PINNED_VLLM_VERSION)
            )
        out[LLM_CONSTRUCTOR_KWARGS[k]] = v
    return out


def sampling_kwargs(internal: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anvil-internal sampler dict to ``vllm.SamplingParams`` kwargs."""
    out: dict[str, Any] = {}
    for k, v in internal.items():
        if k not in SAMPLING_PARAMS_KWARGS:
            raise KeyError(
                f"unknown vLLM SamplingParams flag {k!r}; add it to "
                "anvil/engine/_vllm/version_compat.py:SAMPLING_PARAMS_KWARGS "
                f"after verifying it exists in vllm=={PINNED_VLLM_VERSION}"
            )
        out[SAMPLING_PARAMS_KWARGS[k]] = v
    return out


__all__ = [
    "PINNED_VLLM_VERSION",
    "LLM_CONSTRUCTOR_KWARGS",
    "SAMPLING_PARAMS_KWARGS",
    "llm_kwargs",
    "sampling_kwargs",
]
