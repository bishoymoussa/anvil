"""Tests for the vLLM flag-translation tables (design §3.1).

These are the data-only maps in :mod:`anvil.engine._vllm.version_compat`.
We do not exercise vLLM itself here — that requires vllm installed and a
working CUDA — only the translation contract.
"""

from __future__ import annotations

import pytest

from anvil.engine._vllm import version_compat as vc


def test_pinned_version_is_present() -> None:
    assert vc.PINNED_VLLM_VERSION == "0.20.1"


class TestLLMKwargs:
    def test_translates_known_flags(self) -> None:
        out = vc.llm_kwargs(
            {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "dtype": "bfloat16",
            }
        )
        assert out == {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
        }

    def test_unknown_flag_rejected(self) -> None:
        with pytest.raises(KeyError, match="unknown vLLM LLM constructor flag"):
            vc.llm_kwargs({"some_made_up_flag": True})

    def test_empty_round_trip(self) -> None:
        assert vc.llm_kwargs({}) == {}


class TestSamplingKwargs:
    def test_translates_known_flags(self) -> None:
        out = vc.sampling_kwargs(
            {"temperature": 0.0, "top_p": 1.0, "max_tokens": 256, "stop_token_ids": [128009]}
        )
        assert out == {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "stop_token_ids": [128009],
        }

    def test_unknown_sampling_flag_rejected(self) -> None:
        with pytest.raises(KeyError, match="unknown vLLM SamplingParams flag"):
            vc.sampling_kwargs({"hallucinated_flag": 42})


def test_constructor_kwargs_table_is_alphabetically_safe() -> None:
    """Sanity: every key matches its value when names are intentionally identical.

    Anvil's flag names mostly mirror vLLM's. This test will fail if someone
    accidentally renames an entry on one side without the other (which is a
    common form of API-drift bug).
    """
    intentional_aliases = {"model_id": "model"}
    for k, v in vc.LLM_CONSTRUCTOR_KWARGS.items():
        if k in intentional_aliases:
            assert v == intentional_aliases[k]
        else:
            assert k == v, f"flag {k!r} maps to {v!r} (expected identity)"


def test_sampling_kwargs_table_is_identity() -> None:
    for k, v in vc.SAMPLING_PARAMS_KWARGS.items():
        assert k == v, f"sampling flag {k!r} maps to {v!r} (expected identity)"
