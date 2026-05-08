"""Unit tests for the CaaS rule engine (design §7.3)."""

from __future__ import annotations

import pytest

from anvil.caas import Context, load_kb, match
from anvil.caas.rule_engine import _largest_divisor


class TestLargestDivisor:
    def test_simple_cases(self) -> None:
        assert _largest_divisor(32, 4) == 4
        assert _largest_divisor(32, 3) == 2  # 3 doesn't divide; next is 2
        assert _largest_divisor(32, 5) == 4  # capped at 32, then 4 fits
        assert _largest_divisor(28, 8) == 7  # Qwen2.5 head count is 28

    def test_one_always_works(self) -> None:
        assert _largest_divisor(7, 2) == 1


class TestRuleEngineDispatch:
    def test_tp_divisibility_takes_precedence_over_model_id_only_match(self) -> None:
        """The bug we caught during M3 implementation: a Llama-3-Instruct
        TP-error must match ``tp_attn_heads_divisibility``, not the
        Llama-3-only ``llama3_eot_runaway``."""
        kb = load_kb()
        ctx = Context(
            error="Total number of attention heads (32) must be divisible by tensor parallel size (3)",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine_name="vllm",
            engine_version="0.20.1",
            available_gpus=4,
            num_attention_heads=32,
        )
        m = match(ctx, kb)
        assert m is not None
        assert m.entry.id == "tp_attn_heads_divisibility"
        assert m.action.flag == "--tensor-parallel-size"
        assert m.action.value == 4

    def test_llama3_eot_runaway_requires_both_model_and_error(self) -> None:
        kb = load_kb()
        # Llama-3-Instruct + the error trigger → matches.
        ctx = Context(
            error="generation_length >= max_tokens",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine_name="vllm",
            engine_version="0.20.1",
        )
        m = match(ctx, kb)
        assert m is not None and m.entry.id == "llama3_eot_runaway"

        # Same error but on a different model → no match (model-id filter rejects).
        ctx2 = Context(
            error="generation_length >= max_tokens",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            engine_name="vllm",
            engine_version="0.20.1",
        )
        m2 = match(ctx2, kb)
        # Qwen instruct model does match instruct_model_no_chat_template_warning IF
        # error contains 'no_chat_template_applied'. Our error is generation_length,
        # so no match, OR matches reasoning model entry. Let's just assert the
        # llama-3-only entry doesn't fire on Qwen.
        if m2 is not None:
            assert m2.entry.id != "llama3_eot_runaway"

    def test_engine_version_constraint_filters_entries(self) -> None:
        """``generation_config_overrides_sampler_v0_8`` should fire on
        vllm>=0.8 but not vllm 0.7.x."""
        kb = load_kb()
        ctx_match = Context(
            error="generation_config.json defaults applied",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine_name="vllm",
            engine_version="0.8.0",
        )
        m = match(ctx_match, kb)
        assert m is not None and m.entry.id == "generation_config_overrides_sampler_v0_8"

        # vllm 0.7.x — entry shouldn't fire (the engine constraint is >=0.8.0).
        ctx_no = Context(
            error="generation_config.json defaults applied",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            engine_name="vllm",
            engine_version="0.7.3",
        )
        m_no = match(ctx_no, kb)
        if m_no is not None:
            assert m_no.entry.id != "generation_config_overrides_sampler_v0_8"

    def test_no_match_returns_none(self) -> None:
        kb = load_kb()
        ctx = Context(
            error="totally unrelated error message that no signature matches",
            model_id="some-random/model",
            engine_name="vllm",
            engine_version="0.20.1",
        )
        m = match(ctx, kb)
        # Either None, or a model-id-only entry might match — check we don't
        # falsely fire on errors with no relevant signature.
        if m is not None:
            assert "model_id_matches:" in " ".join(m.entry.signatures), (
                "false positive: matched a non-model-id entry"
            )

    @pytest.mark.parametrize(
        # (heads, requested_tp, gpus_available, expected_largest_divisor)
        ("heads", "tp_size", "available_gpus", "expected_value"),
        [
            (32, 3, 4, 4),  # 4 divides 32, ≤ 4 → 4
            (32, 5, 4, 4),  # available is 4; same as above
            (28, 8, 8, 7),  # 8 doesn't divide 28; next ≤ 8 is 7
            (64, 3, 4, 4),  # 4 divides 64, ≤ 4 → 4
            (64, 5, 4, 4),  # same
        ],
    )
    def test_largest_divisor_via_rule_engine(
        self, heads: int, tp_size: int, available_gpus: int, expected_value: int
    ) -> None:
        kb = load_kb()
        ctx = Context(
            error=f"Total number of attention heads ({heads}) must be divisible by tensor parallel size ({tp_size})",
            model_id="some/model",
            engine_name="vllm",
            engine_version="0.20.1",
            available_gpus=available_gpus,
            num_attention_heads=heads,
        )
        m = match(ctx, kb)
        assert m is not None and m.entry.id == "tp_attn_heads_divisibility"
        assert m.action.value == expected_value
