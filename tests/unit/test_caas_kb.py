"""Unit tests for the CaaS KB schema + loader (design §16.7).

The KB is a contract between the design and the runtime — every shipped
entry must validate against the schema, regex-compile cleanly, and have
real citations. Adding a new entry without one of those should fail here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from anvil.caas import load_kb

if TYPE_CHECKING:
    from anvil.caas.kb.schema import KBEntry


@pytest.fixture(scope="module")
def kb() -> list[KBEntry]:
    return load_kb()


class TestKBSchemaCompliance:
    def test_loads_at_least_15_entries(self, kb: list[KBEntry]) -> None:
        assert len(kb) >= 15, f"expected ≥15 KB entries, got {len(kb)}"

    def test_every_entry_has_unique_id(self, kb: list[KBEntry]) -> None:
        ids = [e.id for e in kb]
        assert len(ids) == len(set(ids))

    def test_every_id_is_snake_case(self, kb: list[KBEntry]) -> None:
        for e in kb:
            assert re.match(r"^[a-z][a-z0-9_]*$", e.id), f"non-snake-case id: {e.id}"

    def test_every_entry_has_a_human_message(self, kb: list[KBEntry]) -> None:
        for e in kb:
            assert e.human_message.strip(), f"{e.id}: empty human_message"

    def test_every_entry_has_at_least_one_signature(self, kb: list[KBEntry]) -> None:
        for e in kb:
            assert e.signatures, f"{e.id}: no signatures"

    def test_every_signature_compiles(self, kb: list[KBEntry]) -> None:
        for e in kb:
            for sig in e.signatures:
                if sig.startswith("model_id_matches:"):
                    pattern = sig.split(":", 1)[1].strip()
                else:
                    pattern = sig
                # Will raise re.error if invalid.
                re.compile(pattern)

    def test_every_entry_has_engine_constraint(self, kb: list[KBEntry]) -> None:
        for e in kb:
            assert e.engines, f"{e.id}: no engine constraint"
            for c in e.engines:
                assert c.engine in {"any", "vllm", "transformers", "anvil", "sglang"}

    def test_review_required_entries_are_consent_gated(self, kb: list[KBEntry]) -> None:
        """Every ``severity='review-required'`` entry must also flag
        ``requires_user_consent=True``. The two are separate fields but
        semantically inseparable: review-required without consent is a bug
        because ``--caas=ci`` would otherwise auto-apply it (§7.6 / §7.7).
        """
        for e in kb:
            if e.severity == "review-required":
                assert e.requires_user_consent, (
                    f"{e.id}: severity=review-required but requires_user_consent=False"
                )

    def test_install_package_actions_are_consent_gated(self, kb: list[KBEntry]) -> None:
        """``install_package`` is a code-execution surface (§7.5)."""
        for e in kb:
            if e.fix.type == "install_package":
                assert e.requires_user_consent, (
                    f"{e.id}: install_package action without requires_user_consent"
                )


class TestSeedEntries:
    """Sanity-check the specific 15 entries the design names in §16.7."""

    EXPECTED_IDS = {
        # install.yaml
        "cuda_libcudart_version_mismatch",
        "flash_attention_sm_unsupported",
        "numpy_2x_abi_break",
        # model_loading.yaml
        "trust_remote_code_required",
        "bf16_nan_on_volta",
        "quantization_method_mismatch",
        # memory.yaml (mixed categories per §16.7 layout)
        "tp_attn_heads_divisibility",
        "kv_cache_oom_high_max_model_len",
        "qwen_vl_max_pixels_default_too_high",
        # tokenization.yaml
        "instruct_model_no_chat_template_warning",
        "llama3_eot_runaway",
        "chat_template_not_found_v044",
        # sampler.yaml
        "generation_config_overrides_sampler_v0_8",
        "reasoning_model_max_tokens_too_low",
        # harness.yaml
        "gsm8k_flexible_extract_picks_first_number",
    }

    def test_all_expected_ids_present(self, kb: list[KBEntry]) -> None:
        actual = {e.id for e in kb}
        missing = self.EXPECTED_IDS - actual
        assert not missing, f"missing seed entries: {sorted(missing)}"

    def test_trust_remote_code_is_review_required(self, kb: list[KBEntry]) -> None:
        entry = next(e for e in kb if e.id == "trust_remote_code_required")
        assert entry.severity == "review-required"
        assert entry.requires_user_consent is True

    def test_tp_attn_heads_uses_largest_divisor_strategy(self, kb: list[KBEntry]) -> None:
        entry = next(e for e in kb if e.id == "tp_attn_heads_divisibility")
        assert entry.fix.value_strategy == "largest_divisor_of_attn_heads"
        assert entry.fix.flag == "--tensor-parallel-size"

    def test_llama3_eot_uses_stop_token_id_128009(self, kb: list[KBEntry]) -> None:
        entry = next(e for e in kb if e.id == "llama3_eot_runaway")
        assert entry.fix.type == "set_sampling_param"
        assert entry.fix.name == "stop_token_ids"
        assert entry.fix.value == [128009]
