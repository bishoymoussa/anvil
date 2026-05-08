"""M1 acceptance tests (design §16.10).

The literal manuscript tests are:

* ``test_milestone_1_chat_template_hash_stable`` — ChatTemplate hash is
  the same across two ``from_model`` calls. **Runs in the default suite**:
  no GPU needed; HF Hub network access is required, so we wrap in
  :func:`pytest.mark.requires_network` and provide an offline fallback
  that uses a fixture template string.

* ``test_milestone_1_vllm_matches_hf_within_tolerance`` — needs both
  vLLM and HF backends running real Llama-3.1-8B on real MMLU. Marked
  ``requires_hf_gated`` + ``requires_gpu`` + ``requires_vllm`` + ``slow``.

* ``test_milestone_1_known_baseline`` — Llama-3.1-8B-Instruct on MMLU
  5-shot ≈ 0.69 ± 0.01 (published). Same hardware constraints.

The default ``pytest`` invocation runs the offline fallback for
chat-template stability and skips the live tests. Running the live tests
needs ``HF_TOKEN``, a working CUDA, and ``uv pip install -e ".[vllm]"``.
"""

from __future__ import annotations

import os

import pytest

import anvil

# ---------------------------------------------------------------- offline


def test_milestone_1_chat_template_hash_stable_offline() -> None:
    """Two ChatTemplates with identical source must produce identical hashes.

    Doesn't require any network — the M0 unit tests cover this for trivial
    sources; this M1 test uses a more realistic chat-template shape (Qwen
    2.5-style) inline so we don't need the Hub.
    """
    qwen_like = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}"
        "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% else %}"
        "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    a = anvil.ChatTemplate(jinja_source=qwen_like, name="qwen2.5-instruct@v1")
    b = anvil.ChatTemplate(jinja_source=qwen_like, name="qwen2.5-instruct@v1")
    assert a.hash == b.hash
    # Cosmetic edits don't break the hash.
    cosmetic = qwen_like.replace("{% for ", "{%for ").replace(" %}", "%}")
    c = anvil.ChatTemplate(jinja_source=cosmetic)
    assert a.hash == c.hash


# ---------------------------------------------------------------- live (skipped)


@pytest.mark.requires_network
def test_milestone_1_chat_template_from_hub_stable() -> None:
    """The literal §16.10 test: load Qwen/Qwen2.5-7B-Instruct's chat template
    twice from the Hub and compare hashes.

    Requires network (Hub) but not GPU. The M0/M1 design says ``ChatTemplate``
    is reproducible across machines as a byte-stable hashed object — this
    test enforces it against a real model card.
    """
    ct1 = anvil.ChatTemplate.from_model("Qwen/Qwen2.5-7B-Instruct")
    ct2 = anvil.ChatTemplate.from_model("Qwen/Qwen2.5-7B-Instruct")
    assert ct1.hash == ct2.hash
    assert ct1.source in ("chat_template.json", "tokenizer_config.json")


@pytest.mark.requires_hf_gated
@pytest.mark.requires_gpu
@pytest.mark.requires_vllm
@pytest.mark.slow
def test_milestone_1_vllm_matches_hf_within_tolerance() -> None:
    """vLLM and HF agree to 0.5pp on Llama-3.1-8B-Instruct MMLU 5-shot/200 docs.

    Run with: ``pytest -m "requires_vllm and requires_hf_gated"``.
    """
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")
    cfg = dict(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tasks=["mmlu"],
        n_fewshot=5,
        limit=200,
        sampler=anvil.Sampler.greedy(),
    )
    a = anvil.eval(**cfg, engine="hf")  # type: ignore[arg-type]
    b = anvil.eval(**cfg, engine="vllm")  # type: ignore[arg-type]
    delta = abs(a.scores["mmlu"]["accuracy"] - b.scores["mmlu"]["accuracy"])
    assert delta < 0.005, f"vllm/hf disagree by {delta:.4f} on MMLU"


@pytest.mark.requires_hf_gated
@pytest.mark.requires_gpu
@pytest.mark.slow
def test_milestone_1_known_baseline() -> None:
    """Llama-3.1-8B-Instruct on MMLU 5-shot ≈ 0.69 ± 0.01 (published).

    The published baseline is the headline number Anvil's manifest is
    designed to reproduce. Drift outside ±0.01 means something silently
    regressed (chat template, sampler, EOS, fewshot exemplars) — that's the
    failure mode the design is built to prevent.
    """
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("HF_TOKEN not set")
    result = anvil.eval(
        model="meta-llama/Llama-3.1-8B-Instruct",
        tasks=["mmlu"],
        n_fewshot=5,
    )
    score = result.scores["mmlu"]["accuracy"]
    assert 0.68 < score < 0.70, f"MMLU 5-shot drifted to {score:.4f}"
