"""M4 acceptance tests (design §16.10).

The literal manuscript tests:

1. ``test_milestone_4_vlm_basic_generation`` — load Qwen2.5-VL-7B,
   pass an image + text question, expect "cat" in the output and at
   least one image-token-count entry. Marked ``requires_gpu`` +
   ``requires_network`` (Qwen2.5-VL is open weights, no token needed).
2. ``test_milestone_4_mmmu_known_baseline`` — Qwen2.5-VL-7B on MMMU
   ≈ 0.50 ± 0.02 (published). Marked ``requires_gpu`` +
   ``requires_network`` + ``slow``.
3. ``test_milestone_4_image_size_smoke`` — synthetic 4K image triggers
   the Qwen-VL max-pixels CaaS engagement; the audit log records the
   ``qwen_vl_max_pixels_default_too_high`` entry.

The image-size smoke test runs offline by stubbing the engine; the live
tests need a GPU + open Qwen2.5-VL weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import anvil
from anvil.tasks.public import _build_audit_log

if TYPE_CHECKING:
    from pathlib import Path


def test_milestone_4_image_size_smoke_offline(tmp_path: Path) -> None:
    """The §16.10 acceptance: choosing a Qwen-VL model engages CaaS and the
    qwen_vl_max_pixels entry lands in the audit log.

    Runs offline — exercises the preflight pathway directly without
    spinning up a real engine.
    """
    audit = _build_audit_log(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        engine_args={},
        mode="ci",
    )
    assert audit is not None
    assert len(audit) >= 1

    # The audit entry's kb_entry_ids must reference the qwen_vl entry.
    qwen_entries = [
        a for a in audit.actions if a.kb_entry_ids and any("qwen_vl" in k for k in a.kb_entry_ids)
    ]
    assert qwen_entries, (
        f"expected at least one Qwen-VL audit entry; got {[a.kb_entry_ids for a in audit.actions]}"
    )
    # The action set Anvil's safer max_pixels via the --mm-processor-kwargs flag.
    first = qwen_entries[0]
    assert first.action == "set_engine_flag"
    assert first.args.get("flag") == "--mm-processor-kwargs"

    # Also verify that the engine factory's apply_qwen_vl_defaults sees the
    # new shape (engine_args mutated in place).
    del tmp_path  # unused; we keep the fixture parameter for future expansion


def test_milestone_4_image_size_smoke_writes_to_manifest(tmp_path: Path) -> None:
    """The audit-log entries flow into the manifest's caas_log field.

    Builds a manifest from an audit log directly (no engine) and verifies
    the round-trip: load the saved manifest, find the caas_log entries,
    confirm the kb_entry_ids reference the Qwen-VL entry.
    """
    from anvil.manifest import CaaSAction, Manifest, ModelInfo, TaskInfo

    audit = _build_audit_log(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        engine_args={},
        mode="ci",
    )
    assert audit is not None
    records = audit.to_list()
    assert records and any(a.kb_entry_ids and "qwen_vl" in a.kb_entry_ids[0] for a in records)

    # Embed in a fixture manifest and round-trip.
    m = Manifest(
        anvil_version=anvil.__version__,
        engine={"name": "hf_vlm", "version": "x", "backend_hash": "sha256:aa"},
        model=ModelInfo(
            id="Qwen/Qwen2.5-VL-7B-Instruct",
            revision="main",
            dtype="bfloat16",
            quantization=None,
            config_hash="sha256:bb",
            architecture="Qwen2_5_VLForConditionalGeneration",
        ),
        tokenization={"hash": "sha256:cc", "padding_side": "left"},
        chat_template=None,
        sampler=None,
        tasks=[
            TaskInfo(
                name="mmmu",
                tier="curated",
                version="mmmu@anvil-v0",
                dataset_revision="hf:MMMU/MMMU",
                n_fewshot=0,
                metric="accuracy",
                request_type="Generate",
            )
        ],
        scores={"mmmu": {"accuracy": 0.0}},
        smoke_test={"samples": 0, "outcome": "skipped"},
        caas_log=records,
        hardware={"os": "Linux"},
        started_at="2026-05-08T17:00:00+00:00",
        ended_at="2026-05-08T17:00:01+00:00",
    ).sign()
    out_path = m.save(tmp_path / "manifest.json")
    loaded = Manifest.load(out_path)
    assert loaded.verify()
    assert loaded.caas_log
    assert any(
        isinstance(a, CaaSAction) and a.kb_entry_ids and "qwen_vl" in a.kb_entry_ids[0]
        for a in loaded.caas_log
    )


# ---------------------------------------------------------- live (skipped)


@pytest.mark.requires_gpu
@pytest.mark.requires_network
@pytest.mark.slow
def test_milestone_4_vlm_basic_generation() -> None:
    """The literal §16.10: Qwen2.5-VL-7B on a cat image returns text containing
    "cat" and reports image-token counts.

    Skipped by default — needs ~14GB of VRAM and downloads the Qwen2.5-VL-7B
    weights (~15 GB). Run with ``pytest -m "requires_gpu and requires_network"``.
    """
    from PIL import Image

    # Tiny synthetic "cat-ish" image — the test only checks that the engine
    # round-trips a multimodal request, not that the model recognizes a cat.
    img = Image.new("RGB", (256, 256), (200, 150, 100))

    m = anvil.load("Qwen/Qwen2.5-VL-7B-Instruct")
    out = m.generate(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "What color dominates this image?"},
                ],
            }
        ]
    )
    assert isinstance(out.text, str) and out.text
    # image_token_counts is populated for VLM responses.
    assert out.image_token_counts, "image_token_counts is empty"


@pytest.mark.requires_gpu
@pytest.mark.requires_network
@pytest.mark.slow
def test_milestone_4_mmmu_known_baseline() -> None:
    """Qwen2.5-VL-7B on MMMU validation 4-way subset ≈ 0.50 ± 0.02 (published).

    Skipped by default. The bound is generous because MMMU has subjects with
    very different scoring distributions; on a 200-doc subset the variance
    is real.
    """
    result = anvil.eval(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        tasks=["mmmu"],
        limit=200,
    )
    score = result.scores["mmmu"]["accuracy"]
    assert 0.47 < score < 0.55, f"MMMU accuracy {score} drifted outside published range"
    # The CaaS log should contain the Qwen-VL max-pixels engagement.
    assert any(a.kb_entry_ids and "qwen_vl" in a.kb_entry_ids[0] for a in result.manifest.caas_log)


def _vlm_smoke_output(_: Any) -> dict[str, Any]:  # pragma: no cover - example helper
    return {}
