"""Tests for the manifest schema, signing, and canonical JSON (design §8, §16.5)."""

from __future__ import annotations

import json
import math

import pytest

from anvil import Manifest
from anvil.exceptions import ManifestError
from anvil.manifest.canonical import canonical_json
from anvil.manifest.schema import ModelInfo, TaskInfo


def _fixture(**overrides: object) -> Manifest:
    base: dict[str, object] = {
        "anvil_version": "0.0.1",
        "engine": {"name": "hf", "version": "4.57.6", "backend_hash": "sha256:aa"},
        "model": ModelInfo(
            id="test/model",
            revision="main",
            dtype="bfloat16",
            quantization=None,
            config_hash="sha256:bb",
            architecture="LlamaForCausalLM",
        ),
        "tokenization": {"hash": "sha256:cc", "padding_side": "left"},
        "chat_template": None,
        "sampler": {"hash": "sha256:dd", "temperature": 0.0, "max_tokens": 256},
        "tasks": [
            TaskInfo(
                name="gsm8k",
                tier="curated",
                version="gsm8k@anvil-v0",
                dataset_revision="hf:openai/gsm8k",
                n_fewshot=5,
                metric="accuracy",
                request_type="Generate",
            )
        ],
        "scores": {"gsm8k": {"accuracy": 0.7451}},
        "smoke_test": {"samples": 0, "outcome": "skipped"},
        "caas_log": [],
        "hardware": {"os": "Linux 6.8"},
        "started_at": "2026-05-08T12:00:00+00:00",
        "ended_at": "2026-05-08T12:01:00+00:00",
    }
    base.update(overrides)
    return Manifest(**base)  # type: ignore[arg-type]


class TestCanonicalJSON:
    def test_signature_field_excluded(self) -> None:
        m = _fixture()
        signed = m.sign()
        # The unsigned and signed canonical JSON must be byte-identical
        # because manifest_signature is excluded by canonicalization.
        assert signed.canonical_json() == m.canonical_json()

    def test_byte_stable_within_process(self) -> None:
        m = _fixture()
        a = m.canonical_json()
        b = m.canonical_json()
        assert a == b

    def test_keys_sorted_lexicographically(self) -> None:
        m = _fixture()
        text = m.canonical_json()
        parsed = json.loads(text)
        # Top-level keys must be sorted.
        assert list(parsed) == sorted(parsed)
        # And a nested dict (engine) must be sorted too.
        assert list(parsed["engine"]) == sorted(parsed["engine"])

    def test_nan_rejected(self) -> None:
        m = _fixture(scores={"gsm8k": {"accuracy": math.nan}})
        with pytest.raises(ManifestError, match="non-finite"):
            m.canonical_json()

    def test_inf_rejected(self) -> None:
        m = _fixture(scores={"gsm8k": {"accuracy": math.inf}})
        with pytest.raises(ManifestError, match="non-finite"):
            m.canonical_json()


class TestSignAndVerify:
    def test_sign_then_verify_passes(self) -> None:
        m = _fixture().sign()
        assert m.manifest_signature.startswith("sha256:")
        assert m.verify()

    def test_unsigned_fails_verify(self) -> None:
        assert _fixture().verify() is False

    def test_tampered_score_fails_verify(self) -> None:
        m = _fixture().sign()
        tampered = m.model_copy(update={"scores": {"gsm8k": {"accuracy": 0.99}}})
        assert tampered.verify() is False

    def test_resign_is_idempotent(self) -> None:
        m1 = _fixture().sign()
        m2 = m1.sign()  # signing an already-signed manifest produces the same sig.
        assert m1.manifest_signature == m2.manifest_signature


class TestSaveAndLoad:
    def test_save_then_load_round_trip(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(tmp_path) / "manifest.json"  # type: ignore[arg-type]
        m = _fixture()
        out_path = m.save(path)
        loaded = Manifest.load(out_path)
        assert loaded.verify()
        assert loaded.scores == m.scores

    def test_load_missing_raises(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(tmp_path) / "nope.json"  # type: ignore[arg-type]
        with pytest.raises(ManifestError, match="not found"):
            Manifest.load(path)


class TestDiff:
    def test_diff_picks_up_score_change(self) -> None:
        a = _fixture(scores={"gsm8k": {"accuracy": 0.74}}).sign()
        b = _fixture(scores={"gsm8k": {"accuracy": 0.81}}).sign()
        d = Manifest.diff(a, b)
        assert "scores.gsm8k.accuracy" in d
        assert d["scores.gsm8k.accuracy"] == (0.74, 0.81)

    def test_diff_ignores_timestamps(self) -> None:
        a = _fixture(started_at="2026-05-08T00:00:00+00:00")
        b = _fixture(started_at="2026-05-08T11:11:11+00:00")
        # When everything else is identical, the diff is empty (timestamps
        # are excluded from the diff because they always vary across runs).
        assert Manifest.diff(a, b) == {}

    def test_canonical_json_called_directly(self) -> None:
        # The module-level helper produces the same string as the method.
        m = _fixture()
        assert canonical_json(m) == m.canonical_json()
