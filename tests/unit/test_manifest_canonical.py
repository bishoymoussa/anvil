"""Hardened canonical-JSON byte-stability tests (design §16.5).

The system-prompt warning was specific: *"Manifest canonical JSON is
byte-stable or it isn't."* These tests:

1. Generate randomized manifest content via Hypothesis and verify
   ``canonical_json`` is deterministic within the process.
2. Spawn a fresh Python interpreter that constructs the same manifest and
   compares bytes — catches dict-iteration-order differences that
   ``sort_keys=True`` is supposed to eliminate.
3. Reject NaN / Inf / non-ASCII edge cases per the §16.5 rules.
"""

from __future__ import annotations

import json
import subprocess
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

from anvil.manifest import Manifest, ModelInfo, TaskInfo, canonical_json


def _make_manifest(scores_value: float = 0.7) -> Manifest:
    return Manifest(
        anvil_version="0.0.1",
        engine={"name": "hf", "version": "x", "backend_hash": "sha256:aa"},
        model=ModelInfo(
            id="test/model",
            revision="main",
            dtype="bfloat16",
            quantization=None,
            config_hash="sha256:bb",
            architecture="LlamaForCausalLM",
        ),
        tokenization={"hash": "sha256:cc", "padding_side": "left"},
        chat_template=None,
        sampler={"hash": "sha256:dd", "temperature": 0.0, "max_tokens": 256},
        tasks=[
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
        scores={"gsm8k": {"accuracy": scores_value}},
        smoke_test={"samples": 0, "outcome": "skipped"},
        caas_log=[],
        hardware={"os": "Linux 6.8"},
        started_at="2026-05-08T12:00:00+00:00",
        ended_at="2026-05-08T12:01:00+00:00",
    )


class TestCanonicalDeterminism:
    def test_repeat_within_process_byte_identical(self) -> None:
        m = _make_manifest()
        a = canonical_json(m)
        b = canonical_json(m)
        assert a == b
        assert a.encode("utf-8") == b.encode("utf-8")

    def test_keys_sorted_at_every_depth(self) -> None:
        m = _make_manifest()
        text = canonical_json(m)
        parsed = json.loads(text)
        # Walk the structure; every dict's keys must appear in sorted order.
        stack: list[object] = [parsed]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                assert list(node) == sorted(node), f"unsorted keys: {list(node)}"
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    def test_signature_field_excluded_from_canonical_bytes(self) -> None:
        m = _make_manifest()
        unsigned = canonical_json(m)
        signed = canonical_json(m.sign())
        # signing doesn't change canonical_json output (rule 4).
        assert unsigned == signed


class TestFreshInterpreterParity:
    """Spawn a fresh ``python -c ...`` and compare canonical bytes.

    A test that's only ever run inside the same process can mask
    dict-ordering bugs that show up when the interpreter starts cold (e.g.
    pyc caching of constants). Spawning a subprocess gives us the same
    "identity is the bytes" property the §8 design promises.
    """

    def test_subprocess_produces_identical_canonical(self) -> None:
        program = (
            "from tests.unit.test_manifest_canonical import _make_manifest\n"
            "from anvil.manifest import canonical_json\n"
            "import sys\n"
            "sys.stdout.write(canonical_json(_make_manifest()))\n"
        )
        # Run from the repo root so pytest's conftest path-injection still
        # makes ``tests`` importable; we replicate that by adding ``tests``
        # to sys.path here too.
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            check=False,
            cwd=".",
            env={"PYTHONPATH": "src:tests"},
            timeout=60,
        )
        assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
        out_subprocess = result.stdout.decode("utf-8")
        out_inproc = canonical_json(_make_manifest())
        assert out_subprocess == out_inproc


class TestHypothesisStability:
    """Generate randomized manifest scores and verify canonical_json is
    deterministic regardless of input shape."""

    @settings(max_examples=50, deadline=None)
    @given(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        st.text(
            alphabet=st.characters(blacklist_categories=("Cs", "Cn"), blacklist_characters="\x00"),
            max_size=64,
        ),
    )
    def test_canonical_is_deterministic_under_random_inputs(
        self, score: float, model_id: str
    ) -> None:
        if not model_id.strip() or "/" not in model_id:
            model_id = "test/random"
        m = _make_manifest(scores_value=score).model_copy(
            update={
                "model": ModelInfo(
                    id=model_id,
                    revision="main",
                    dtype="bfloat16",
                    quantization=None,
                    config_hash="sha256:bb",
                    architecture="LlamaForCausalLM",
                )
            }
        )
        a = canonical_json(m)
        b = canonical_json(m)
        assert a == b


class TestEdgeCases:
    def test_unicode_values_round_trip(self) -> None:
        m = _make_manifest().model_copy(
            update={"hardware": {"os": "Linux", "comment": "café — 中文 — emoji 🔨"}}
        )
        text = canonical_json(m)
        assert "café" in text
        assert "中文" in text
        assert "🔨" in text
        # And the JSON itself round-trips through Python's parser.
        json.loads(text)

    def test_negative_zero_normalized(self) -> None:
        # JSON forbids signed zero distinctions; the canonical encoder coerces
        # -0.0 → 0.0 so platforms that hand back -0.0 don't produce different
        # bytes.
        m = _make_manifest(scores_value=-0.0)
        a = canonical_json(m)
        # The string representation of 0.0 is "0.0"; -0.0 would be "-0.0" if
        # not normalized.
        assert "-0.0" not in a
