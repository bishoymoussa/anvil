"""Tests for ``anvil.Sampler`` (design §4.3)."""

from __future__ import annotations

import pytest

from anvil import Sampler


class TestSamplerHash:
    def test_greedy_default_is_argmax_invariant(self) -> None:
        s = Sampler.greedy()
        assert s.is_argmax_invariant()
        assert s.temperature == 0.0
        assert s.top_p == 1.0
        assert s.source == "greedy"

    def test_two_greedy_samplers_have_equal_hash(self) -> None:
        assert Sampler.greedy().hash == Sampler.greedy().hash

    def test_hash_changes_when_temperature_changes(self) -> None:
        a = Sampler.greedy()
        b = Sampler(temperature=0.7)
        assert a.hash != b.hash

    def test_hash_is_independent_of_source_field(self) -> None:
        # Two samplers with identical knobs but different `source` labels
        # produce the same outputs, so they must have the same hash.
        a = Sampler.greedy(max_tokens=128)
        b = Sampler(temperature=0.0, max_tokens=128, source="explicit")
        assert a.hash == b.hash

    def test_seed_is_part_of_hash(self) -> None:
        a = Sampler(temperature=0.7, seed=1)
        b = Sampler(temperature=0.7, seed=2)
        assert a.hash != b.hash


class TestSamplerDiff:
    def test_diff_explains_temperature_delta(self) -> None:
        d = Sampler.greedy().diff(Sampler(temperature=0.7))
        assert "temperature" in d
        assert d["temperature"] == (0.0, 0.7)

    def test_diff_excludes_source(self) -> None:
        # source isn't an output-affecting field, so it's not in the diff.
        a = Sampler.greedy()
        b = Sampler(temperature=0.0, source="explicit")
        assert "source" not in a.diff(b)

    def test_identical_samplers_diff_empty(self) -> None:
        s = Sampler.greedy()
        assert s.diff(s) == {}


class TestSamplerValidation:
    def test_negative_temperature_rejected(self) -> None:
        with pytest.raises(ValueError, match="temperature"):
            Sampler(temperature=-0.5)

    def test_top_p_above_1_rejected(self) -> None:
        with pytest.raises(ValueError, match="top_p"):
            Sampler(top_p=1.5)

    def test_zero_max_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            Sampler(max_tokens=0)

    def test_top_k_zero_rejected(self) -> None:
        # 0 is ambiguous; the user must say "-1 (off)" or "≥1".
        with pytest.raises(ValueError, match="top_k"):
            Sampler(top_k=0)


class TestSamplerArgmaxInvariance:
    def test_temperature_one_is_not_argmax_invariant(self) -> None:
        assert not Sampler(temperature=1.0).is_argmax_invariant()

    def test_top_k_one_is_argmax_invariant(self) -> None:
        # top_k=1 is mathematically equivalent to argmax under temperature=0.
        assert Sampler(temperature=0.0, top_k=1).is_argmax_invariant()

    def test_repetition_penalty_breaks_invariance(self) -> None:
        assert not Sampler(temperature=0.0, repetition_penalty=1.1).is_argmax_invariant()


def test_to_manifest_field_includes_source_and_hash() -> None:
    field = Sampler.greedy().to_manifest_field()
    assert field["source"] == "greedy"
    assert field["hash"].startswith("sha256:")
    assert field["temperature"] == 0.0
