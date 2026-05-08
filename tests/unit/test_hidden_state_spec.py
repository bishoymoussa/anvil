"""Tests for ``HiddenStateSpec`` (design §4.5)."""

from __future__ import annotations

import pytest
import torch

from anvil import HiddenStateSpec


def test_estimate_bytes_scales_with_layers_and_dim() -> None:
    s = HiddenStateSpec(layers=(0, 12, 24), positions="last", dtype=torch.float32)
    assert s.estimate_bytes(seq_len=128, hidden_dim=4096) == 3 * 1 * 4096 * 4


def test_estimate_bytes_with_all_positions() -> None:
    s = HiddenStateSpec(layers=(-1,), positions="all", dtype=torch.float16)
    # 1 layer * 128 positions * 4096 dim * 2 bytes.
    assert s.estimate_bytes(seq_len=128, hidden_dim=4096) == 1 * 128 * 4096 * 2


def test_explicit_positions_tuple() -> None:
    s = HiddenStateSpec(layers=(0,), positions=(0, 5, 10))
    assert s.estimate_bytes(seq_len=20, hidden_dim=10) == 3 * 10 * 4


def test_hash_stability() -> None:
    a = HiddenStateSpec(layers=(0, -1), positions="last")
    b = HiddenStateSpec(layers=(0, -1), positions="last")
    assert a.hash == b.hash


def test_layers_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="layers"):
        HiddenStateSpec(layers=())


def test_to_manifest_field_round_trip() -> None:
    s = HiddenStateSpec(layers=(0, 5), positions="last", dtype=torch.bfloat16)
    field = s.to_manifest_field()
    assert field["layers"] == [0, 5]
    assert field["positions"] == "last"
    assert field["dtype"] == "torch.bfloat16"
