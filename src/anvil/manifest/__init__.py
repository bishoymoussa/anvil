"""Reproducibility manifest (design §8).

The manifest is the **primary output** of an evaluation run. It is sealed
at end-of-run, signed (sha256 of canonical JSON), and self-describes the
run completely enough that ``replay`` reproduces the scores byte-identical.
"""

from __future__ import annotations

from anvil.manifest.canonical import canonical_json
from anvil.manifest.diff import DiffEntry, diff, diff_entries
from anvil.manifest.freeze import strip_caas
from anvil.manifest.replay import replay
from anvil.manifest.schema import CaaSAction, Manifest, ModelInfo, TaskInfo
from anvil.manifest.sign import compute_signature, sign
from anvil.manifest.verify import verify, verify_or_raise

__all__ = [
    "Manifest",
    "ModelInfo",
    "TaskInfo",
    "CaaSAction",
    "canonical_json",
    "sign",
    "compute_signature",
    "verify",
    "verify_or_raise",
    "diff",
    "diff_entries",
    "DiffEntry",
    "replay",
    "strip_caas",
]
