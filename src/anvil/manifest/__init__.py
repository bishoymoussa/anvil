"""Reproducibility manifest (design §8).

For M0 we ship the schema and canonical JSON; signing, verify, diff, and
replay land in M2 (design §16.10).
"""

from __future__ import annotations

from anvil.manifest.canonical import canonical_json
from anvil.manifest.schema import CaaSAction, Manifest, ModelInfo, TaskInfo

__all__ = ["Manifest", "ModelInfo", "TaskInfo", "CaaSAction", "canonical_json"]
