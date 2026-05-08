"""Global configuration object (design manuscript §16.9).

Precedence (highest to lowest): CLI flag > env var (``ANVIL_*``) > local
``./anvil.yaml`` > user ``~/.config/anvil/config.yaml`` > defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

CaaSMode = Literal["off", "advisory", "research", "ci"]


class Config(BaseModel):
    """Process-wide settings.

    Anything here is a tuning knob, not a per-call argument. Per-call options
    belong on ``Sampler``, ``HiddenStateSpec``, etc.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "anvil",
        description="Where Anvil writes weight metadata, manifests, and KB caches.",
    )
    hf_token: str | None = Field(
        default=None,
        description="HuggingFace token. v0 only honors the HF_TOKEN env var; this "
        "field is reserved for v0.5 config-file support.",
    )
    caas_mode: CaaSMode = Field(default="research")
    log_level: str = Field(default="INFO")
    log_format: Literal["auto", "human", "json"] = Field(default="auto")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level YAML must be a mapping, got {type(data).__name__}")
    return data


def _from_env() -> dict[str, Any]:
    out: dict[str, Any] = {}
    if v := os.getenv("ANVIL_CACHE_DIR"):
        out["cache_dir"] = Path(v)
    if v := os.getenv("HF_TOKEN"):
        out["hf_token"] = v
    if v := os.getenv("ANVIL_CAAS_MODE"):
        out["caas_mode"] = v
    if v := os.getenv("ANVIL_LOG_LEVEL"):
        out["log_level"] = v
    if v := os.getenv("ANVIL_LOG_FORMAT"):
        out["log_format"] = v
    return out


def load_config(*, cli_overrides: dict[str, Any] | None = None) -> Config:
    """Materialize a ``Config`` from defaults + files + env + CLI overrides."""
    user_cfg = _read_yaml(Path.home() / ".config" / "anvil" / "config.yaml")
    local_cfg = _read_yaml(Path.cwd() / "anvil.yaml")
    env_cfg = _from_env()
    cli_cfg = dict(cli_overrides) if cli_overrides else {}

    merged: dict[str, Any] = {}
    for layer in (user_cfg, local_cfg, env_cfg, cli_cfg):
        merged.update(layer)
    return Config(**merged)


__all__ = ["Config", "CaaSMode", "load_config"]
