"""YAML config loader with simple `defaults: <name>` inheritance."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def _deep_update(base: dict, overlay: dict) -> dict:
    out = deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)
    base_name = cfg.pop("defaults", None)
    if base_name is not None:
        base_path = path.parent / f"{base_name}.yaml"
        base = load_config(base_path)
        cfg = _deep_update(base, cfg)
    return cfg
