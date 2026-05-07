"""Smoke tests for config loading."""
from pathlib import Path

from thermofrag.utils.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_loads():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    assert cfg["model"]["qm"]["hidden"] == 128
    assert cfg["sampling"]["beta_schedule"]["beta_T"] == 5.0


def test_tiny_inherits():
    cfg = load_config(REPO_ROOT / "configs" / "tiny.yaml")
    # tiny should override hidden but inherit other keys
    assert cfg["model"]["qm"]["hidden"] == 64
    assert cfg["model"]["qm"]["cutoff"] == 5.0  # inherited from default
    assert cfg["training"]["batch_size"] == 8
