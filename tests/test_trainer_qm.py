"""Trainer._fit_qm integration test.

Builds a synthetic SPICE-shard directory from the fake HDF5 preprocessor
pipeline, points a tiny config at it, and runs a short training loop. Verifies:
  - metrics.jsonl is populated
  - loss EMA is finite
  - a checkpoint is saved and round-trips through torch.load
  - loss at end is below loss at start (real training signal, not NaN drift)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from scripts.preprocess import preprocess_spice
from thermofrag.training.trainer import Trainer, build_qm_head, build_spice_loaders
from thermofrag.utils.config import load_config


def _make_synthetic_spice_hdf5(h5_path: Path, n_mols: int = 20, n_conf_per: int = 3):
    """Create a fake SPICE-schema HDF5 with a translation- and rotation-
    invariant harmonic energy. PaiNN is SE(3) equivariant (uses only pairwise
    radial features), so the target has to respect the same symmetries for
    regression to converge.

    Energy     E(x) = 0.001 * sum_{i<j} ||x_i - x_j||^2     (hartree, x in bohr)
    Gradient   dE/dx_i = 0.002 * (n * x_i - sum_k x_k)      (hartree/bohr)
    Formation  E - mean(E) across conformers of a molecule  (kcal/mol after loader)
    """
    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "w") as f:
        for i in range(n_mols):
            n_atoms = int(rng.integers(3, 12))
            z = rng.choice([1, 6, 7, 8], size=n_atoms, p=[0.5, 0.3, 0.1, 0.1]).astype(np.int16)
            conf = rng.normal(size=(n_conf_per, n_atoms, 3)).astype(np.float32)  # bohr
            # Pair-distance energy, translation- and rotation-invariant.
            diff = conf[:, :, None, :] - conf[:, None, :, :]  # [C, N, N, 3]
            pair2 = (diff ** 2).sum(-1)  # [C, N, N]
            mask = np.triu(np.ones((n_atoms, n_atoms)), k=1)
            tot = 0.001 * (pair2 * mask).sum(axis=(1, 2)).astype(np.float64)
            form = tot - tot.mean()
            # dE/dx_i = 0.002 * (N*x_i - sum_k x_k), per conformer.
            sums = conf.sum(axis=1, keepdims=True)  # [C, 1, 3]
            grad = (0.002 * (n_atoms * conf - sums)).astype(np.float32)
            g = f.create_group(f"mol_{i:03d}")
            g.create_dataset("atomic_numbers", data=z)
            g.create_dataset("conformations", data=conf)
            g.create_dataset("dft_total_gradient", data=grad)
            g.create_dataset("dft_total_energy", data=tot)
            g.create_dataset("formation_energy", data=form)
            g.create_dataset("smiles", data=np.bytes_(f"mol_{i:03d}"))
            g.create_dataset("subset", data=np.bytes_("PubChem"))


def _build_config(repo_root: Path, shard_dir: Path, out_root: Path) -> dict:
    cfg = load_config(repo_root / "configs" / "tiny.yaml")
    cfg["run"]["out_dir"] = str(out_root)
    cfg["run"]["device"] = "cpu"  # pytest shouldn't require a GPU
    cfg["run"]["precision"] = "fp32"
    cfg["run"]["name"] = "trainer_qm_test"
    cfg["data"]["qm_train"] = str(shard_dir)
    cfg["training"]["phase"] = "pretrain_qm"
    cfg["training"]["batch_size"] = 4
    cfg["training"]["epochs"] = 5
    cfg["training"]["log_every"] = 2
    cfg["training"]["ckpt_every"] = 20
    cfg["training"]["lr"] = 1e-3
    cfg["training"]["weight_decay"] = 0.0
    cfg["training"]["grad_clip"] = 2.0
    return cfg


def test_trainer_qm_end_to_end(tmp_path):
    from thermofrag.utils.seed import seed_everything

    seed_everything(123)  # isolate from prior tests' RNG state

    h5 = tmp_path / "fake.hdf5"
    _make_synthetic_spice_hdf5(h5, n_mols=30, n_conf_per=3)
    shard_dir = tmp_path / "shards"
    preprocess_spice(h5, shard_dir, n_target=200, shard_size=40, max_heavy_atoms=50, seed=0, val_frac=0.2)

    # Split layout present
    assert (shard_dir / "train").is_dir() and (shard_dir / "val").is_dir()

    repo = Path(__file__).resolve().parents[1]
    out = tmp_path / "runs"
    cfg = _build_config(repo, shard_dir, out)

    model = build_qm_head(cfg)
    train_loader, val_loader = build_spice_loaders(cfg)
    assert val_loader is not None, "val_loader not auto-discovered from split layout"
    trainer = Trainer(cfg, model, train_loader, val_loader=val_loader, device="cpu")

    trainer.fit(max_steps=300)

    # metrics.jsonl was populated
    lines = trainer.metrics_path.read_text().strip().splitlines()
    assert len(lines) >= 5, f"expected log rows, got {len(lines)}"
    rows = [json.loads(ln) for ln in lines]
    train_rows = [r for r in rows if "loss" in r and "eval" not in r]
    assert len(train_rows) >= 5, "expected per-step training rows"
    early = sum(r["loss"] for r in train_rows[:3]) / 3
    late = sum(r["loss"] for r in train_rows[-3:]) / 3
    # Loss should show meaningful training progress. After force/energy
    # normalization both branches are O(1), so convergence rate is slower
    # than the raw-scale runs — 15% drop is still a clean signal of
    # gradient flow without being flaky under RNG.
    assert late < 0.85 * early, f"loss did not drop across run: early={early:.4f} late={late:.4f}"
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in train_rows), "non-finite loss in training"

    # Final eval row present with RMSE metrics (in physical units)
    eval_rows = [r for r in rows if "eval" in r]
    assert eval_rows, "no eval row emitted at end of fit()"
    em = eval_rows[-1]["eval"]
    assert {"loss_phys", "energy_rmse_kcal", "force_rmse_kcal_per_A", "n"} <= set(em), (
        f"missing eval metrics: {em.keys()}"
    )
    assert em["n"] > 0, "val set empty"
    assert torch.isfinite(torch.tensor(em["energy_rmse_kcal"])).all()

    # A checkpoint exists and round-trips
    ckpts = list(trainer.ckpt_dir.glob("qm_*.pt"))
    assert ckpts, "no checkpoint files written"
    sd = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    assert "state_dict" in sd and "step" in sd
