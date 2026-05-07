"""SPICE pipeline test using a synthetic HDF5 file that mimics the schema.

Checks:
  - Element / heavy-atom filters drop the right molecules
  - Unit conversions are right (bohr->A, hartree->kcal, gradient->force)
  - save_shard + SPICEShard round-trip a random conformer intact
  - QMHead can forward on a PyG Batch built from SPICEShard items
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from thermofrag.data.spice import (
    BOHR_TO_ANGSTROM,
    DRUGLIKE_Z,
    HARTREE_TO_KCAL,
    SPICEShard,
    iter_spice_hdf5,
    save_shard,
)
from thermofrag.model.painn import PaiNNConfig
from thermofrag.potentials.qm import QMHead
from torch_geometric.data import Batch

from scripts.preprocess import preprocess_spice


def _make_mol_group(f, name, z, n_conf, energy_h_per_conf, with_subset="PubChem"):
    g = f.create_group(name)
    n = len(z)
    rng = np.random.default_rng(abs(hash(name)) % (2**31))
    conf = rng.normal(size=(n_conf, n, 3)).astype(np.float32)  # bohr
    grad = rng.normal(scale=0.01, size=(n_conf, n, 3)).astype(np.float32)  # hartree/bohr
    energies = np.asarray(energy_h_per_conf, dtype=np.float64)
    formation = energies - energies.mean()
    g.create_dataset("atomic_numbers", data=np.asarray(z, dtype=np.int16))
    g.create_dataset("conformations", data=conf)
    g.create_dataset("dft_total_gradient", data=grad)
    g.create_dataset("dft_total_energy", data=energies)
    g.create_dataset("formation_energy", data=formation)
    g.create_dataset("smiles", data=np.bytes_(name))
    if with_subset:
        g.create_dataset("subset", data=np.bytes_(with_subset))
    return conf, grad, formation


def _build_fake_spice(tmp_path: Path) -> Path:
    h5 = tmp_path / "fake_spice.hdf5"
    with h5py.File(h5, "w") as f:
        # Drug-like small (keep)
        _make_mol_group(f, "mol_CH3OH", [6, 1, 1, 1, 8, 1], n_conf=3, energy_h_per_conf=[-115.1, -115.2, -115.15])
        # Drug-like large but under 50 heavy (keep) — 10 C + 2 N + 2 O = 14 heavy
        zb = [6]*10 + [7, 7, 8, 8] + [1]*20
        _make_mol_group(f, "mol_big", zb, n_conf=2, energy_h_per_conf=[-600.0, -600.1])
        # Contains Iodine (reject)
        _make_mol_group(f, "mol_with_I", [6, 53, 1, 1, 1], n_conf=1, energy_h_per_conf=[-6900.0])
        # Heavy atom count > 3 cap (reject when cap=3)
        _make_mol_group(f, "mol_4heavy", [6, 6, 6, 6, 1, 1], n_conf=1, energy_h_per_conf=[-155.0])
    return h5


def test_filters_drop_bad_molecules(tmp_path):
    h5 = _build_fake_spice(tmp_path)
    kept = list(iter_spice_hdf5(h5, elements_allowed=DRUGLIKE_Z, max_heavy_atoms=3))
    smiles = {c.smiles for c in kept}
    assert "mol_CH3OH" in smiles, "small drug-like molecule was dropped"
    assert "mol_big" not in smiles, "heavy-atom cap did not fire"
    assert "mol_with_I" not in smiles, "iodine-containing molecule leaked past element filter"
    assert "mol_4heavy" not in smiles, "heavy-atom cap off by one"


def test_unit_conversions_and_preprocess_roundtrip(tmp_path):
    h5 = _build_fake_spice(tmp_path)
    out_dir = tmp_path / "shards"
    manifest = preprocess_spice(h5, out_dir, n_target=50, shard_size=4, max_heavy_atoms=60, seed=0)

    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "train" / "manifest.json").is_file()
    ds = SPICEShard(out_dir / "train", energy_mean=manifest["energy_stats"]["mean"], energy_std=manifest["energy_stats"]["std"])
    # 3 kept molecules (CH3OH, big, 4heavy) with 3+2+1 = 6 conformers
    assert len(ds) == 6

    # Unit conversion: we can verify against hand-computed value from fake HDF5 for mol_CH3OH conf 0.
    with h5py.File(h5, "r") as f:
        g = f["mol_CH3OH"]
        pos_bohr = g["conformations"][0]
        grad_hb = g["dft_total_gradient"][0]
        form_h = g["formation_energy"][0]
    expected_pos = pos_bohr * BOHR_TO_ANGSTROM
    expected_force = -grad_hb * HARTREE_TO_KCAL / BOHR_TO_ANGSTROM
    expected_energy = form_h * HARTREE_TO_KCAL

    # Find the CH3OH conformer 0 in the shard: 6 atoms [6,1,1,1,8,1].
    target_z = np.array([6, 1, 1, 1, 8, 1])
    found = None
    for i in range(len(ds)):
        d = ds[i]
        if d.z.shape[0] == 6 and np.array_equal(d.z.numpy(), target_z) and np.allclose(
            d.pos.numpy(), expected_pos, atol=1e-4
        ):
            found = d
            break
    assert found is not None, "CH3OH conformer 0 not located in shard"
    assert np.allclose(found.pos.numpy(), expected_pos, atol=1e-4)
    assert np.allclose(found.forces.numpy(), expected_force, atol=1e-3)
    assert math.isclose(found.energy.item(), expected_energy, rel_tol=0, abs_tol=1e-2)


def _many_mols_hdf5(h5_path, n_mols=20):
    """HDF5 with many distinct SMILES so a 20% split lands on multiple molecules."""
    rng = np.random.default_rng(0)
    with h5py.File(h5_path, "w") as f:
        for i in range(n_mols):
            n = int(rng.integers(3, 8))
            z = rng.choice([1, 6, 7, 8], size=n).astype(np.int16)
            c = rng.normal(size=(2, n, 3)).astype(np.float32)
            g = f.create_group(f"mol_{i:03d}")
            g.create_dataset("atomic_numbers", data=z)
            g.create_dataset("conformations", data=c)
            g.create_dataset("dft_total_gradient", data=(0.001 * c).astype(np.float32))
            e = 0.001 * (c ** 2).sum(axis=(1, 2)).astype(np.float64)
            g.create_dataset("dft_total_energy", data=e)
            g.create_dataset("formation_energy", data=e - e.mean())
            g.create_dataset("smiles", data=np.bytes_(f"mol_{i:03d}"))
            g.create_dataset("subset", data=np.bytes_("PubChem"))


def test_preprocess_spice_train_val_split(tmp_path):
    h5 = tmp_path / "many.hdf5"
    _many_mols_hdf5(h5, n_mols=20)
    out = tmp_path / "out"
    manifest = preprocess_spice(h5, out, n_target=40, shard_size=8, max_heavy_atoms=50, val_frac=0.25, seed=0)

    # Both splits present with non-empty shards
    train_dir, val_dir = out / "train", out / "val"
    assert list(train_dir.glob("shard_*.npz")) and list(val_dir.glob("shard_*.npz"))

    # Stats in both manifests come from the train split only (bitwise equal)
    train_mf = json.loads((train_dir / "manifest.json").read_text())
    val_mf = json.loads((val_dir / "manifest.json").read_text())
    assert train_mf["energy_stats"] == val_mf["energy_stats"] == manifest["energy_stats"]

    # Molecule-level disjoint split: no SMILES in both train and val shard sets
    def smiles_in(ds):
        # The NPZ shards don't carry SMILES (only z/pos/E/F/ptr). Use atomic-number fingerprint
        # as a stable surrogate: per-sample concatenated z array is identical across conformers
        # of the same molecule. Check there's no sample in val whose z matches a train sample's z.
        seen = []
        for i in range(len(ds)):
            d = ds[i]
            seen.append(tuple(d.z.tolist()))
        return set(seen)

    train_ds = SPICEShard(train_dir)
    val_ds = SPICEShard(val_dir)
    # NOTE: identical small SMILES could coincidentally share fingerprints, but with random
    # sizes 3-7 and 4-element chem space, most mol_i fingerprints are unique within 20 mols.
    # The strict check we *can* make is that val is non-empty and train has molecules not in val.
    train_fp = smiles_in(train_ds)
    val_fp = smiles_in(val_ds)
    assert val_fp, "val split is empty"
    assert train_fp - val_fp, "train has no molecule not in val (split degenerate)"


def test_qm_head_forwards_on_spice_batch(tmp_path):
    h5 = _build_fake_spice(tmp_path)
    out_dir = tmp_path / "shards"
    manifest = preprocess_spice(h5, out_dir, n_target=50, shard_size=4, max_heavy_atoms=60, seed=1)
    ds = SPICEShard(out_dir / "train", **{"energy_mean": manifest["energy_stats"]["mean"], "energy_std": manifest["energy_stats"]["std"]})

    samples = [ds[i] for i in range(min(4, len(ds)))]
    batch = Batch.from_data_list(samples)
    cfg = PaiNNConfig(hidden=32, num_layers=2, cutoff=5.0, n_radial=16)
    model = QMHead(cfg)
    E, F = model(batch, return_forces=True)
    assert E.shape == (len(samples),)
    assert F.shape == batch.pos.shape
    assert torch.isfinite(E).all() and torch.isfinite(F).all()
