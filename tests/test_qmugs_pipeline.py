"""QMugs pipeline test using synthetic SDF files that mimic the layout.

We generate a small tree of RDKit-written SDF files with embedded
``DFT:TOTAL_ENERGY`` properties (as QMugs does), run the preprocessor, load
through QMugsShard, and confirm the loader, units, and QMHead forward pass
all compose.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem

from thermofrag.data.qmugs import HARTREE_TO_KCAL, QMugsShard, iter_qmugs_sdf_dir
from thermofrag.model.painn import PaiNNConfig
from thermofrag.potentials.qm import QMHead
from torch_geometric.data import Batch

from scripts.preprocess import preprocess_qmugs


DRUGLIKE_SMILES = [
    "CCO",              # ethanol
    "CC(=O)O",          # acetic acid
    "c1ccccc1O",        # phenol
    "CN1CCC(N)CC1",     # drug-like amine
    "Clc1ccc(Cl)cc1",   # dichlorobenzene
    "c1ccc2[nH]ccc2c1", # indole
]


def _write_synthetic_sdf_tree(root: Path, energies_hartree: list[float]) -> list[Path]:
    """Produce one SDF per SMILES under root/chembl_XXXX/conf_00.sdf."""
    paths = []
    for i, (smi, e_h) in enumerate(zip(DRUGLIKE_SMILES, energies_hartree)):
        sub = root / f"chembl_{i:04d}"
        sub.mkdir(parents=True, exist_ok=True)
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=42 + i) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol, maxIters=30)
        mol.SetProp("DFT:TOTAL_ENERGY", f"{e_h:.8f}")
        mol.SetProp("SMILES", smi)
        p = sub / "conf_00.sdf"
        w = Chem.SDWriter(str(p))
        w.write(mol)
        w.close()
        paths.append(p)
    return paths


def test_iter_qmugs_parses_energy_and_coords(tmp_path):
    energies = [-115.1, -228.4, -307.2, -290.8, -1150.1, -401.3]
    _write_synthetic_sdf_tree(tmp_path, energies)

    rows = list(iter_qmugs_sdf_dir(tmp_path))
    assert len(rows) == len(energies), f"expected {len(energies)} SDFs parsed, got {len(rows)}"
    # Energies arrive in kcal/mol
    observed = sorted(r.energy for r in rows)
    expected = sorted(e * HARTREE_TO_KCAL for e in energies)
    assert np.allclose(observed, expected, atol=1e-3)
    # Positions are 3D
    for r in rows:
        assert r.pos.shape[1] == 3
        assert r.z.shape[0] == r.pos.shape[0]


def test_qmugs_preprocess_roundtrip_and_qm_head(tmp_path):
    energies = [-115.0, -230.0, -305.0, -290.0, -1150.0, -400.0]
    _write_synthetic_sdf_tree(tmp_path / "sdf", energies)
    out = tmp_path / "shards"
    mf = preprocess_qmugs(tmp_path / "sdf", out, n_target=20, shard_size=3, max_heavy_atoms=50, seed=0)

    assert (out / "manifest.json").is_file()
    ds = QMugsShard(out, energy_mean=mf["energy_stats"]["mean"], energy_std=mf["energy_stats"]["std"])
    assert len(ds) == len(energies), f"expected {len(energies)} unique SMILES, got {len(ds)}"

    # Sample 0 should have expected energy (in kcal/mol) up to RDKit rounding.
    e_seen = sorted(float(ds[i].energy) for i in range(len(ds)))
    e_expected = sorted(e * HARTREE_TO_KCAL for e in energies)
    assert np.allclose(e_seen, e_expected, atol=1e-2)

    # QMHead can consume a QMugs batch (no forces). Forward must run without
    # touching batch.forces (that path is guarded by ``hasattr`` in Trainer).
    samples = [ds[i] for i in range(min(4, len(ds)))]
    batch = Batch.from_data_list(samples)
    cfg = PaiNNConfig(hidden=32, num_layers=2, cutoff=5.0, n_radial=16)
    model = QMHead(cfg)
    E = model(batch, return_forces=False)
    assert E.shape == (len(samples),)
    assert torch.isfinite(E).all()
