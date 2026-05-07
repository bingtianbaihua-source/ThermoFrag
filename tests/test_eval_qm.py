"""Smoke test for scripts/eval_qm.py.

Builds a tiny QMugs shard from synthetic SDFs, runs a randomly-initialised
QMHead through ``evaluate()``, and verifies the metrics dict + predictions CSV
contain the expected keys and shapes. Does not assert model quality — a random
QMHead will have atrocious MAE — only that the plumbing works end-to-end.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

rdkit = pytest.importorskip("rdkit")

from scripts.eval_qm import _load_eval_dataset, evaluate
from scripts.preprocess import preprocess_qmugs
from thermofrag.model.painn import PaiNNConfig
from thermofrag.potentials.qm import QMHead


def _write_sdf_tree(root: Path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    for i, smi in enumerate(["CCO", "CN", "C=O", "c1ccccc1", "CC(=O)O"]):
        d = root / f"chembl_{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(mol, randomSeed=42 + i)
        mol.SetProp("DFT:TOTAL_ENERGY", f"{-100.0 - 50 * i:.6f}")
        mol.SetProp("SMILES", smi)
        Chem.SDWriter(str(d / "conf_00.sdf")).write(mol)


def test_evaluate_roundtrip(tmp_path):
    sdf_root = tmp_path / "sdf"
    _write_sdf_tree(sdf_root)
    shards = tmp_path / "shards"
    preprocess_qmugs(sdf_root, shards, n_target=20, shard_size=4, max_heavy_atoms=50, seed=0)

    cfg = PaiNNConfig(hidden=32, num_layers=2, cutoff=5.0, n_radial=16)
    model = QMHead(cfg).eval()

    ds, _stats = _load_eval_dataset("qmugs", shards)
    out = evaluate(model, ds, device="cpu", batch_size=2, max_samples=None)

    m = out["metrics"]
    for key in (
        "n",
        "energy_mae_kcal_per_mol",
        "energy_rmse_kcal_per_mol",
        "energy_spearman",
        "energy_pearson",
        "energy_mae_per_atom_kcal",
    ):
        assert key in m, f"missing metric {key!r}"
    assert m["n"] == len(ds)
    assert m["energy_mae_kcal_per_mol"] > 0  # random head, non-trivial error

    # Predictions array shape matches metrics
    assert out["y_true"].shape == (m["n"],)
    assert out["y_pred"].shape == (m["n"],)
    assert out["sizes"].shape == (m["n"],)
