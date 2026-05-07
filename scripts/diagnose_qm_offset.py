"""Diagnose whether the QM head's absolute-energy MAE is a per-element reference offset.

Fits a per-element linear correction on the SPICE-val prediction residuals. If after
correction the MAE drops near C1's threshold, the model's ranking is fine and only the
atom-reference needs a recalibration step (e.g., on QMugs train once it's downloaded).

Writes results/eval/phase1/offset_diagnosis.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    shard = np.load(root / "data/processed/spice/val/shard_0000.npz", allow_pickle=True)
    z_all = shard["z"].astype(np.int32)
    ptr = shard["ptr"].astype(np.int64)
    energy_true = shard["energy"].astype(np.float64)
    n_mol = len(energy_true)

    # Read predictions.
    import csv
    pred_rows = list(csv.DictReader(open(root / "results/eval/phase1/predictions.csv")))
    assert len(pred_rows) == n_mol, (len(pred_rows), n_mol)
    energy_pred = np.array([float(r["energy_pred"]) for r in pred_rows], dtype=np.float64)

    # Per-element counts per molecule.
    maxz = int(z_all.max()) + 1
    counts = np.zeros((n_mol, maxz), dtype=np.int32)
    for i in range(n_mol):
        a, b = int(ptr[i]), int(ptr[i + 1])
        for zi in z_all[a:b]:
            counts[i, int(zi)] += 1

    unique_z = np.where(counts.sum(0) > 0)[0]
    err = energy_pred - energy_true
    X = counts[:, unique_z].astype(np.float64)
    # OLS: err ~ sum_e beta_e * count_e (no intercept, element reference only)
    beta, *_ = np.linalg.lstsq(X, err, rcond=None)

    corrected = err - X @ beta
    mae_raw = float(np.mean(np.abs(err)))
    rmse_raw = float(np.sqrt(np.mean(err**2)))
    mae_corr = float(np.mean(np.abs(corrected)))
    rmse_corr = float(np.sqrt(np.mean(corrected**2)))

    n_atoms = counts.sum(axis=1)
    mae_per_atom_raw = float(np.mean(np.abs(err) / np.maximum(n_atoms, 1)))
    mae_per_atom_corr = float(np.mean(np.abs(corrected) / np.maximum(n_atoms, 1)))

    # Spearman via rank-Pearson
    def rank(x):
        order = np.argsort(x, kind="stable")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r

    def spearman(a, b):
        ra, rb = rank(a), rank(b)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
        return float((ra * rb).sum() / denom) if denom > 0 else 0.0

    sp_raw = spearman(energy_true, energy_pred)
    sp_corr = spearman(energy_true, energy_pred - X @ beta)

    result = {
        "n_mol": int(n_mol),
        "unique_z": unique_z.tolist(),
        "beta_kcal_per_atom_by_z": {int(z): float(b) for z, b in zip(unique_z, beta)},
        "raw": {
            "mae_kcal_per_mol": mae_raw,
            "rmse_kcal_per_mol": rmse_raw,
            "mae_per_atom_kcal": mae_per_atom_raw,
            "spearman": sp_raw,
        },
        "after_element_offset": {
            "mae_kcal_per_mol": mae_corr,
            "rmse_kcal_per_mol": rmse_corr,
            "mae_per_atom_kcal": mae_per_atom_corr,
            "spearman": sp_corr,
        },
        "notes": "SPICE val residuals fit with per-element OLS offsets (no intercept).",
    }

    out_path = root / "results/eval/phase1/offset_diagnosis.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[diag] wrote {out_path}")


if __name__ == "__main__":
    main()
