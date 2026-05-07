"""Compute C1 metrics restricted to the drug-like regime.

The PLAN.md C1 claim targets drug-like molecules, but the SPICE val split
includes tiny species (C2H6, CS2H4, ...) that carry extreme per-mol energy
residuals and dominate the aggregate MAE. This script splits the val set
by n_atoms and reports both the aggregate metric and the drug-like-filtered
metric (n_atoms >= 20 and >= 30).

Outputs a clean JSON for paper Methods / SI.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_add

from thermofrag.training.trainer import build_qm_head


def _load(ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_qm_head(blob["cfg"])
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device).eval()
    return model


def _flat_shards(val_dir: Path):
    shards = sorted(val_dir.glob("shard_*.npz"))
    z_chunks, pos_chunks, e_chunks, ptr_chunks = [], [], [], []
    offset = 0
    for shard in shards:
        with np.load(shard, mmap_mode="r") as o:
            z = np.asarray(o["z"]).astype(np.int64)
            pos = np.asarray(o["pos"]).astype(np.float32)
            energy = np.asarray(o["energy"]).astype(np.float32)
            ptr = np.asarray(o["ptr"]).astype(np.int64)
        z_chunks.append(z); pos_chunks.append(pos); e_chunks.append(energy)
        if not ptr_chunks:
            ptr_chunks.append(ptr)
        else:
            ptr_chunks.append(ptr[1:] + offset)
        offset += int(ptr[-1])
    return (
        np.concatenate(z_chunks),
        np.concatenate(pos_chunks),
        np.concatenate(e_chunks),
        np.concatenate(ptr_chunks),
    )


def _predict(model, z_all, pos_all, ptr_all, energy_mean, energy_std, device, batch_size=32):
    K = len(ptr_all) - 1
    y_pred = np.empty(K, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, K, batch_size):
            stop = min(start + batch_size, K)
            data_list = []
            for i in range(start, stop):
                a0, a1 = int(ptr_all[i]), int(ptr_all[i + 1])
                data_list.append(Data(
                    z=torch.from_numpy(z_all[a0:a1]),
                    pos=torch.from_numpy(pos_all[a0:a1]),
                    num_nodes=a1 - a0,
                ))
            batch = Batch.from_data_list(data_list).to(device)
            scalar, _ = model.backbone(batch)
            atom_e = model.energy_mlp(scalar).squeeze(-1) + model.atom_ref(batch.z).squeeze(-1)
            e_norm = scatter_add(atom_e, batch.batch, dim=0)
            y_pred[start:stop] = e_norm.float().cpu().numpy() * energy_std + energy_mean
    return y_pred


def _metrics_subset(y_true: np.ndarray, y_pred: np.ndarray, n_atoms: np.ndarray) -> dict:
    err = y_pred - y_true
    # Spearman
    order_t = np.argsort(y_true); order_p = np.argsort(y_pred)
    rt = np.empty_like(order_t, dtype=np.float64)
    rp = np.empty_like(order_p, dtype=np.float64)
    rt[order_t] = np.arange(len(y_true))
    rp[order_p] = np.arange(len(y_pred))
    rt -= rt.mean(); rp -= rp.mean()
    denom = math.sqrt((rt * rt).sum() * (rp * rp).sum())
    sp = float((rt * rp).sum() / max(denom, 1e-12))
    return {
        "n": int(len(y_true)),
        "mae_kcal_per_mol": float(np.mean(np.abs(err))),
        "rmse_kcal_per_mol": float(math.sqrt(np.mean(err * err))),
        "median_abs_err_kcal": float(np.median(np.abs(err))),
        "spearman": sp,
        "mae_per_atom_kcal": float(np.mean(np.abs(err) / np.maximum(n_atoms, 1))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-small", type=Path, required=True)
    p.add_argument("--ckpt-large", type=Path, required=True)
    p.add_argument("--spice-val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    mf = args.spice_val.parent / "manifest.json"
    stats = json.loads(mf.read_text())["energy_stats"]
    energy_mean = float(stats["mean"])
    energy_std = float(stats["std"])

    z_all, pos_all, e_true, ptr_all = _flat_shards(args.spice_val)
    n_atoms = np.diff(ptr_all)
    K = len(e_true)

    out = {
        "dataset": str(args.spice_val),
        "n_total": int(K),
        "subsets": {},
    }
    for tag, ckpt in [("small", args.ckpt_small), ("large", args.ckpt_large)]:
        model = _load(ckpt, device)
        y_pred = _predict(model, z_all, pos_all, ptr_all, energy_mean, energy_std, device)
        out[tag] = {}
        # Full
        out[tag]["aggregate"] = _metrics_subset(e_true.astype(np.float64), y_pred, n_atoms)
        # Filtered subsets — drug-like definitions
        for thresh in (15, 20, 25, 30, 40):
            mask = n_atoms >= thresh
            if mask.sum() == 0:
                continue
            out[tag][f"n_atoms_ge_{thresh}"] = _metrics_subset(
                e_true.astype(np.float64)[mask], y_pred[mask], n_atoms[mask]
            )
        # Heavy-only (exclude H from counting)
        heavy_counts = np.array([
            int((z_all[int(ptr_all[i]):int(ptr_all[i+1])] > 1).sum()) for i in range(K)
        ])
        for thresh in (10, 15, 20):
            mask = heavy_counts >= thresh
            if mask.sum() == 0:
                continue
            out[tag][f"heavy_ge_{thresh}"] = _metrics_subset(
                e_true.astype(np.float64)[mask], y_pred[mask], n_atoms[mask]
            )

    # Build subset n counts
    for thresh in (15, 20, 25, 30, 40):
        out["subsets"][f"n_atoms_ge_{thresh}"] = int((n_atoms >= thresh).sum())

    (args.out / "c1_druglike.json").write_text(json.dumps(out, indent=2))
    print(f"[c1] saved {args.out / 'c1_druglike.json'}")
    print()
    print("=== Per-mol MAE (kcal/mol) by atom-count threshold ===")
    print(f"{'subset':<20} {'small':>14} {'large':>14}")
    for key in ["aggregate", "n_atoms_ge_15", "n_atoms_ge_20", "n_atoms_ge_25", "n_atoms_ge_30", "n_atoms_ge_40"]:
        if key in out["small"] and key in out["large"]:
            s = out["small"][key]
            l = out["large"][key]
            print(f"{key:<20} n={s['n']:>5}  {s['mae_kcal_per_mol']:>6.2f}"
                  f"  n={l['n']:>5}  {l['mae_kcal_per_mol']:>6.2f}"
                  f"  (per-atom small={s['mae_per_atom_kcal']:.2f}, large={l['mae_per_atom_kcal']:.2f})")


if __name__ == "__main__":
    main()
