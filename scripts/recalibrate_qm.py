"""Post-hoc per-element recalibration of a trained QMHead checkpoint.

Phase 1 post-hoc trick: a trained PaiNN QMHead has per-element offset errors
(see ``results/eval/phase1/offset_diagnosis.json``). We fit an OLS correction
E_pred - E_true ~= sum_e n_e * beta_e on the SPICE TRAIN split (never val),
then fold the beta_e into ``QMHead.atom_ref`` so downstream predictions are
automatically corrected.

This addresses the part of the C1 MAE gap attributable to imperfect atomic
reference energies. The learned bonding-energy residual is left untouched.

Usage:
    python scripts/recalibrate_qm.py \
        --ckpt-in  results/checkpoints/qm_final.pt \
        --ckpt-out results/checkpoints/qm_recalibrated.pt \
        --train data/processed/spice/train \
        [--max-samples 20000] [--device cuda]

The script writes a new checkpoint with the same (state_dict, cfg) layout as
Trainer._save_ckpt, plus a nested "recalibration" block recording the fitted
betas and the before/after train-split MAE.
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


def _load_checkpoint(ckpt_path: Path, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" not in blob or "cfg" not in blob:
        raise ValueError(f"Not a Trainer checkpoint: {ckpt_path}")
    model = build_qm_head(blob["cfg"])
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device).eval()
    return model, blob["cfg"], blob


def _load_train_stats(train_dir: Path) -> dict:
    mf = train_dir.parent / "manifest.json"
    if mf.is_file():
        meta = json.loads(mf.read_text())
        return meta.get("energy_stats", {"mean": 0.0, "std": 1.0, "force_std": 1.0})
    return {"mean": 0.0, "std": 1.0, "force_std": 1.0}


def _iter_shards_flat(train_dir: Path, max_samples: int | None):
    """Yield (z[N], pos[N,3], energy[K], ptr[K+1]) in big flat chunks.

    Each yielded tuple corresponds to one shard; we stop once we've covered
    ``max_samples`` total molecules. Reading is mmap-friendly via np.load.
    """
    shards = sorted(train_dir.glob("shard_*.npz"))
    taken = 0
    for shard in shards:
        with np.load(shard, mmap_mode="r") as o:
            ptr = np.asarray(o["ptr"]).astype(np.int64)
            z = np.asarray(o["z"]).astype(np.int64)
            pos = np.asarray(o["pos"]).astype(np.float32)
            energy = np.asarray(o["energy"]).astype(np.float32)
        K = len(energy)
        if max_samples is not None and taken + K > max_samples:
            K_use = max_samples - taken
            a1 = int(ptr[K_use])
            yield z[:a1], pos[:a1], energy[:K_use], ptr[: K_use + 1]
            taken += K_use
            return
        yield z, pos, energy, ptr
        taken += K
        if max_samples is not None and taken >= max_samples:
            return


def _predict_flat(model, z_all, pos_all, energy_all, ptr_all, energy_mean, energy_std,
                  device: str, batch_size: int):
    """Run the model over flat arrays; return (y_true, y_pred, counts [K, maxz+1])."""
    K = len(energy_all)
    maxz = int(z_all.max()) + 1
    counts = np.zeros((K, maxz), dtype=np.int64)
    y_pred = np.empty(K, dtype=np.float64)

    # Vectorized count fill before loop
    mol_idx = np.repeat(np.arange(K), np.diff(ptr_all))
    np.add.at(counts, (mol_idx, z_all), 1)

    model.eval()
    with torch.no_grad():
        for start in range(0, K, batch_size):
            stop = min(start + batch_size, K)
            data_list = []
            for i in range(start, stop):
                a0, a1 = int(ptr_all[i]), int(ptr_all[i + 1])
                data_list.append(
                    Data(
                        z=torch.from_numpy(z_all[a0:a1]),
                        pos=torch.from_numpy(pos_all[a0:a1]),
                        num_nodes=a1 - a0,
                    )
                )
            batch = Batch.from_data_list(data_list).to(device)
            scalar, _ = model.backbone(batch)
            atom_e = model.energy_mlp(scalar).squeeze(-1) + model.atom_ref(batch.z).squeeze(-1)
            e_norm = scatter_add(atom_e, batch.batch, dim=0)
            y_pred[start:stop] = e_norm.float().cpu().numpy() * energy_std + energy_mean

    y_true = energy_all.astype(np.float64)
    return y_true, y_pred, counts


def _fit_per_element_ols(y_true: np.ndarray, y_pred: np.ndarray, counts: np.ndarray):
    present = np.where(counts.sum(axis=0) > 0)[0]
    X = counts[:, present].astype(np.float64)
    err = (y_pred - y_true).astype(np.float64)
    beta, *_ = np.linalg.lstsq(X, err, rcond=None)
    return present, beta


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_atoms: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "n": int(y_true.shape[0]),
        "energy_mae_kcal_per_mol": float(np.mean(np.abs(err))),
        "energy_rmse_kcal_per_mol": float(math.sqrt(np.mean(err * err))),
        "energy_mae_per_atom_kcal": float(np.mean(np.abs(err) / np.maximum(n_atoms, 1))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-in", type=Path, required=True)
    p.add_argument("--ckpt-out", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True, help="SPICE train shard dir")
    p.add_argument("--max-samples", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[recal] device={device}  ckpt={args.ckpt_in}", flush=True)

    model, cfg, blob = _load_checkpoint(args.ckpt_in, device)
    stats = _load_train_stats(args.train)
    energy_mean = float(stats["mean"])
    energy_std = float(stats["std"])
    print(f"[recal] energy_mean={energy_mean:.2f} std={energy_std:.2f}", flush=True)

    # Concatenate shards up to max_samples
    chunks = list(_iter_shards_flat(args.train, args.max_samples))
    z_all = np.concatenate([c[0] for c in chunks])
    pos_all = np.concatenate([c[1] for c in chunks])
    energy_all = np.concatenate([c[2] for c in chunks])
    # Merge ptrs with global offset
    ptrs = []
    offset = 0
    for c in chunks:
        if not ptrs:
            ptrs.append(c[3])
        else:
            ptrs.append(c[3][1:] + offset)
        offset = int(ptrs[-1][-1])
    ptr_all = np.concatenate(ptrs)
    K = len(energy_all)
    print(f"[recal] using K={K} molecules  atoms={len(z_all)}", flush=True)

    y_true, y_pred, counts = _predict_flat(
        model, z_all, pos_all, energy_all, ptr_all,
        energy_mean=energy_mean, energy_std=energy_std,
        device=device, batch_size=args.batch_size,
    )
    n_atoms = counts.sum(axis=1)
    pre = _metrics(y_true, y_pred, n_atoms)
    print(f"[recal] pre  MAE={pre['energy_mae_kcal_per_mol']:.2f} "
          f"per-atom={pre['energy_mae_per_atom_kcal']:.2f}", flush=True)

    present_z, betas = _fit_per_element_ols(y_true, y_pred, counts)
    beta_map = {int(z): float(b) for z, b in zip(present_z, betas)}
    print("[recal] OLS betas (kcal/mol/atom):", flush=True)
    for z, b in beta_map.items():
        print(f"        z={z:2d}  beta={b:+.3f}", flush=True)

    # Fold betas into atom_ref (in normalized units)
    with torch.no_grad():
        ref = model.atom_ref.weight.detach().clone()
        for z, b in beta_map.items():
            if z < ref.shape[0]:
                ref[z, 0] -= b / energy_std
        model.atom_ref.weight.data.copy_(ref.to(model.atom_ref.weight.device))

    # Recompute post metrics
    _, y_pred_post, _ = _predict_flat(
        model, z_all, pos_all, energy_all, ptr_all,
        energy_mean=energy_mean, energy_std=energy_std,
        device=device, batch_size=args.batch_size,
    )
    post = _metrics(y_true, y_pred_post, n_atoms)
    print(f"[recal] post MAE={post['energy_mae_kcal_per_mol']:.2f} "
          f"per-atom={post['energy_mae_per_atom_kcal']:.2f}", flush=True)

    out_blob = {
        "step": blob.get("step"),
        "state_dict": model.state_dict(),
        "cfg": cfg,
        "recalibration": {
            "source_ckpt": str(args.ckpt_in),
            "train_dir": str(args.train),
            "max_samples": int(args.max_samples),
            "energy_std_kcal_per_mol": energy_std,
            "betas_kcal_per_atom_by_z": beta_map,
            "train_metrics_pre": pre,
            "train_metrics_post": post,
        },
    }
    args.ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_blob, args.ckpt_out)
    print(f"[recal] saved -> {args.ckpt_out}", flush=True)


if __name__ == "__main__":
    main()
