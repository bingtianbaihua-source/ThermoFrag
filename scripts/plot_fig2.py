"""Fig 2 (C1 QM consistency).

Three-panel figure:
  (a) Predicted vs DFT energy scatter on SPICE val (recalibrated head).
  (b) Per-element force-component MAE bar plot.
  (c) Pre- vs post-recalibration MAE / per-atom MAE / Spearman table panel.

QMugs is not downloaded in this project; SPICE val stands in for the held-out
benchmark. This is documented in the Phase 1 memory and the paper Methods.

Usage:
    python scripts/plot_fig2.py \
        --ckpt-recal results/checkpoints/qm_recalibrated.pt \
        --ckpt-raw   results/checkpoints/qm_final.pt \
        --spice-val  data/processed/spice/val \
        --out        results/eval/phase1_recal
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


Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl", 35: "Br"}


def _load(ckpt: Path, device: str):
    blob = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_qm_head(blob["cfg"])
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device).eval()
    return model


def _flat_shards(val_dir: Path):
    shards = sorted(val_dir.glob("shard_*.npz"))
    z_chunks, pos_chunks, f_chunks, e_chunks, ptr_chunks = [], [], [], [], []
    offset = 0
    for shard in shards:
        with np.load(shard, mmap_mode="r") as o:
            z = np.asarray(o["z"]).astype(np.int64)
            pos = np.asarray(o["pos"]).astype(np.float32)
            forces = np.asarray(o["forces"]).astype(np.float32)
            energy = np.asarray(o["energy"]).astype(np.float32)
            ptr = np.asarray(o["ptr"]).astype(np.int64)
        z_chunks.append(z)
        pos_chunks.append(pos)
        f_chunks.append(forces)
        e_chunks.append(energy)
        if not ptr_chunks:
            ptr_chunks.append(ptr)
        else:
            ptr_chunks.append(ptr[1:] + offset)
        offset += int(ptr[-1])
    return (
        np.concatenate(z_chunks),
        np.concatenate(pos_chunks),
        np.concatenate(f_chunks),
        np.concatenate(e_chunks),
        np.concatenate(ptr_chunks),
    )


def _predict(model, z_all, pos_all, ptr_all, energy_mean, energy_std, device, batch_size=48,
             with_forces=True):
    K = len(ptr_all) - 1
    y_pred = np.empty(K, dtype=np.float64)
    # Per-atom force prediction and ground-truth force would be compared.
    f_pred_chunks = []
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
        if with_forces:
            batch.pos.requires_grad_(True)
        scalar, _ = model.backbone(batch)
        atom_e = model.energy_mlp(scalar).squeeze(-1) + model.atom_ref(batch.z).squeeze(-1)
        e_norm = scatter_add(atom_e, batch.batch, dim=0)
        y_pred[start:stop] = e_norm.detach().float().cpu().numpy() * energy_std + energy_mean
        if with_forces:
            grad = torch.autograd.grad(e_norm.sum(), batch.pos, create_graph=False)[0]
            f_pred_chunks.append((-grad * energy_std).detach().float().cpu().numpy())
    f_pred = np.concatenate(f_pred_chunks, axis=0) if with_forces else None
    return y_pred, f_pred


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, n_atoms: np.ndarray) -> dict:
    err = y_pred - y_true
    # Spearman via rank-Pearson
    order_t = np.argsort(y_true)
    order_p = np.argsort(y_pred)
    rt = np.empty_like(order_t, dtype=np.float64)
    rp = np.empty_like(order_p, dtype=np.float64)
    rt[order_t] = np.arange(len(y_true))
    rp[order_p] = np.arange(len(y_pred))
    rt -= rt.mean()
    rp -= rp.mean()
    sp = float((rt * rp).sum() / max(math.sqrt((rt * rt).sum() * (rp * rp).sum()), 1e-12))
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "spearman": sp,
        "mae_per_atom": float(np.mean(np.abs(err) / np.maximum(n_atoms, 1))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-recal", type=Path, required=True)
    p.add_argument("--ckpt-raw", type=Path, required=True)
    p.add_argument("--spice-val", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    # Load stats from SPICE train manifest
    mf = (args.spice_val.parent / "manifest.json")
    stats = json.loads(mf.read_text())["energy_stats"]
    energy_mean = float(stats["mean"])
    energy_std = float(stats["std"])

    z_all, pos_all, f_true_all, e_true_all, ptr_all = _flat_shards(args.spice_val)
    if args.max_samples is not None:
        K = min(args.max_samples, len(e_true_all))
        a1 = int(ptr_all[K])
        z_all = z_all[:a1]; pos_all = pos_all[:a1]; f_true_all = f_true_all[:a1]
        e_true_all = e_true_all[:K]; ptr_all = ptr_all[: K + 1]
    K = len(e_true_all)
    n_atoms = np.diff(ptr_all)
    print(f"[fig2] SPICE val K={K} atoms={len(z_all)}", flush=True)

    model_recal = _load(args.ckpt_recal, device)
    y_pred_recal, f_pred_recal = _predict(
        model_recal, z_all, pos_all, ptr_all, energy_mean, energy_std, device, args.batch_size
    )
    m_recal = _metrics(e_true_all.astype(np.float64), y_pred_recal, n_atoms)

    model_raw = _load(args.ckpt_raw, device)
    y_pred_raw, _ = _predict(
        model_raw, z_all, pos_all, ptr_all, energy_mean, energy_std, device, args.batch_size,
        with_forces=False
    )
    m_raw = _metrics(e_true_all.astype(np.float64), y_pred_raw, n_atoms)

    # Per-element force MAE (recalibrated model)
    force_err = np.abs(f_pred_recal - f_true_all)  # [N_atoms, 3]
    force_mae = force_err.mean(axis=1)             # [N_atoms]
    per_elem = {}
    for z_val in np.unique(z_all):
        mask = z_all == z_val
        if mask.sum() >= 10:
            per_elem[int(z_val)] = float(force_mae[mask].mean())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 0.9], wspace=0.35)

    # Panel (a) scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.hexbin(e_true_all, y_pred_recal, gridsize=60, bins='log', cmap='viridis')
    lo = float(min(e_true_all.min(), y_pred_recal.min()))
    hi = float(max(e_true_all.max(), y_pred_recal.max()))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    ax.set_xlabel("DFT energy (kcal/mol)")
    ax.set_ylabel("Predicted energy (kcal/mol)")
    ax.set_title(f"(a) SPICE val, n={K}\nMAE={m_recal['mae']:.1f}  ρ={m_recal['spearman']:.3f}")

    # Panel (b) per-element force MAE
    ax = fig.add_subplot(gs[0, 1])
    zs = sorted(per_elem.keys())
    vals = [per_elem[z] for z in zs]
    labels = [Z_TO_SYMBOL.get(z, str(z)) for z in zs]
    colors = plt.cm.tab10(np.linspace(0, 1, len(zs)))
    ax.bar(range(len(zs)), vals, color=colors, edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(zs)))
    ax.set_xticklabels(labels)
    ax.axhline(5.0, ls='--', color='red', lw=0.8, label='C1 target 5 kcal/mol/Å')
    ax.set_xlabel("Element")
    ax.set_ylabel("Force MAE (kcal/mol/Å)")
    ax.set_title("(b) Per-element force MAE")
    ax.legend(fontsize=8)

    # Panel (c) table
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    rows = [
        ["Metric", "raw 1M", "recalibrated"],
        ["Spearman", f"{m_raw['spearman']:.4f}", f"{m_recal['spearman']:.4f}"],
        ["MAE/mol (kcal)", f"{m_raw['mae']:.1f}", f"{m_recal['mae']:.1f}"],
        ["MAE/atom (kcal)", f"{m_raw['mae_per_atom']:.2f}", f"{m_recal['mae_per_atom']:.2f}"],
        ["RMSE (kcal)", f"{m_raw['rmse']:.1f}", f"{m_recal['rmse']:.1f}"],
    ]
    table = ax.table(cellText=rows, loc='center', cellLoc='center', colWidths=[0.38, 0.3, 0.32])
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_fontsize(10); cell.set_text_props(weight='bold')
        else:
            cell.set_fontsize(9)
    table.scale(1, 1.6)
    ax.set_title("(c) C1 metrics")

    fig.suptitle("Figure 2 — QM head consistency (C1)", fontsize=12)
    out_path = args.out / "fig2_qm.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig2] saved {out_path}")

    (args.out / "fig2_metrics.json").write_text(json.dumps({
        "n_val": K,
        "raw": m_raw,
        "recal": m_recal,
        "per_element_force_mae_kcal_per_A": {Z_TO_SYMBOL.get(z, str(z)): v for z, v in per_elem.items()},
    }, indent=2))
    print(f"[fig2] metrics -> {args.out / 'fig2_metrics.json'}")


if __name__ == "__main__":
    main()
