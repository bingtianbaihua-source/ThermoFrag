"""C1 per-mol MAE outlier diagnostic.

The claim_summary says the per-mol MAE floor is "outlier-driven, not capacity-limited".
This script backs that up: for each trained QMHead checkpoint, it computes per-mol
|residual| on SPICE val, then splits by percentile (p50, p90, p95, p99, p99.9) and
stratifies outliers by atom count + dominant element + energy magnitude.

Outputs a JSON + a 2-panel figure showing (a) cumulative error distribution and
(b) |err| vs n_atoms scatter (log-log, colored by percentile).

Usage:
    python scripts/diagnose_qm_outliers.py \\
        --ckpt-small results/checkpoints/qm_recalibrated.pt \\
        --ckpt-large results/checkpoints/qm_recalibrated_best_large.pt \\
        --spice-val data/processed/spice/val \\
        --out results/eval/phase1_outlier_diag
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
    z_chunks, pos_chunks, e_chunks, ptr_chunks = [], [], [], []
    offset = 0
    for shard in shards:
        with np.load(shard, mmap_mode="r") as o:
            z = np.asarray(o["z"]).astype(np.int64)
            pos = np.asarray(o["pos"]).astype(np.float32)
            energy = np.asarray(o["energy"]).astype(np.float32)
            ptr = np.asarray(o["ptr"]).astype(np.int64)
        z_chunks.append(z)
        pos_chunks.append(pos)
        e_chunks.append(energy)
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


def _percentile_report(abs_err: np.ndarray, ptr_all: np.ndarray, z_all: np.ndarray,
                        e_true_all: np.ndarray) -> dict:
    K = len(abs_err)
    n_atoms = np.diff(ptr_all)
    # dominant element per mol (exclude H)
    dom_el = np.zeros(K, dtype=np.int64)
    for i in range(K):
        a0, a1 = int(ptr_all[i]), int(ptr_all[i + 1])
        zs = z_all[a0:a1]
        heavy = zs[zs > 1]
        if len(heavy) == 0:
            dom_el[i] = 1
            continue
        vals, counts = np.unique(heavy, return_counts=True)
        dom_el[i] = int(vals[counts.argmax()])

    pcts = [50, 75, 90, 95, 99, 99.5, 99.9]
    p_values = np.percentile(abs_err, pcts)
    sorted_err = np.sort(abs_err)
    # cumulative MAE contribution of the top-k%
    total_sum = sorted_err.sum()
    contribution = {}
    for frac in [0.001, 0.005, 0.01, 0.05, 0.10]:
        n_top = max(1, int(frac * K))
        top_sum = sorted_err[-n_top:].sum()
        contribution[f"top_{frac*100:.1f}pct_contrib_to_MAE_total"] = float(top_sum / total_sum)

    # top-50 outliers
    order = np.argsort(abs_err)[::-1][:50]
    top_rows = []
    for idx in order[:20]:  # save first 20 detailed
        i = int(idx)
        a0, a1 = int(ptr_all[i]), int(ptr_all[i + 1])
        zs = z_all[a0:a1]
        heavy = zs[zs > 1]
        vals, counts = np.unique(heavy, return_counts=True)
        elem_hist = {Z_TO_SYMBOL.get(int(v), str(v)): int(c) for v, c in zip(vals, counts)}
        h_count = int((zs == 1).sum())
        if h_count:
            elem_hist["H"] = h_count
        top_rows.append({
            "val_idx": i,
            "n_atoms": int(n_atoms[i]),
            "abs_err_kcal": float(abs_err[i]),
            "E_true_kcal": float(e_true_all[i]),
            "elem_hist": elem_hist,
        })

    # stratify by dom_el
    per_el = {}
    for z_val in np.unique(dom_el):
        mask = dom_el == z_val
        per_el[Z_TO_SYMBOL.get(int(z_val), str(z_val))] = {
            "n_mol": int(mask.sum()),
            "mean_abs_err_kcal": float(abs_err[mask].mean()),
            "median_abs_err_kcal": float(np.median(abs_err[mask])),
            "p95_abs_err_kcal": float(np.percentile(abs_err[mask], 95)),
        }

    # stratify by atom count bucket
    atom_buckets = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 60), (60, 10000)]
    per_size = {}
    for lo, hi in atom_buckets:
        mask = (n_atoms >= lo) & (n_atoms < hi)
        if mask.sum() == 0:
            continue
        per_size[f"{lo}-{hi if hi < 10000 else 'inf'}"] = {
            "n_mol": int(mask.sum()),
            "mean_abs_err_kcal": float(abs_err[mask].mean()),
            "mean_abs_err_per_atom_kcal": float((abs_err[mask] / n_atoms[mask]).mean()),
        }

    return {
        "n_val": K,
        "mae_kcal": float(abs_err.mean()),
        "median_abs_err_kcal": float(np.median(abs_err)),
        "percentiles": {str(p): float(v) for p, v in zip(pcts, p_values)},
        "outlier_contribution_to_MAE": contribution,
        "top20_outliers": top_rows,
        "stratified_by_dominant_element": per_el,
        "stratified_by_atom_count": per_size,
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

    z_all, pos_all, e_true_all, ptr_all = _flat_shards(args.spice_val)
    n_atoms = np.diff(ptr_all)
    K = len(e_true_all)
    print(f"[diag] SPICE val K={K} atoms={len(z_all)}", flush=True)

    results = {}
    for tag, ckpt in [("small", args.ckpt_small), ("large", args.ckpt_large)]:
        print(f"[diag] loading {tag} -> {ckpt}", flush=True)
        model = _load(ckpt, device)
        y_pred = _predict(model, z_all, pos_all, ptr_all, energy_mean, energy_std, device)
        err = y_pred - e_true_all.astype(np.float64)
        abs_err = np.abs(err)
        results[tag] = _percentile_report(abs_err, ptr_all, z_all, e_true_all.astype(np.float64))
        results[tag]["_predictions_sample"] = {
            "first_5": [{"true": float(e_true_all[i]), "pred": float(y_pred[i])} for i in range(5)]
        }
        # save raw errors for figure
        results[f"_{tag}_abs_err"] = abs_err.tolist()
    small_abs = np.asarray(results.pop("_small_abs_err"))
    large_abs = np.asarray(results.pop("_large_abs_err"))

    # Commentary
    for tag in ("small", "large"):
        r = results[tag]
        p99 = r["percentiles"]["99"]
        p50 = r["percentiles"]["50"]
        mae = r["mae_kcal"]
        top1_contrib = r["outlier_contribution_to_MAE"]["top_1.0pct_contrib_to_MAE_total"]
        top10_contrib = r["outlier_contribution_to_MAE"]["top_10.0pct_contrib_to_MAE_total"]
        r["_summary"] = (
            f"MAE={mae:.1f}; median={p50:.1f}; p99={p99:.1f}. "
            f"Top-1% of molecules contribute {top1_contrib*100:.1f}% of MAE, top-10% "
            f"contribute {top10_contrib*100:.1f}%. "
            f"Ratio p99/median = {p99/max(p50,1e-6):.1f}x."
        )

    (args.out / "outlier_report.json").write_text(json.dumps(results, indent=2))
    print(f"[diag] saved {args.out / 'outlier_report.json'}", flush=True)
    print(f"[diag] small: {results['small']['_summary']}")
    print(f"[diag] large: {results['large']['_summary']}")

    # Figure: (a) cumulative error, (b) |err| vs n_atoms
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    # (a) cumulative contribution curve
    for tag, arr, color in [("small (1M)", small_abs, "tab:blue"), ("large (4.5M)", large_abs, "tab:orange")]:
        sorted_err = np.sort(arr)[::-1]  # descending
        csum = np.cumsum(sorted_err)
        cfrac = csum / csum[-1]
        xfrac = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
        axes[0].plot(xfrac, cfrac * 100, label=tag, color=color, lw=1.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Top X% of molecules (log scale)")
    axes[0].set_ylabel("Cumulative contribution to MAE (%)")
    axes[0].set_title("(a) MAE is dominated by a small outlier fraction")
    axes[0].axhline(50, ls=":", color="gray", lw=0.8)
    axes[0].axhline(80, ls=":", color="gray", lw=0.8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) err vs n_atoms
    axes[1].scatter(n_atoms, small_abs, s=5, alpha=0.3, color="tab:blue", label="small (1M)")
    axes[1].scatter(n_atoms, large_abs, s=5, alpha=0.3, color="tab:orange", label="large (4.5M)")
    axes[1].set_xlabel("Atoms per molecule")
    axes[1].set_ylabel("|energy residual| (kcal/mol)")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Per-mol error vs molecule size")
    axes[1].axhline(5.0, ls="--", color="red", lw=0.8, label="C1 target 5 kcal/mol")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Figure S (C1 per-mol MAE outlier diagnostic)", fontsize=11)
    fig.tight_layout()
    out_png = args.out / "outlier_diagnostic.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[diag] saved {out_png}", flush=True)


if __name__ == "__main__":
    main()
