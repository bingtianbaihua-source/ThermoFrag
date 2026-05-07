"""Fig 2 (C1 QM consistency) — drug-like stratified version.

Four-panel figure:
  (a) Predicted vs DFT energy scatter, large model, SPICE val
  (b) Per-element force-component MAE bar plot (large, recal)
  (c) Drug-like stratification table — per-atom MAE by n_atoms cutoff
  (d) Cumulative MAE contribution curve — visualizes outlier concentration

Uses the large (4.54M) recalibrated checkpoint + the small (1M) as comparison.
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
        z_chunks.append(z); pos_chunks.append(pos)
        f_chunks.append(forces); e_chunks.append(energy)
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


def _predict(model, z_all, pos_all, ptr_all, energy_mean, energy_std, device, batch_size=32, with_forces=True):
    K = len(ptr_all) - 1
    y_pred = np.empty(K, dtype=np.float64)
    f_pred_chunks = []
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


def _spearman(y_true, y_pred):
    order_t = np.argsort(y_true); order_p = np.argsort(y_pred)
    rt = np.empty_like(order_t, dtype=np.float64); rp = np.empty_like(order_p, dtype=np.float64)
    rt[order_t] = np.arange(len(y_true)); rp[order_p] = np.arange(len(y_pred))
    rt -= rt.mean(); rp -= rp.mean()
    return float((rt * rp).sum() / max(math.sqrt((rt * rt).sum() * (rp * rp).sum()), 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-large", type=Path, default=Path("results/checkpoints/qm_recalibrated_best_large.pt"))
    p.add_argument("--ckpt-small", type=Path, default=Path("results/checkpoints/qm_recalibrated.pt"))
    p.add_argument("--spice-val", type=Path, default=Path("data/processed/spice/val"))
    p.add_argument("--out", type=Path, default=Path("results/eval/phase1_fig2_v2"))
    p.add_argument("--filename", default="fig2_qm.png")
    p.add_argument("--metrics-filename", default="fig2_metrics.json")
    p.add_argument("--figure-title", default="Figure 2 — QM head consistency (C1), stratified by drug-like regime")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    mf = args.spice_val.parent / "manifest.json"
    stats = json.loads(mf.read_text())["energy_stats"]
    energy_mean = float(stats["mean"]); energy_std = float(stats["std"])

    z_all, pos_all, f_true, e_true, ptr_all = _flat_shards(args.spice_val)
    n_atoms = np.diff(ptr_all)
    K = len(e_true)

    # Large model — primary
    model_large = _load(args.ckpt_large, device)
    y_pred_large, f_pred_large = _predict(model_large, z_all, pos_all, ptr_all, energy_mean, energy_std, device, 32, True)
    abs_err_large = np.abs(y_pred_large - e_true.astype(np.float64))

    # Small model — comparison
    model_small = _load(args.ckpt_small, device)
    y_pred_small, _ = _predict(model_small, z_all, pos_all, ptr_all, energy_mean, energy_std, device, 32, False)
    abs_err_small = np.abs(y_pred_small - e_true.astype(np.float64))

    # Drug-like strata
    strata = []
    for thresh in (0, 15, 20, 25, 30, 40):
        mask = n_atoms >= thresh
        if mask.sum() == 0:
            continue
        strata.append({
            "thresh": thresh,
            "n": int(mask.sum()),
            "small_per_atom_mae": float((abs_err_small[mask] / np.maximum(n_atoms[mask], 1)).mean()),
            "small_per_mol_mae": float(abs_err_small[mask].mean()),
            "large_per_atom_mae": float((abs_err_large[mask] / np.maximum(n_atoms[mask], 1)).mean()),
            "large_per_mol_mae": float(abs_err_large[mask].mean()),
        })

    # Per-element force MAE (large)
    force_err = np.abs(f_pred_large - f_true).mean(axis=1)
    per_elem = {}
    for z_val in np.unique(z_all):
        mask = z_all == z_val
        if mask.sum() >= 10:
            per_elem[int(z_val)] = float(force_err[mask].mean())

    # Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.9], hspace=0.35, wspace=0.3)

    # (a) scatter — large model, all val
    ax = fig.add_subplot(gs[0, 0])
    ax.hexbin(e_true, y_pred_large, gridsize=60, bins='log', cmap='viridis')
    lo = float(min(e_true.min(), y_pred_large.min()))
    hi = float(max(e_true.max(), y_pred_large.max()))
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
    sp_large = _spearman(e_true.astype(np.float64), y_pred_large)
    mae_large = float(abs_err_large.mean())
    ax.set_xlabel("DFT energy (kcal/mol)"); ax.set_ylabel("Predicted energy (kcal/mol)")
    ax.set_title(f"(a) SPICE val, n={K}\nMAE={mae_large:.1f}  ρ={sp_large:.4f}  (large 4.54M recal)")

    # (b) Per-element force MAE
    ax = fig.add_subplot(gs[0, 1])
    zs = sorted(per_elem.keys())
    vals = [per_elem[z] for z in zs]
    labels = [Z_TO_SYMBOL.get(z, str(z)) for z in zs]
    colors = plt.cm.tab10(np.linspace(0, 1, len(zs)))
    ax.bar(range(len(zs)), vals, color=colors, edgecolor='black', lw=0.5)
    ax.set_xticks(range(len(zs))); ax.set_xticklabels(labels)
    ax.axhline(5.0, ls='--', color='red', lw=0.8, label='C1 target 5 kcal/mol/Å')
    ax.set_xlabel("Element"); ax.set_ylabel("Force MAE (kcal/mol/Å)")
    ax.set_title("(b) Per-element force MAE")
    ax.legend(fontsize=8)

    # (c) Drug-like stratification
    ax = fig.add_subplot(gs[1, 0])
    headers = ["n_atoms ≥", "n mols", "small\nper-atom", "large\nper-atom", "large\nper-mol"]
    rows = [headers]
    for s in strata:
        lbl = "all" if s["thresh"] == 0 else f"{s['thresh']}"
        rows.append([lbl, f"{s['n']}",
                     f"{s['small_per_atom_mae']:.2f}",
                     f"{s['large_per_atom_mae']:.2f}",
                     f"{s['large_per_mol_mae']:.1f}"])
    ax.axis('off')
    tab = ax.table(cellText=rows, cellLoc='center', loc='center',
                   colWidths=[0.17, 0.15, 0.22, 0.22, 0.18])
    for (r, c), cell in tab.get_celld().items():
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_fontsize(9); cell.set_text_props(weight='bold')
        else:
            cell.set_fontsize(9)
            if c >= 2:
                val = float(rows[r][c])
                # highlight chemical-accuracy per-atom cells
                if c in (2, 3) and val < 1.0:
                    cell.set_facecolor('#c8e6c9')
    tab.scale(1, 1.55)
    ax.set_title("(c) Per-atom MAE by molecule size\n(green: within chemical accuracy, <1 kcal/mol/atom)", fontsize=10)

    # (d) Cumulative MAE contribution (outlier concentration)
    ax = fig.add_subplot(gs[1, 1])
    for tag, arr, color in [("small 1M", abs_err_small, "tab:blue"), ("large 4.54M", abs_err_large, "tab:orange")]:
        sorted_err = np.sort(arr)[::-1]
        csum = np.cumsum(sorted_err)
        cfrac = csum / csum[-1]
        xfrac = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
        ax.plot(xfrac, cfrac * 100, label=tag, color=color, lw=1.7)
    ax.set_xscale("log")
    ax.set_xlabel("Top X% of molecules by |residual| (log scale)")
    ax.set_ylabel("Cumulative contribution to MAE (%)")
    ax.set_title("(d) MAE is outlier-concentrated")
    for yref in (25, 50, 80):
        ax.axhline(yref, ls=":", color="gray", lw=0.6)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(args.figure_title, fontsize=12)
    out_png = args.out / args.filename
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"[fig2] saved {out_png}")

    metrics_path = args.out / args.metrics_filename
    metrics_path.write_text(json.dumps({
        "n_val": K,
        "large_aggregate": {"spearman": sp_large, "mae_kcal": mae_large},
        "strata": strata,
        "per_element_force_mae_kcal_per_A": {Z_TO_SYMBOL.get(z, str(z)): v for z, v in per_elem.items()},
    }, indent=2))
    print(f"[fig2] metrics -> {metrics_path}")

    if args.manifest is not None:
        manifest = {
            "figure": args.figure_title,
            "script": str(Path(__file__).as_posix()),
            "outputs": [str(out_png), str(out_pdf), str(metrics_path)],
            "panels": {
                "a": {
                    "description": "Predicted versus DFT energy scatter on SPICE validation molecules",
                    "data_sources": [
                        str(args.spice_val),
                        str(args.ckpt_large),
                        str(args.ckpt_small),
                    ],
                },
                "b": {
                    "description": "Per-element force MAE for the large recalibrated QM head",
                    "data_sources": [
                        str(args.spice_val),
                        str(args.ckpt_large),
                    ],
                },
                "c": {
                    "description": "Drug-like molecule-size stratification of per-atom and per-molecule MAE",
                    "data_sources": [
                        str(args.spice_val),
                        str(args.ckpt_large),
                        str(args.ckpt_small),
                        str(metrics_path),
                    ],
                },
                "d": {
                    "description": "Cumulative MAE contribution curve showing outlier concentration",
                    "data_sources": [
                        str(args.spice_val),
                        str(args.ckpt_large),
                        str(args.ckpt_small),
                    ],
                },
            },
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"[fig2] manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
