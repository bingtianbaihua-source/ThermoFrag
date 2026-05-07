"""Evaluate a QMHead checkpoint on a held-out QM dataset.

Usage:
    python scripts/eval_qm.py \
        --ckpt results/checkpoints/qm_final.pt \
        --data data/processed/qmugs \
        --out  results/eval/qmugs \
        --dataset qmugs \
        [--max-samples 10000] [--device cuda]

Produces, under ``--out``:
    metrics.json     energy MAE / RMSE / Spearman / n (top-level summary)
    predictions.csv  per-sample (idx, n_atoms, energy_true, energy_pred, |err|)
    scatter.png      true-vs-predicted scatter (if matplotlib available)

The checkpoint is what ``Trainer._save_ckpt`` writes: a dict with ``state_dict``
and ``cfg``. The cfg is used to rebuild the PaiNN/QMHead with the exact geometry
the weights were trained with. Energy standardisation (mean/std) is read from the
eval dataset's own manifest so predictions are directly comparable to ground truth
in kcal/mol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_add

from thermofrag.data.qmugs import QMugsShard
from thermofrag.data.spice import SPICEShard
from thermofrag.training.trainer import build_qm_head


def _load_checkpoint(ckpt_path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" not in blob or "state_dict" not in blob:
        raise ValueError(f"Not a Trainer checkpoint (missing keys): {ckpt_path}")
    model = build_qm_head(blob["cfg"])
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device).eval()
    return model, blob["cfg"]


def _load_eval_dataset(dataset: str, data_dir: Path):
    stats = {"mean": 0.0, "std": 1.0}
    mf_path = data_dir / "manifest.json"
    if mf_path.is_file():
        stats = json.loads(mf_path.read_text()).get("energy_stats", stats)
    if dataset == "qmugs":
        ds = QMugsShard(data_dir, energy_mean=stats["mean"], energy_std=stats["std"])
    elif dataset == "spice":
        # SPICE may use split layout; accept either a flat dir or <dir>/val|train
        shard_dir = data_dir if any(data_dir.glob("shard_*.npz")) else data_dir / "val"
        if not any(shard_dir.glob("shard_*.npz")):
            shard_dir = data_dir / "train"
        sub_mf = shard_dir.parent / "manifest.json"
        if sub_mf.is_file():
            stats = json.loads(sub_mf.read_text()).get("energy_stats", stats)
        ds = SPICEShard(shard_dir, energy_mean=stats["mean"], energy_std=stats["std"])
    else:
        raise ValueError(f"unknown --dataset {dataset}")
    return ds, stats


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation without scipy: rank both, Pearson on ranks."""

    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r

    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def evaluate(model, dataset, device: str, batch_size: int, max_samples: int | None) -> dict:
    """Run the model over ``dataset`` and collect per-sample predictions in kcal/mol."""
    mean = dataset.energy_mean
    std = dataset.energy_std

    n_total = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    indices = list(range(n_total))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, sizes = [], [], []
    with torch.enable_grad():  # QMHead may use autograd for forces; we don't need it here
        for batch in loader:
            batch = batch.to(device)
            with torch.no_grad():
                scalar, _ = model.backbone(batch)
                atom_e = model.energy_mlp(scalar).squeeze(-1) + model.atom_ref(batch.z).squeeze(-1)
                E_pred_norm = scatter_add(atom_e, batch.batch, dim=0)
            y_true.append(batch.energy.float().cpu().numpy())
            # de-normalize predictions back to kcal/mol
            E_pred = E_pred_norm.float().cpu().numpy() * std + mean
            y_pred.append(E_pred)
            # per-graph atom counts for the CSV
            counts = torch.bincount(batch.batch, minlength=batch.num_graphs).cpu().numpy()
            sizes.append(counts)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    sizes = np.concatenate(sizes)
    err = y_pred - y_true
    metrics = {
        "n": int(y_true.shape[0]),
        "energy_mae_kcal_per_mol": float(np.mean(np.abs(err))),
        "energy_rmse_kcal_per_mol": float(math.sqrt(np.mean(err * err))),
        "energy_spearman": _spearman(y_true, y_pred),
        "energy_pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0,
        "energy_mae_per_atom_kcal": float(np.mean(np.abs(err) / np.maximum(sizes, 1))),
    }
    return {"metrics": metrics, "y_true": y_true, "y_pred": y_pred, "sizes": sizes}


def _scatter_plot(y_true, y_pred, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.4)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("DFT energy (kcal/mol)")
    ax.set_ylabel("Predicted energy (kcal/mol)")
    ax.set_title(f"n={len(y_true)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="processed shard dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dataset", choices=["qmugs", "spice"], required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    model, _cfg = _load_checkpoint(args.ckpt, device)
    dataset, stats = _load_eval_dataset(args.dataset, args.data)
    print(f"[eval] ckpt={args.ckpt}  dataset={args.dataset}  n={len(dataset)}  device={device}")

    result = evaluate(model, dataset, device, args.batch_size, args.max_samples)
    m = result["metrics"]

    (args.out / "metrics.json").write_text(json.dumps(m, indent=2))
    # CSV
    import csv
    with (args.out / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "n_atoms", "energy_true", "energy_pred", "abs_err"])
        for i, (yt, yp, n) in enumerate(zip(result["y_true"], result["y_pred"], result["sizes"])):
            w.writerow([i, int(n), f"{yt:.6f}", f"{yp:.6f}", f"{abs(yp - yt):.6f}"])
    _scatter_plot(result["y_true"], result["y_pred"], args.out / "scatter.png")

    print(json.dumps(m, indent=2))
    print(f"[eval] metrics -> {args.out / 'metrics.json'}")
    print(f"[eval] predictions -> {args.out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
