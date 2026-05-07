#!/usr/bin/env python
"""Build Fig. 5: uncertainty-aware refusal from the mu-head Laplace posterior.

This script recomputes the plotted data from the conditional ChEMBL LMDB and
the trained ThermoFrag checkpoint. It writes a three-panel figure, raw plotting
tables/arrays, and a manifest that records the data source for each panel.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, QED

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.utils.config import load_config

try:
    from thermofrag.utils.sa_scorer import calculateScore as _sa_score
except ImportError:
    _sa_score = None


OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "grey": "#999999",
    "black": "#000000",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")


def load_mu_head(ckpt_path: Path, n_properties: int, hidden: int) -> ChemicalPotentialHead:
    head = ChemicalPotentialHead(n_properties=n_properties, hidden=hidden)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    mu_state = {k[len("mu.") :]: v for k, v in state.items() if k.startswith("mu.")}
    head.load_state_dict(mu_state, strict=False)
    head.eval()
    return head


def props_from_smiles(smi: str) -> tuple[float | None, float | None, float | None]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None
    logp = float(Crippen.MolLogP(mol))
    qed = float(QED.qed(mol))
    sa = float(_sa_score(mol)) if _sa_score else float("nan")
    return logp, qed, sa


def pareto_mask(df: pd.DataFrame, maximize: tuple[str, ...], minimize: tuple[str, ...]) -> np.ndarray:
    vals = []
    for col in maximize:
        vals.append(-df[col].to_numpy())
    for col in minimize:
        vals.append(df[col].to_numpy())
    arr = np.stack(vals, axis=1)
    out = np.ones(arr.shape[0], dtype=bool)
    for i in range(arr.shape[0]):
        dominated = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
        if dominated.any():
            out[i] = False
    return out


def collect_reference(dataset: ZINCFragmentDataset, n_ref: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n_ref, len(dataset)), replace=False)
    prop_index = {p: i for i, p in enumerate(dataset.phi_properties)}
    rows = []
    phi_raw = []
    for i in idx:
        rec = dataset[int(i)]
        phi = rec.phi.detach().cpu().numpy()
        phi_raw.append(phi)
        rows.append(
            {
                "logP": float(phi[prop_index["logP"]]),
                "qed": float(phi[prop_index["qed"]]),
                "sa": float(phi[prop_index["sa"]]),
            }
        )
    return pd.DataFrame(rows), np.asarray(phi_raw, dtype=np.float32)


def collect_generated(decoded_dir: Path) -> pd.DataFrame:
    rows = []
    for pq in sorted(decoded_dir.glob("*.parquet")):
        df = pd.read_parquet(pq)
        if "smiles" not in df.columns:
            continue
        for smi in df.loc[df["smiles"].notna(), "smiles"]:
            logp, qed, sa = props_from_smiles(str(smi))
            if logp is None:
                continue
            rows.append({"logP": logp, "qed": qed, "sa": sa, "smiles": str(smi)})
    return pd.DataFrame(rows)


def variance_norm(head: ChemicalPotentialHead, y: np.ndarray, batch: int = 512) -> np.ndarray:
    vals = []
    with torch.no_grad():
        for start in range(0, len(y), batch):
            yb = torch.from_numpy(y[start : start + batch]).float()
            var = head.predictive_variance(yb)
            vals.append(torch.linalg.norm(var, dim=-1).cpu().numpy())
    return np.concatenate(vals)


def build_pareto_thin_ood(y_id: np.ndarray, n_ood: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, k_props = y_id.shape
    out = np.empty((n_ood, k_props), dtype=np.float32)
    for i in range(n_ood):
        y = y_id[rng.integers(n)].copy()
        k = rng.integers(3, k_props + 1)
        axes = rng.choice(k_props, size=k, replace=False)
        y[axes] = rng.choice([-1.0, 1.0], size=k) * rng.uniform(2.5, 4.0, size=k)
        out[i] = y
    return out


def roc_curve_numpy(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order].astype(np.int64)
    scores = y_score[order]
    distinct = np.r_[True, scores[1:] != scores[:-1]]
    tps = np.cumsum(y)[distinct]
    fps = np.cumsum(1 - y)[distinct]
    positives = max(int(y_true.sum()), 1)
    negatives = max(int((1 - y_true).sum()), 1)
    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    auroc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auroc


def variance_field(
    head: ChemicalPotentialHead,
    y_id: np.ndarray,
    prop_i: int,
    prop_j: int,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo = y_id.min(axis=0)
    hi = y_id.max(axis=0)
    xi = np.linspace(max(-4.0, lo[prop_i] - 0.5), min(4.0, hi[prop_i] + 0.5), n_grid)
    xj = np.linspace(max(-4.0, lo[prop_j] - 0.5), min(4.0, hi[prop_j] + 0.5), n_grid)
    xx, yy = np.meshgrid(xi, xj, indexing="xy")
    y = np.zeros((n_grid * n_grid, y_id.shape[1]), dtype=np.float32)
    y[:, prop_i] = xx.ravel()
    y[:, prop_j] = yy.ravel()
    z = variance_norm(head, y).reshape(n_grid, n_grid)
    return xx, yy, z


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", type=Path, default=Path("data/processed/chembl_conditional.lmdb"))
    parser.add_argument("--ckpt", type=Path, default=Path("results/checkpoints/joint_final.pt"))
    parser.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    parser.add_argument("--decoded-dir", type=Path, default=Path("results/eval/phase4/decoded"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/paper/fig5_ood_refusal.png"))
    parser.add_argument("--raw-dir", type=Path, default=Path("results/eval/phase5_fig5_ood_refusal"))
    parser.add_argument("--n-ref", type=int, default=4000)
    parser.add_argument("--n-id", type=int, default=2000)
    parser.add_argument("--n-ood", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_style()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    hidden = int(cfg.get("joint_conditional", {}).get("mu_hidden", 256))

    dataset_val = ZINCFragmentDataset(args.lmdb, split="val")
    dataset_train = ZINCFragmentDataset(args.lmdb, split="train")
    prop_index = {p: i for i, p in enumerate(dataset_val.phi_properties)}
    n_props = int(dataset_val.phi_dim)
    head = load_mu_head(args.ckpt, n_properties=n_props, hidden=hidden)

    ref_df, phi_raw = collect_reference(dataset_val, args.n_ref, args.seed)
    phi_z = ((phi_raw - dataset_val.phi_mean) / dataset_val.phi_std).astype(np.float32)
    pareto = pareto_mask(ref_df, maximize=("qed",), minimize=("sa",))
    ref_df["pareto"] = pareto
    gen_df = collect_generated(args.decoded_dir)

    # ID/OOD scores use the train split, matching the original Phase-5 protocol.
    id_df, id_phi_raw = collect_reference(dataset_train, args.n_id, args.seed)
    y_id = ((id_phi_raw - dataset_train.phi_mean) / dataset_train.phi_std).astype(np.float32)
    y_ood = build_pareto_thin_ood(y_id, args.n_ood, args.seed + 17)
    var_id = variance_norm(head, y_id)
    var_ood = variance_norm(head, y_ood)
    y_true = np.concatenate([np.zeros_like(var_id), np.ones_like(var_ood)])
    y_score = np.concatenate([var_id, var_ood])
    fpr, tpr, auroc = roc_curve_numpy(y_true, y_score)
    var_ratio = float(var_ood.mean() / (var_id.mean() + 1e-12))

    xx, yy, z = variance_field(
        head,
        phi_z,
        prop_i=prop_index["logP"],
        prop_j=prop_index["qed"],
        n_grid=42,
    )

    ref_path = args.raw_dir / "fig5_reference_points.csv"
    gen_path = args.raw_dir / "fig5_generated_points.csv"
    roc_path = args.raw_dir / "fig5_roc_points.csv"
    score_path = args.raw_dir / "fig5_variance_scores.csv"
    grid_path = args.raw_dir / "fig5_variance_field.npz"
    report_path = args.raw_dir / "fig5_report.json"
    ref_df.to_csv(ref_path, index=False)
    gen_df.to_csv(gen_path, index=False)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(roc_path, index=False)
    pd.DataFrame(
        {
            "label": np.r_[np.repeat("ID", len(var_id)), np.repeat("OOD", len(var_ood))],
            "variance_norm": y_score,
        }
    ).to_csv(score_path, index=False)
    np.savez_compressed(grid_path, xx=xx, yy=yy, variance_norm=z, phi_z=phi_z)

    fig = plt.figure(figsize=(7.2, 2.75))
    gs = fig.add_gridspec(1, 3, wspace=0.64)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(ref_df.loc[~pareto, "qed"], ref_df.loc[~pareto, "sa"], s=5, c="#c7c7c7", alpha=0.35, lw=0)
    ax.scatter(
        ref_df.loc[pareto, "qed"],
        ref_df.loc[pareto, "sa"],
        s=17,
        c=OKABE["blue"],
        edgecolor="black",
        linewidth=0.25,
        label=f"Pareto n={int(pareto.sum())}",
    )
    if len(gen_df):
        ax.scatter(gen_df["qed"], gen_df["sa"], s=8, c=OKABE["vermillion"], alpha=0.45, lw=0, label=f"TF n={len(gen_df)}")
    ax.invert_yaxis()
    ax.set_xlabel("QED")
    ax.set_ylabel("SA score")
    ax.set_title("Pareto reachability")
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(
        z,
        origin="lower",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="auto",
        cmap="viridis",
    )
    ax.scatter(phi_z[:800, prop_index["logP"]], phi_z[:800, prop_index["qed"]], s=4, c="white", alpha=0.5, lw=0)
    ax.set_xlabel("logP target (z)")
    ax.set_ylabel("QED target (z)")
    ax.set_title("Laplace variance field")
    cbar = fig.colorbar(im, ax=ax, fraction=0.052, pad=0.03)
    cbar.set_label("variance norm", fontsize=7, labelpad=3)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(fpr, tpr, color=OKABE["vermillion"], lw=1.8)
    ax.plot([0, 1], [0, 1], ls="--", color=OKABE["grey"], lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("OOD detection")
    ax.text(0.45, 0.18, f"AUROC = {auroc:.4f}\nvariance ratio = {var_ratio:.2f}x", transform=ax.transAxes)
    panel_label(ax, "c")

    fig.suptitle("Uncertainty-aware refusal of out-of-distribution requests", y=1.04, fontsize=10)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    report = {
        "n_ref": int(len(ref_df)),
        "n_pareto": int(pareto.sum()),
        "n_generated": int(len(gen_df)),
        "n_id": int(len(var_id)),
        "n_ood": int(len(var_ood)),
        "n_properties": int(n_props),
        "auroc": auroc,
        "c5_target": 0.8,
        "pass_c5": bool(auroc > 0.8),
        "var_id_mean": float(var_id.mean()),
        "var_ood_mean": float(var_ood.mean()),
        "var_ratio_mean": var_ratio,
        "grid_var_min": float(z.min()),
        "grid_var_max": float(z.max()),
    }
    report_path.write_text(json.dumps(report, indent=2))

    manifest = {
        "figure": "Fig. 5",
        "script": str(Path(__file__).as_posix()),
        "outputs": [str(args.out), str(args.out.with_suffix(".pdf"))],
        "raw_outputs": [str(ref_path), str(gen_path), str(grid_path), str(roc_path), str(score_path), str(report_path)],
        "panels": {
            "a": {
                "description": "ChEMBL holdout QED-SA Pareto frontier with ThermoFrag generated molecules overlaid",
                "data_sources": [str(args.lmdb), str(args.decoded_dir), str(ref_path), str(gen_path)],
            },
            "b": {
                "description": "Laplace posterior variance norm over a logP-QED target grid",
                "data_sources": [str(args.lmdb), str(args.ckpt), str(grid_path)],
            },
            "c": {
                "description": "ROC curve for the Laplace variance norm as an OOD request detector",
                "data_sources": [str(args.lmdb), str(args.ckpt), str(roc_path), str(score_path), str(report_path)],
            },
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {args.out}")
    print(f"[write] {args.out.with_suffix('.pdf')}")
    print(f"[write] {manifest_path}")
    print(f"[write] {report_path}")


if __name__ == "__main__":
    main()
