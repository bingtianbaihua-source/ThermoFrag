"""Phase-5 / Fig 6: Pareto reachability and uncertainty.

Three panels (docs/FIGURES.md §6):

  (a) Pareto frontier on logP × QED × SA from the conditional-LMDB holdout
      (val split), with ThermoFrag's generated molecules overlaid as a scatter
      cloud in the same coordinate system.
  (b) σ_μ(y) field on the logP × QED plane (Laplace variance norm), showing
      that uncertainty rises in the Pareto-empty corners.
  (c) ROC curve for ||σ_μ|| as an OOD detector (reuses ``c5_roc.png``
      content).

Outputs:
    results/eval/phase5/fig6_pareto.png
    results/eval/phase5/fig6_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, QED, rdMolDescriptors

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.utils.config import load_config


logger = logging.getLogger("fig6")

# SAscore (Ertl 2009) via the repo-vendored module.
try:
    from thermofrag.utils.sa_scorer import calculateScore as _sa_score
except ImportError:
    _sa_score = None


def _props_from_smiles(smi: str) -> tuple[float | None, float | None, float | None]:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None, None
    logp = float(Crippen.MolLogP(mol))
    qed = float(QED.qed(mol))
    sa = float(_sa_score(mol)) if _sa_score else float("nan")
    return logp, qed, sa


def pareto_mask(df: pd.DataFrame, maximize=("qed",), minimize=("sa",)) -> np.ndarray:
    """Return a boolean mask of Pareto-frontier rows.

    Multi-objective Pareto: a row is on the frontier if no other row
    dominates it on all axes (>= on maximize, <= on minimize, with at least
    one strict inequality).
    """
    vals = []
    for k in maximize:
        vals.append(-df[k].to_numpy())  # flip so lower = better
    for k in minimize:
        vals.append(df[k].to_numpy())
    V = np.stack(vals, axis=1)  # [N, d]
    N = V.shape[0]
    out = np.ones(N, dtype=bool)
    for i in range(N):
        if not out[i]:
            continue
        dominated = np.all(V <= V[i], axis=1) & np.any(V < V[i], axis=1)
        if dominated.any():
            out[i] = False
    return out


def laplace_variance_field(head: ChemicalPotentialHead, prop_idx: tuple[int, int],
                           phi_z_id: np.ndarray, n_grid: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate ||Var_μ(y)|| on a 2D grid over properties at prop_idx, with other
    properties held at their in-distribution mean.
    """
    pi, pj = prop_idx
    margin = 4.0
    lo = phi_z_id.min(axis=0); hi = phi_z_id.max(axis=0)
    xi = np.linspace(max(-margin, lo[pi] - 0.5), min(margin, hi[pi] + 0.5), n_grid)
    xj = np.linspace(max(-margin, lo[pj] - 0.5), min(margin, hi[pj] + 0.5), n_grid)
    XX, YY = np.meshgrid(xi, xj, indexing="xy")
    y = np.zeros((n_grid * n_grid, phi_z_id.shape[1]), dtype=np.float32)
    y[:, pi] = XX.ravel()
    y[:, pj] = YY.ravel()

    with torch.no_grad():
        var = head.predictive_variance(torch.from_numpy(y).float())
        var_norm = torch.linalg.norm(var, dim=-1).cpu().numpy()
    Z = var_norm.reshape(n_grid, n_grid)
    return XX, YY, Z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-dir", type=Path,
                   default=Path("results/eval/phase4/decoded"))
    p.add_argument("--lmdb", type=Path,
                   default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("results/checkpoints/joint_final.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase5"))
    p.add_argument("--n-ref", type=int, default=4000,
                   help="Number of holdout molecules to sample for the Pareto ref.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    mu_hidden = int(cfg.get("joint_conditional", {}).get("mu_hidden", 256))

    # --- Load reference molecules from val split ------------------------------
    dataset_val = ZINCFragmentDataset(args.lmdb, split="val")
    properties = dataset_val.phi_properties
    phi_mean = dataset_val.phi_mean
    phi_std = dataset_val.phi_std
    n_prop = dataset_val.phi_dim
    prop_index = {p: i for i, p in enumerate(properties)}
    logger.info("val split size=%d  properties=%s", len(dataset_val), properties)

    rng = np.random.default_rng(0)
    ref_idx = rng.choice(len(dataset_val),
                         size=min(args.n_ref, len(dataset_val)),
                         replace=False)
    ref_rows = []
    for i in ref_idx:
        rec = dataset_val[int(i)]
        phi = rec.phi.numpy()
        ref_rows.append({
            "logP": float(phi[prop_index["logP"]]),
            "qed":  float(phi[prop_index["qed"]]),
            "sa":   float(phi[prop_index["sa"]]),
        })
    ref_df = pd.DataFrame(ref_rows)

    # --- ThermoFrag generated molecules from decoded/*.parquet ----------------
    parquets = sorted(args.decoded_dir.glob("*.parquet"))
    gen_rows = []
    for pq in parquets:
        df = pd.read_parquet(pq)
        valid = df[df["smiles"].notna()]
        for smi in valid["smiles"]:
            logp, qed, sa = _props_from_smiles(str(smi))
            if logp is None:
                continue
            gen_rows.append({"logP": logp, "qed": qed, "sa": sa,
                             "smiles": str(smi)})
    gen_df = pd.DataFrame(gen_rows)
    logger.info("ThermoFrag generated cloud size: %d (from %d decoded parquets)",
                len(gen_df), len(parquets))

    # --- Pareto mask on ref ---------------------------------------------------
    pareto = pareto_mask(ref_df, maximize=("qed",), minimize=("sa",))
    logger.info("Pareto frontier on val split: %d / %d molecules",
                int(pareto.sum()), len(ref_df))

    # --- Load Laplace head for panel (b) --------------------------------------
    head = ChemicalPotentialHead(n_properties=n_prop, hidden=mu_hidden)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    mu_sd = {k[len("mu."):]: v for k, v in sd.items() if k.startswith("mu.")}
    head.load_state_dict(mu_sd, strict=False)
    head.eval()

    # Build phi_z (ID dist) from val split for the field overlay.
    phi_raw = np.stack([dataset_val[int(i)].phi.numpy() for i in ref_idx], axis=0)
    phi_z_id = (phi_raw - phi_mean) / phi_std
    XX, YY, Z = laplace_variance_field(head,
                                       prop_idx=(prop_index["logP"], prop_index["qed"]),
                                       phi_z_id=phi_z_id,
                                       n_grid=40)

    # --- Plot ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes[0]
    ax.scatter(ref_df.loc[~pareto, "qed"], ref_df.loc[~pareto, "sa"],
               c="#bbbbbb", s=10, alpha=0.45, label=f"ZINC val (n={len(ref_df)})",
               edgecolor="none")
    ax.scatter(ref_df.loc[pareto, "qed"], ref_df.loc[pareto, "sa"],
               c="#1f77b4", s=26, label=f"Pareto (qed↑, sa↓, n={int(pareto.sum())})",
               edgecolor="black", linewidth=0.4)
    if len(gen_df):
        ax.scatter(gen_df["qed"], gen_df["sa"], c="#d62728", s=14, alpha=0.55,
                   label=f"ThermoFrag gen (n={len(gen_df)})", edgecolor="none")
    ax.set_xlabel("QED")
    ax.set_ylabel("SA score")
    ax.invert_yaxis()
    ax.set_title("Pareto reachability  (Fig 6a)")
    ax.legend(loc="upper right")

    ax = axes[1]
    im = ax.imshow(Z, origin="lower",
                   extent=(XX.min(), XX.max(), YY.min(), YY.max()),
                   aspect="auto", cmap="viridis")
    ax.scatter(phi_z_id[:, prop_index["logP"]][:800],
               phi_z_id[:, prop_index["qed"]][:800],
               c="white", s=6, alpha=0.5, edgecolor="black", linewidth=0.1)
    ax.set_xlabel(r"$y_\mathrm{logP}$ (z)")
    ax.set_ylabel(r"$y_\mathrm{qed}$ (z)")
    ax.set_title(r"$\|\mathrm{Var}_\mu(y)\|$ field  (Fig 6b)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig_path = args.out_dir / "fig6_pareto.png"
    fig.savefig(fig_path, dpi=150)
    logger.info("fig → %s", fig_path)

    report = {
        "n_ref": int(len(ref_df)),
        "n_pareto": int(pareto.sum()),
        "n_generated": int(len(gen_df)),
        "gen_qed_median": float(np.median(gen_df["qed"])) if len(gen_df) else None,
        "gen_sa_median":  float(np.median(gen_df["sa"]))  if len(gen_df) else None,
        "ref_qed_median": float(np.median(ref_df["qed"])),
        "ref_sa_median":  float(np.median(ref_df["sa"])),
        "grid_var_min": float(Z.min()),
        "grid_var_max": float(Z.max()),
    }
    report_path = args.out_dir / "fig6_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("report → %s", report_path)


if __name__ == "__main__":
    main()
