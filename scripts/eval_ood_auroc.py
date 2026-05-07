"""Phase-5 / claim C5: OOD-target detection via Laplace variance.

Protocol (docs/PLAN.md C5, docs/METHOD.md §7):

  - ID targets ``y_id``: sampled from the empirical ``phi_z`` distribution of the
    conditional LMDB training split.
  - OOD targets ``y_ood``: Pareto-frontier extremes. For each of ``n_ood`` draws,
    pick a random subset ``S`` of properties of size k ∈ [3, n_properties]
    and set ``y_ood[j] = sign * r`` for j ∈ S, where r ~ U[2.5, 4.0]·σ and the
    signs are chosen so the combination is rare (near simultaneous extremes of
    several properties — those are the "thin corners" of the multivariate phi
    distribution). Other entries are filled from ID.

  - The Laplace predictive variance norm ``||Var_μ(y)||`` is the uncertainty
    signal. ID should have low norm, OOD high. AUROC on the binary label
    (ID=0, OOD=1) measures detection quality.

Reports:
    results/eval/phase5/c5_ood_auroc.json
    results/eval/phase5/c5_variance_hist.png

Exit criterion: AUROC > 0.8.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.utils.config import load_config


logger = logging.getLogger("c5_ood")


def load_mu_head(ckpt_path: Path, n_properties: int, hidden: int = 256) -> ChemicalPotentialHead:
    head = ChemicalPotentialHead(n_properties=n_properties, hidden=hidden)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    mu_sd = {k[len("mu."):]: v for k, v in sd.items() if k.startswith("mu.")}
    missing, unexpected = head.load_state_dict(mu_sd, strict=False)
    logger.info("mu head missing keys: %s", missing)
    logger.info("mu head unexpected:   %s", unexpected)
    head.eval()
    return head


def collect_phi_z(dataset: ZINCFragmentDataset, n: int, seed: int = 0) -> np.ndarray:
    """Return a [n, K] stack of standardized phi vectors sampled from the dataset."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    phi_mean = dataset.phi_mean
    phi_std = dataset.phi_std
    rows = []
    for i in idx:
        rec = dataset[int(i)]
        # ZINCFragmentDataset stores ``y`` already standardised when the LMDB
        # was built conditionally. phi_mean/phi_std are there for inverse
        # transforms, but records already carry y_std for training. We read
        # directly.
        y = getattr(rec, "y", None)
        if y is not None:
            rows.append(y.detach().cpu().numpy())
        else:
            # Fall back to raw phi if present, then standardise.
            phi = getattr(rec, "phi", None)
            if phi is None:
                raise RuntimeError(f"record {i} has neither y nor phi")
            phi_np = phi.detach().cpu().numpy()
            rows.append(((phi_np - phi_mean) / phi_std).astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


def build_pareto_ood(phi_z_id: np.ndarray, n_ood: int,
                     k_min: int = 3, r_lo: float = 2.5, r_hi: float = 4.0,
                     seed: int = 1) -> np.ndarray:
    """Return n_ood OOD y_z vectors by pushing k ≥ k_min axes to ±r σ.

    Sign is chosen randomly so the combination is not axis-aligned with
    typical gradient-frontier samples (mixing ±; hence "Pareto-thin" rather
    than "Pareto-optimal").
    """
    rng = np.random.default_rng(seed)
    N, K = phi_z_id.shape
    out = np.empty((n_ood, K), dtype=np.float32)
    for i in range(n_ood):
        base_idx = rng.integers(N)
        y = phi_z_id[base_idx].copy()
        k = rng.integers(k_min, K + 1)
        axes = rng.choice(K, size=k, replace=False)
        for a in axes:
            sign = rng.choice([-1.0, 1.0])
            r = rng.uniform(r_lo, r_hi)
            y[a] = sign * r
        out[i] = y
    return out


def variance_norm(head: ChemicalPotentialHead, y: np.ndarray, batch: int = 512) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for start in range(0, len(y), batch):
            yb = torch.from_numpy(y[start:start + batch]).float()
            var = head.predictive_variance(yb)  # [B, K]
            outs.append(torch.linalg.norm(var, dim=-1).cpu().numpy())
    return np.concatenate(outs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", type=Path,
                   default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("results/checkpoints/joint_final.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"),
                   help="Phase-3 config with hidden dim used at training time.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase5"))
    p.add_argument("--n-id", type=int, default=2000)
    p.add_argument("--n-ood", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    mu_hidden = int(cfg.get("joint_conditional", {}).get("mu_hidden", 256))

    dataset = ZINCFragmentDataset(args.lmdb, split="train")
    n_prop = dataset.phi_dim
    logger.info("LMDB: %d records, phi_dim=%d, properties=%s",
                len(dataset), n_prop, dataset.phi_properties)

    head = load_mu_head(args.ckpt, n_properties=n_prop, hidden=mu_hidden)
    logger.info("Laplace fitted: %s", bool(head._laplace_fitted.item()))

    # Pull ID phi vectors (already z-standardised by the LMDB builder).
    phi_z_id = collect_phi_z(dataset, n=args.n_id, seed=args.seed)
    # Build Pareto-thin OOD y_z.
    y_z_ood = build_pareto_ood(phi_z_id, n_ood=args.n_ood, seed=args.seed + 17)

    # Compute Laplace variance norm.
    var_id = variance_norm(head, phi_z_id)
    var_ood = variance_norm(head, y_z_ood)

    logger.info("Var||·|| ID:  mean=%.4f  std=%.4f  max=%.4f",
                var_id.mean(), var_id.std(), var_id.max())
    logger.info("Var||·|| OOD: mean=%.4f  std=%.4f  max=%.4f",
                var_ood.mean(), var_ood.std(), var_ood.max())

    # AUROC with OOD=1.
    y_true = np.concatenate([np.zeros(len(var_id)), np.ones(len(var_ood))])
    y_score = np.concatenate([var_id, var_ood])
    auroc = float(roc_auc_score(y_true, y_score))

    report = {
        "n_id":   int(len(var_id)),
        "n_ood":  int(len(var_ood)),
        "n_properties": int(n_prop),
        "auroc":  auroc,
        "c5_target": 0.8,
        "pass_c5": auroc > 0.8,
        "stats": {
            "var_id_mean":  float(var_id.mean()),
            "var_id_std":   float(var_id.std()),
            "var_id_max":   float(var_id.max()),
            "var_ood_mean": float(var_ood.mean()),
            "var_ood_std":  float(var_ood.std()),
            "var_ood_max":  float(var_ood.max()),
            "var_ratio_mean": float(var_ood.mean() / (var_id.mean() + 1e-12)),
        },
    }
    report_path = args.out_dir / "c5_ood_auroc.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("AUROC = %.4f  (target > 0.8 %s)",
                auroc, "✓" if report["pass_c5"] else "✗")
    logger.info("report → %s", report_path)

    # Plot the variance-norm histogram.
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
    bins = np.linspace(0,
                       max(var_id.max(), var_ood.max()) * 1.02,
                       40)
    ax.hist(var_id, bins=bins, alpha=0.55, label=f"ID (n={len(var_id)})",
            color="#1f77b4", density=True)
    ax.hist(var_ood, bins=bins, alpha=0.55, label=f"OOD (n={len(var_ood)})",
            color="#d62728", density=True)
    ax.set_xlabel(r"$\|\mathrm{Var}_\mu(y)\|_2$")
    ax.set_ylabel("density")
    ax.set_title(f"Laplace variance on ID vs Pareto-thin OOD   (AUROC={auroc:.3f})")
    ax.legend()
    fig.tight_layout()
    fig_path = args.out_dir / "c5_variance_hist.png"
    fig.savefig(fig_path, dpi=150)
    logger.info("fig → %s", fig_path)

    # Also save the ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax2.plot(fpr, tpr, lw=2.0, color="#d62728")
    ax2.plot([0, 1], [0, 1], "--", color="grey", alpha=0.6)
    ax2.set_xlabel("false positive rate")
    ax2.set_ylabel("true positive rate")
    ax2.set_title(f"OOD detection ROC  AUROC={auroc:.3f}")
    fig2.tight_layout()
    fig2.savefig(args.out_dir / "c5_roc.png", dpi=150)
    logger.info("roc → %s", args.out_dir / "c5_roc.png")


if __name__ == "__main__":
    main()
