"""SI / Fig S1: detailed-balance numerical check.

For the Fragment MH kernel used by Phase-2/3 PCD, the proposal is a symmetric
node-label flip so q(m|m') = q(m'|m). Under MH, the instantaneous per-sample
acceptance probability is

    A(m -> m') = min(1, exp(-β ΔV))    with ΔV = V(m') - V(m).

We verify this empirically by:
  1. Sampling many (m, m', ΔV, accept) tuples from the kernel.
  2. Binning by ΔV and plotting the empirical acceptance rate against the
     theoretical curve ``min(1, exp(-β ΔV))``.
  3. Reporting the max absolute residual and a linear-regression slope; both
     should be close to zero and one respectively for the kernel to be in
     detailed balance.

Outputs:
    results/eval/phase5/s1_detailed_balance.json
    results/eval/phase5/s1_detailed_balance.png
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
import torch
from torch_geometric.data import Batch

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.coupling import CouplingPotential


logger = logging.getLogger("db_check")


def load_coupling(ckpt_path: Path, n_fragments: int, hidden: int, num_layers: int,
                  n_bond_types: int = 8) -> CouplingPotential:
    coupling = CouplingPotential(n_fragments=n_fragments,
                                 n_bond_types=n_bond_types,
                                 hidden=hidden, num_layers=num_layers)
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # If checkpoint is the joint wrapper, strip "coupling." prefix.
    has_prefix = any(k.startswith("coupling.") for k in sd)
    if has_prefix:
        sd = {k[len("coupling."):]: v for k, v in sd.items() if k.startswith("coupling.")}
    coupling.load_state_dict(sd, strict=False)
    coupling.eval()
    return coupling


@torch.no_grad()
def collect_proposal_tuples(coupling: CouplingPotential, dataset: ZINCFragmentDataset,
                            n_samples: int, beta: float, batch_size: int = 64,
                            seed: int = 0, device: str = "cpu"):
    """Sample (ΔV, accept) pairs by running the flip kernel once per graph.

    Returns numpy arrays ``delta_v``, ``accept`` each of length ``n_samples``.
    """
    coupling = coupling.to(device)
    rng = np.random.default_rng(seed)
    n_frag = dataset.n_fragments

    delta_vs = []
    accepts = []

    collected = 0
    while collected < n_samples:
        take = min(batch_size, n_samples - collected)
        idxs = rng.choice(len(dataset), size=take, replace=False)
        data_list = [dataset[int(i)] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)

        V_old = coupling(batch).cpu().numpy()  # [B]

        # Flip one random node per graph.
        ptr = batch.ptr.cpu().numpy()
        B = take
        flip_pos = np.zeros(B, dtype=np.int64)
        for k in range(B):
            a, b = int(ptr[k]), int(ptr[k + 1])
            flip_pos[k] = a + rng.integers(0, b - a)
        new_ids = rng.integers(0, n_frag, size=B)

        proposed = batch.clone()
        for k in range(B):
            proposed.frag_id[flip_pos[k]] = int(new_ids[k])
        V_new = coupling(proposed).cpu().numpy()  # [B]

        delta_v = V_new - V_old
        # Per-sample MH acceptance draw; compare u < exp(-β ΔV).
        u = rng.uniform(size=B)
        accept = (np.log(u) < -beta * delta_v).astype(np.int32)

        delta_vs.append(delta_v)
        accepts.append(accept)
        collected += B

    return np.concatenate(delta_vs), np.concatenate(accepts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", type=Path,
                   default=Path("data/processed/zinc_unconditional.lmdb"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("results/checkpoints/coupling_final.pt"))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--n-samples", type=int, default=8000)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase5"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ZINCFragmentDataset(args.lmdb, split="train")
    logger.info("dataset: %d graphs, n_fragments=%d", len(dataset), dataset.n_fragments)
    coupling = load_coupling(args.ckpt, n_fragments=dataset.n_fragments,
                             hidden=args.hidden, num_layers=args.num_layers)
    delta_v, accept = collect_proposal_tuples(coupling, dataset,
                                              n_samples=args.n_samples,
                                              beta=args.beta,
                                              device=args.device)
    logger.info("collected %d flip tuples", len(delta_v))
    logger.info("ΔV: mean=%.3f std=%.3f min=%.3f max=%.3f",
                delta_v.mean(), delta_v.std(), delta_v.min(), delta_v.max())
    logger.info("accept rate = %.4f", accept.mean())

    # Bin by ΔV and compute empirical acceptance per bin.
    n_bins = 24
    lo, hi = np.percentile(delta_v, [1, 99])
    bins = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    emp, theory, counts = [], [], []
    for i in range(n_bins):
        mask = (delta_v >= bins[i]) & (delta_v < bins[i + 1])
        if mask.sum() < 10:
            continue
        emp.append(accept[mask].mean())
        theory.append(min(1.0, float(np.exp(-args.beta * centers[i]))))
        counts.append(int(mask.sum()))
    emp = np.asarray(emp); theory = np.asarray(theory); counts = np.asarray(counts)

    residual = emp - theory
    resid_max = float(np.max(np.abs(residual)))
    resid_rmse = float(np.sqrt(np.mean(residual ** 2)))
    # Linear regression of emp on theory (should be slope≈1, intercept≈0).
    slope, intercept = np.polyfit(theory, emp, 1)

    logger.info("residual max |emp - theory| = %.4f", resid_max)
    logger.info("residual RMSE            = %.4f", resid_rmse)
    logger.info("linear fit: emp = %.3f · theory + %.3f", slope, intercept)

    report = {
        "n_samples": int(len(delta_v)),
        "beta": float(args.beta),
        "delta_v_stats": {
            "mean": float(delta_v.mean()),
            "std":  float(delta_v.std()),
            "min":  float(delta_v.min()),
            "max":  float(delta_v.max()),
        },
        "overall_accept_rate": float(accept.mean()),
        "bin_centers":       centers[:len(emp)].tolist(),
        "empirical_accept":  emp.tolist(),
        "theoretical_accept": theory.tolist(),
        "bin_counts":        counts.tolist(),
        "residual_max_abs":  resid_max,
        "residual_rmse":     resid_rmse,
        "linear_fit_slope":  float(slope),
        "linear_fit_intercept": float(intercept),
        "pass": bool((resid_max < 0.08) and (abs(slope - 1.0) < 0.08)),
    }
    report_path = args.out_dir / "s1_detailed_balance.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("report → %s", report_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    ax = axes[0]
    ax.plot(centers[:len(emp)], emp, "o-", label="empirical", color="#d62728")
    ax.plot(centers[:len(emp)], theory, "--", label=r"$\min(1, e^{-\beta \Delta V})$",
            color="#1f77b4")
    ax.set_xlabel(r"$\Delta V = V(m') - V(m)$")
    ax.set_ylabel("acceptance rate")
    ax.set_title(f"MH acceptance vs theory  (β = {args.beta})")
    ax.legend()

    ax2 = axes[1]
    ax2.scatter(theory, emp, c=counts, cmap="viridis", s=40,
                edgecolor="black", linewidth=0.5)
    ax2.plot([0, 1], [0, 1], "--", color="grey")
    ax2.set_xlabel("theoretical acceptance")
    ax2.set_ylabel("empirical acceptance")
    ax2.set_title(f"emp = {slope:.3f}·theory + {intercept:.3f}")
    ax2.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig_path = args.out_dir / "s1_detailed_balance.png"
    fig.savefig(fig_path, dpi=150)
    logger.info("fig → %s", fig_path)


if __name__ == "__main__":
    main()
