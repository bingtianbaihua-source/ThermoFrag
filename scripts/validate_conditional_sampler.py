"""Phase-4 sampler sanity: does conditioning actually steer the chain?

For each property axis k and sign s in {+1, -1}, set y_std = s * delta * e_k
(with other axes at 0), run the conditional MH kernel for ``--mh-steps`` sweeps
starting from random ZINC seeds, and measure the mean standardized fragment-sum
phi per axis. If the sampler honors the μ head, the produced phi on axis k
should shift in the same sign as s -- i.e. ``phi_std_gen[+] - phi_std_gen[-]``
is a diagonally dominant matrix.

This is the Phase-4 analog of Phase 2's KL sanity (docs/MILESTONES.md) -- we
want to know that ``scripts/sample.py`` is working before investing in Vina
docking.

Outputs:
    --out/<run_name>/response_matrix.npy    # [K, K] (diff target-axis x measured-axis)
    --out/<run_name>/response_matrix.png    # heatmap
    --out/<run_name>/report.json            # diag/off-diag summary

Usage::

    python scripts/validate_conditional_sampler.py \
        --checkpoint results/checkpoints/joint_final.pt \
        --library data/processed/fragment_library.parquet \
        --data data/processed/chembl_conditional.lmdb \
        --delta 1.5 --n 128 --mh-steps 40 \
        --out results/eval/phase4/sampler_validation
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.sampling.conditional_mh import (
    ConditionalFragmentMH,
    ConditionalMHStats,
    build_frag_phi_table,
    _phi_of_batch,
)

# Re-use the same loader as scripts/sample.py.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample import _load_joint_checkpoint  # noqa: E402


def _sample_phi_under_y(
    kernel: ConditionalFragmentMH,
    pool: ZINCFragmentDataset,
    seed_idxs: list[int],
    y_std: np.ndarray,  # [K]
    frag_phi: torch.Tensor,
    phi_mean: torch.Tensor,
    phi_std: torch.Tensor,
    batch_size: int,
    mh_steps: int,
    device: str,
) -> tuple[np.ndarray, float]:
    """Run MH and return per-sample standardized-phi rows (shape [n, K]) + accept rate."""
    y_tensor = torch.as_tensor(y_std, dtype=torch.float32, device=device)
    rows: list[np.ndarray] = []
    stats = ConditionalMHStats(H_mean_history=[])
    for start in range(0, len(seed_idxs), batch_size):
        idxs = seed_idxs[start : start + batch_size]
        data_list = [pool[i] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)
        B = int(batch.num_graphs)
        y = y_tensor.unsqueeze(0).expand(B, -1).contiguous()
        kernel.run(batch, y, n_steps=int(mh_steps), stats=stats)
        phi_z = _phi_of_batch(batch, frag_phi, phi_mean, phi_std)  # [B, K]
        rows.append(phi_z.cpu().numpy())
    return np.concatenate(rows, axis=0), stats.accept_rate


def _plot_heatmap(M: np.ndarray, labels: list[str], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lim = float(np.max(np.abs(M)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("measured axis of phi_gen")
    ax.set_ylabel("target axis y set to +delta")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--delta", type=float, default=1.5, help="magnitude of y_std excursion per axis")
    p.add_argument("--n", type=int, default=128, help="chains per y setting")
    p.add_argument("--mh-steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    coupling, mu, cfg = _load_joint_checkpoint(args.checkpoint, device)
    n_fragments = int(cfg["model"]["coupling"]["n_fragments"])
    properties = list(cfg["model"]["external_field"]["properties"])
    K = len(properties)

    pool = ZINCFragmentDataset(args.data, split=args.split)
    phi_mean_np = np.asarray(pool.phi_mean, dtype=np.float32)
    phi_std_np = np.asarray(pool.phi_std, dtype=np.float32)

    print("[validate] building per-fragment phi table")
    frag_phi_np = build_frag_phi_table(args.library, properties)[:n_fragments]
    frag_phi = torch.from_numpy(frag_phi_np).to(device)
    phi_mean = torch.from_numpy(phi_mean_np).to(device)
    phi_std = torch.from_numpy(phi_std_np).to(device)

    kernel = ConditionalFragmentMH(
        coupling=lambda b: coupling(b),
        mu_head=lambda y: mu(y),
        frag_phi=frag_phi,
        phi_mean=phi_mean,
        phi_std=phi_std,
        n_fragments=n_fragments,
        beta=float(args.beta),
    )

    g = torch.Generator().manual_seed(int(args.seed))
    n = min(int(args.n), len(pool))
    seed_idxs = torch.randperm(len(pool), generator=g)[:n].tolist()

    # Also measure the baseline phi_std of the seed pool (no MH, no conditioning)
    # so we can report "sampler effect on axis k" vs "baseline shift".
    base_rows = []
    for start in range(0, n, args.batch_size):
        idxs = seed_idxs[start : start + args.batch_size]
        data_list = [pool[i] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)
        with torch.no_grad():
            phi_z = _phi_of_batch(batch, frag_phi, phi_mean, phi_std).cpu().numpy()
        base_rows.append(phi_z)
    phi_baseline = np.concatenate(base_rows, axis=0)  # [n, K]

    pos_means = np.zeros((K, K), dtype=np.float32)
    neg_means = np.zeros((K, K), dtype=np.float32)
    accept_rates: dict[str, float] = {}
    t0 = time.time()
    for k, name in enumerate(properties):
        for sign, key in [(+1.0, "pos"), (-1.0, "neg")]:
            y_std = np.zeros(K, dtype=np.float32)
            y_std[k] = sign * float(args.delta)
            phi_rows, acc = _sample_phi_under_y(
                kernel, pool, seed_idxs, y_std, frag_phi, phi_mean, phi_std,
                batch_size=args.batch_size, mh_steps=args.mh_steps, device=device,
            )
            mean_row = phi_rows.mean(axis=0)
            if key == "pos":
                pos_means[k] = mean_row
            else:
                neg_means[k] = mean_row
            accept_rates[f"{name}_{key}"] = acc
            print(
                f"[validate] axis={name:>4s} sign={sign:+.0f}  "
                f"accept={acc:.3f}  phi_z[axis]={mean_row[k]:+.3f}  "
                f"t={time.time()-t0:.1f}s"
            )

    # Response matrix: row = target axis set to +delta (minus -delta), col = measured axis.
    response = pos_means - neg_means  # [K, K]
    np.save(args.out / "response_matrix.npy", response)
    _plot_heatmap(response, properties, args.out / "response_matrix.png")

    # Summary stats.
    diag = np.diag(response)
    off_diag = response - np.diag(diag)
    report = {
        "properties": properties,
        "delta": args.delta,
        "n_per_setting": n,
        "mh_steps": args.mh_steps,
        "beta": args.beta,
        "seed_baseline_phi_z_mean": phi_baseline.mean(axis=0).tolist(),
        "diag_response": diag.tolist(),
        "diag_response_mean": float(diag.mean()),
        "diag_positive_fraction": float((diag > 0).mean()),
        "max_abs_off_diag": float(np.abs(off_diag).max()),
        "off_diag_l1_mean": float(np.abs(off_diag).mean()),
        "accept_rates": accept_rates,
        "pos_means": pos_means.tolist(),
        "neg_means": neg_means.tolist(),
        "pass_sampler_sanity": bool((diag > 0).mean() >= 0.75),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(
        {k: v for k, v in report.items() if k not in ("pos_means", "neg_means")},
        indent=2,
    ))
    print(f"[validate] wrote -> {args.out}")


if __name__ == "__main__":
    main()
