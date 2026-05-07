"""Phase-5 / claim C6 ablation: sample with μ trained but V=0 (no coupling).

Loads the Phase-3 ``joint_final.pt`` checkpoint, reuses its μ head, but
substitutes the coupling potential V with a zero-mapping so the sampler runs
on

    H_ablate(m; y) = 0 - μ(y)·φ(m) = -μ(y)·φ(m).

This is the **no-coupling** ablation called for in docs/PLAN.md C6. Compare
against the conditional pools in ``results/eval/phase4/samples/`` to test
whether the coupling potential is necessary for C2 / C3.

Mirrors ``scripts/sample_nomu_ablation.py`` for paired comparisons.
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.sampling.conditional_mh import (
    ConditionalFragmentMH,
    ConditionalMHStats,
    build_frag_phi_table,
)
from thermofrag.utils.config import load_config


def _load_mu_from_joint(ckpt_path: Path, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" not in blob or "cfg" not in blob:
        raise ValueError(f"{ckpt_path} is not a Trainer checkpoint")
    cfg = blob["cfg"]
    me = cfg["model"]["external_field"]
    mu = ChemicalPotentialHead(n_properties=len(me["properties"]), hidden=int(me["hidden"]))
    sd = blob["state_dict"]
    mu_sd = {k[len("mu."):]: v for k, v in sd.items() if k.startswith("mu.")}
    mu.load_state_dict(mu_sd, strict=True)
    mu.to(device).eval()
    return mu, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results/checkpoints/joint_final.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    p.add_argument("--data", type=Path, default=None,
                   help="seed pool LMDB; defaults to cfg.data.conditional")
    p.add_argument("--library", type=Path,
                   default=Path("data/processed/fragment_library.parquet"))
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--mh-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0,
                   help="Match the conditional sampler for paired comparison.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config) if args.config.exists() else None
    data_path = args.data or Path(cfg["data"]["conditional"])
    print(f"[nocoup-sample] data={data_path} split={args.split}")

    mu, ckpt_cfg = _load_mu_from_joint(args.checkpoint, device)
    n_fragments = int(ckpt_cfg["model"]["coupling"]["n_fragments"])
    properties = list(ckpt_cfg["model"]["external_field"]["properties"])
    print(f"[nocoup-sample] ckpt n_fragments={n_fragments}  properties={properties}")

    pool = ZINCFragmentDataset(data_path, split=args.split)
    if pool.n_fragments != n_fragments:
        raise ValueError(
            f"dataset n_fragments={pool.n_fragments} != ckpt n_fragments={n_fragments}"
        )
    phi_mean_np = pool.phi_mean
    phi_std_np = pool.phi_std
    print(f"[nocoup-sample] seed pool={len(pool)}")

    print(f"[nocoup-sample] building per-fragment phi table from {args.library}")
    t0 = time.time()
    frag_phi_np = build_frag_phi_table(args.library, properties)
    frag_phi_np = frag_phi_np[:n_fragments]
    print(f"[nocoup-sample]   built in {time.time()-t0:.1f}s  shape={frag_phi_np.shape}")

    frag_phi = torch.from_numpy(frag_phi_np).to(device)
    phi_mean = torch.from_numpy(np.asarray(phi_mean_np, dtype=np.float32)).to(device)
    phi_std = torch.from_numpy(np.asarray(phi_std_np, dtype=np.float32)).to(device)

    # The ablation: coupling returns zeros (a tensor of shape [B]).
    K = frag_phi.shape[1]

    def _coupling_zero(batch) -> torch.Tensor:
        return torch.zeros(int(batch.num_graphs), device=device, dtype=torch.float32)

    kernel = ConditionalFragmentMH(
        coupling=_coupling_zero,
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

    # Use the data record's own y (per-seed phi, standardized) as conditioning
    # so μ drives graph evolution toward a real target (not a uniform y=0 which
    # would degenerate to uniform proposals). This matches the conditional-
    # sampler convention.
    outputs: list[dict] = []
    stats = ConditionalMHStats(H_mean_history=[])
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        idxs = seed_idxs[start : start + args.batch_size]
        data_list = [pool[i] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)
        B = int(batch.num_graphs)
        phi_raw = torch.stack([d.phi for d in data_list], dim=0).to(device)  # [B, K]
        y = ((phi_raw - phi_mean) / phi_std).float()
        _, _ = kernel.run(batch, y, n_steps=int(args.mh_steps), stats=stats)

        batch_cpu = batch.cpu()
        for j, local in enumerate(batch_cpu.to_data_list()):
            seed_smiles = getattr(local, "smiles", None)
            outputs.append({
                "frag_id": local.frag_id.tolist(),
                "edge_index": local.edge_index.t().tolist() if local.edge_index.numel() else [],
                "bond_type": local.bond_type.tolist(),
                "init_idx": idxs[j],
                "seed_smiles": seed_smiles,
                "y_raw": phi_raw[j].cpu().tolist(),
                "y_std": y[j].detach().cpu().tolist(),
                "H_history": [],
            })
        done = min(start + args.batch_size, n)
        dt = time.time() - t0
        print(f"[nocoup-sample]   {done}/{n}  {done/max(dt,1e-6):.1f} chain/s  accept_rate={stats.accept_rate:.3f}")

    with open(args.out, "wb") as f:
        pickle.dump({
            "n_chains": len(outputs),
            "mh_steps": args.mh_steps,
            "beta": args.beta,
            "accept_rate": stats.accept_rate,
            "samples": outputs,
            "ablation": "no_coupling",
        }, f, protocol=4)
    print(f"[nocoup-sample] wrote {len(outputs)} → {args.out}")


if __name__ == "__main__":
    main()
