"""Phase-5 / claim C6 ablation: sample with the joint-trained coupling but μ=0.

Loads the Phase-3 ``joint_final.pt`` checkpoint, reuses its coupling V and
seed pool, but replaces the μ head with a zero-mapping so the sampler runs on

    H_ablate(m; y) = V^couple(m) - 0·φ(m) = V^couple(m).

This is the **no-μ** ablation called for in docs/PLAN.md C6. Compare the
generated samples' downstream docking / strain / OOD metrics against the
conditional pools in ``results/eval/phase4/samples/`` to measure μ's effect.

Mirrors ``scripts/sample.py`` so seed indices, MH step count, batch size, and
output schema are all identical (easy paired comparison).
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
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.sampling.conditional_mh import (
    ConditionalFragmentMH,
    ConditionalMHStats,
    build_frag_phi_table,
)
from thermofrag.utils.config import load_config


def _load_coupling_from_joint(ckpt_path: Path, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" not in blob or "cfg" not in blob:
        raise ValueError(f"{ckpt_path} is not a Trainer checkpoint")
    cfg = blob["cfg"]
    mc = cfg["model"]["coupling"]
    coupling = CouplingPotential(
        n_fragments=int(mc["n_fragments"]),
        n_bond_types=int(mc.get("n_bond_types", 8)),
        hidden=int(mc["hidden"]),
        num_layers=int(mc["num_layers"]),
    )
    sd = blob["state_dict"]
    coupling_sd = {k[len("coupling."):]: v for k, v in sd.items() if k.startswith("coupling.")}
    coupling.load_state_dict(coupling_sd, strict=True)
    coupling.to(device).eval()
    return coupling, cfg


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
    print(f"[nomu-sample] data={data_path} split={args.split}")

    coupling, ckpt_cfg = _load_coupling_from_joint(args.checkpoint, device)
    n_fragments = int(ckpt_cfg["model"]["coupling"]["n_fragments"])
    properties = list(ckpt_cfg["model"]["external_field"]["properties"])
    print(f"[nomu-sample] ckpt n_fragments={n_fragments}  properties={properties}")

    pool = ZINCFragmentDataset(data_path, split=args.split)
    if pool.n_fragments != n_fragments:
        raise ValueError(
            f"dataset n_fragments={pool.n_fragments} != ckpt n_fragments={n_fragments}"
        )
    phi_mean_np = pool.phi_mean
    phi_std_np = pool.phi_std
    print(f"[nomu-sample] seed pool={len(pool)}")

    print(f"[nomu-sample] building per-fragment phi table from {args.library}")
    t0 = time.time()
    frag_phi_np = build_frag_phi_table(args.library, properties)
    frag_phi_np = frag_phi_np[:n_fragments]
    print(f"[nomu-sample]   built in {time.time()-t0:.1f}s  shape={frag_phi_np.shape}")

    frag_phi = torch.from_numpy(frag_phi_np).to(device)
    phi_mean = torch.from_numpy(np.asarray(phi_mean_np, dtype=np.float32)).to(device)
    phi_std = torch.from_numpy(np.asarray(phi_std_np, dtype=np.float32)).to(device)

    # The ablation: mu_head returns zeros of the right shape.
    K = frag_phi.shape[1]

    def _mu_zero(y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(y.shape[0], K, device=y.device, dtype=y.dtype)

    kernel = ConditionalFragmentMH(
        coupling=lambda b: coupling(b),
        mu_head=_mu_zero,
        frag_phi=frag_phi,
        phi_mean=phi_mean,
        phi_std=phi_std,
        n_fragments=n_fragments,
        beta=float(args.beta),
    )

    g = torch.Generator().manual_seed(int(args.seed))
    n = min(int(args.n), len(pool))
    seed_idxs = torch.randperm(len(pool), generator=g)[:n].tolist()

    y_zero_template = torch.zeros(K, device=device)
    outputs: list[dict] = []
    stats = ConditionalMHStats(H_mean_history=[])
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        idxs = seed_idxs[start : start + args.batch_size]
        data_list = [pool[i] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)
        B = int(batch.num_graphs)
        y = y_zero_template.unsqueeze(0).expand(B, -1).contiguous()
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
                "y_raw": [float(x) for x in phi_mean_np.tolist()],
                "y_std": [0.0] * K,
                "H_history": [],
            })
        done = min(start + args.batch_size, n)
        dt = time.time() - t0
        print(f"[nomu-sample]   {done}/{n}  {done/max(dt,1e-6):.1f} chain/s  accept_rate={stats.accept_rate:.3f}")

    with open(args.out, "wb") as f:
        pickle.dump({
            "n_chains": len(outputs),
            "mh_steps": args.mh_steps,
            "beta": args.beta,
            "accept_rate": stats.accept_rate,
            "samples": outputs,
            "ablation": "no_mu",
        }, f, protocol=4)
    print(f"[nomu-sample] wrote {len(outputs)} → {args.out}")


if __name__ == "__main__":
    main()
