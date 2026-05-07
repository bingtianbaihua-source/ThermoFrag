"""Sample fragment-assembly graphs from a trained CouplingPotential via MH.

Phase-2 sampler. Given a checkpoint of the CouplingPotential and a seed pool of
fragment graphs (usually the same ZINC LMDB used for training), runs
``FragmentNodeFlipMH`` for ``--mh-steps`` sweeps per chain and writes the
resulting frag_id arrays + (unchanged) edge structures to disk.

Output format: a single .pkl with a list of dicts::

    {
      "frag_id":    List[int],
      "edge_index": List[Tuple[int,int]],  # kept from the seed graph (MH only flips nodes)
      "bond_type":  List[int],
      "init_idx":   int,                    # index of the seed in the data pool
    }

This is what ``scripts/eval_properties.py`` consumes to evaluate the Phase-2
exit criterion (KL of property/fragment marginals vs ZINC).

Why we keep the edge structure frozen: the Phase-2 MH kernel only proposes node
relabelings; it does not change the graph connectivity. The point of Phase 2 is
for V to learn which fragment *types* sit on which positions — the topological
skeleton is supplied by the seed distribution. Full connectivity-changing
proposals are a Phase-3 deliverable.

Usage::

    python scripts/sample_unconditional.py \
        --checkpoint results/checkpoints/coupling_final.pt \
        --data data/processed/zinc_unconditional.lmdb \
        --n 2000 --mh-steps 50 \
        --out results/eval/phase2/samples.pkl
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.sampling.fragment_mh import FragmentMHStats, FragmentNodeFlipMH


def _load_coupling(ckpt_path: Path, device: str) -> tuple[CouplingPotential, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" not in blob or "cfg" not in blob:
        raise ValueError(f"{ckpt_path} is not a Trainer checkpoint")
    cfg = blob["cfg"]
    mc = cfg["model"]["coupling"]
    model = CouplingPotential(
        n_fragments=int(mc["n_fragments"]),
        n_bond_types=int(mc.get("n_bond_types", 8)),
        hidden=int(mc["hidden"]),
        num_layers=int(mc["num_layers"]),
    )
    model.load_state_dict(blob["state_dict"], strict=True)
    model.to(device).eval()
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="ZINC LMDB path (used as seed pool)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=2000, help="number of chains (= number of samples)")
    p.add_argument("--mh-steps", type=int, default=50, help="MH sweeps per chain")
    p.add_argument("--batch-size", type=int, default=256, help="graphs per MH forward")
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sample] loading checkpoint {args.checkpoint}")
    model, cfg = _load_coupling(args.checkpoint, device)
    print(f"[sample]   n_fragments={cfg['model']['coupling']['n_fragments']}  hidden={cfg['model']['coupling']['hidden']}")

    print(f"[sample] loading seed pool from {args.data} (split={args.split})")
    pool = ZINCFragmentDataset(args.data, split=args.split)
    if pool.n_fragments != int(cfg["model"]["coupling"]["n_fragments"]):
        raise ValueError(
            f"dataset n_fragments={pool.n_fragments} != ckpt n_fragments={cfg['model']['coupling']['n_fragments']}; "
            f"checkpoint and LMDB are incompatible."
        )
    print(f"[sample]   pool size={len(pool)}")

    # Sample seed indices (deterministic).
    g = torch.Generator().manual_seed(args.seed)
    n = min(int(args.n), len(pool))
    seed_idxs = torch.randperm(len(pool), generator=g)[:n].tolist()
    print(f"[sample] n_chains={n}  mh_steps={args.mh_steps}  batch_size={args.batch_size}")

    mh = FragmentNodeFlipMH(
        coupling=lambda b: model(b),
        n_fragments=int(cfg["model"]["coupling"]["n_fragments"]),
        beta=float(args.beta),
    )

    # Process in batches of graphs.
    stats = FragmentMHStats()
    outputs: list[dict] = []
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        idxs = seed_idxs[start : start + args.batch_size]
        data_list = [pool[i] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)
        for _ in range(args.mh_steps):
            batch = mh.step(batch, stats=stats)
        # Split batch back to Data list for saving.
        batch_cpu = batch.cpu()
        for j, local in enumerate(batch_cpu.to_data_list()):
            outputs.append({
                "frag_id": local.frag_id.tolist(),
                "edge_index": local.edge_index.t().tolist() if local.edge_index.numel() else [],
                "bond_type": local.bond_type.tolist(),
                "init_idx": idxs[j],
            })
        done = min(start + args.batch_size, n)
        dt = time.time() - t0
        print(
            f"[sample]   {done}/{n}  {done / max(dt, 1e-6):.1f} chain/s  "
            f"accept_rate={stats.accept_rate:.3f}"
        )

    with open(args.out, "wb") as f:
        pickle.dump({
            "n_chains": len(outputs),
            "mh_steps": args.mh_steps,
            "beta": args.beta,
            "accept_rate": stats.accept_rate,
            "samples": outputs,
        }, f, protocol=4)
    print(f"[sample] wrote {len(outputs)} samples -> {args.out}  accept_rate={stats.accept_rate:.3f}")


if __name__ == "__main__":
    main()
