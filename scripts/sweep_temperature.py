"""Fig 3 — temperature-controlled exploration / precision tradeoff.

The positioning story in docs/FIGURES.md §3: ThermoFrag's β knob is supposed
to do what statistical mechanics says — lower β (higher temperature) explores
more, higher β concentrates on the property-target.

Protocol:
  For each β in ``--betas``:
    1. Sample ``--n`` fragment graphs from the conditional sampler at that β.
    2. Decode to SMILES via ``FragmentLibraryIndex.decode_pool``.
    3. Measure:
       - Diversity: average pairwise Tanimoto distance on Morgan fingerprints
         (lower → less diverse, more concentrated).
       - Target hit rate: fraction of generated molecules whose computed
         ``phi_std`` on the conditioned-axis is within ``--hit-sigma`` of the
         target.
  Plot diversity vs β and hit-rate vs β.

Outputs::

    results/eval/phase5/fig3_temperature.{json, png}

This figure is the positioning argument for reviewers who pattern-match on
"temperature as softmax hack". It shows that β is a real inverse-temperature.
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.data.properties import compute_phi
from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.sampling.conditional_mh import (
    ConditionalFragmentMH,
    ConditionalMHStats,
    build_frag_phi_table,
)
from thermofrag.sampling.decoder import FragmentLibraryIndex, decode_pool
from thermofrag.utils.config import load_config


logger = logging.getLogger("beta_sweep")


def _load_joint(ckpt_path: Path, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = blob["cfg"]
    mc = cfg["model"]["coupling"]
    me = cfg["model"]["external_field"]
    coupling = CouplingPotential(
        n_fragments=int(mc["n_fragments"]),
        n_bond_types=int(mc.get("n_bond_types", 8)),
        hidden=int(mc["hidden"]),
        num_layers=int(mc["num_layers"]),
    )
    mu = ChemicalPotentialHead(n_properties=len(me["properties"]), hidden=int(me["hidden"]))
    sd = blob["state_dict"]
    coupling.load_state_dict({k[len("coupling."):]: v for k, v in sd.items()
                              if k.startswith("coupling.")}, strict=True)
    mu.load_state_dict({k[len("mu."):]: v for k, v in sd.items()
                        if k.startswith("mu.")}, strict=True)
    coupling.to(device).eval()
    mu.to(device).eval()
    return coupling, mu, cfg


def pairwise_tanimoto_distance(smiles_list: list[str], n_max_pairs: int = 4000,
                               seed: int = 0) -> float | None:
    """Sample pairs and compute average Tanimoto *distance* (1 − similarity).

    Higher distance = more diverse.
    """
    if len(smiles_list) < 2:
        return None
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    if len(fps) < 2:
        return None
    rng = np.random.default_rng(seed)
    N = len(fps)
    n_pairs = min(n_max_pairs, N * (N - 1) // 2)
    pairs = set()
    dists = []
    while len(pairs) < n_pairs:
        i, j = rng.integers(N), rng.integers(N)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in pairs:
            continue
        pairs.add(key)
        sim = DataStructs.TanimotoSimilarity(fps[key[0]], fps[key[1]])
        dists.append(1.0 - sim)
    return float(np.mean(dists))


def target_hit_rate(smiles_list: list[str], y_raw: np.ndarray,
                    axis: int, phi_mean: np.ndarray, phi_std: np.ndarray,
                    properties: list[str], hit_sigma: float = 0.5) -> tuple[float, int]:
    """Fraction of molecules within ``hit_sigma`` standard deviations of the
    target ``y_raw[axis]`` on the corresponding property axis."""
    if not smiles_list:
        return 0.0, 0
    tgt_z = (y_raw[axis] - phi_mean[axis]) / phi_std[axis]
    tol_z = hit_sigma
    hits = 0
    n_parse = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            phi = compute_phi(mol, properties)
        except Exception:
            continue
        n_parse += 1
        phi_z = (phi[axis] - phi_mean[axis]) / phi_std[axis]
        if abs(phi_z - tgt_z) <= tol_z:
            hits += 1
    return hits / max(n_parse, 1), n_parse


def sample_at_beta(coupling, mu, frag_phi, phi_mean_t, phi_std_t, pool,
                   n: int, beta: float, mh_steps: int, batch_size: int,
                   y_std: np.ndarray, device: str, seed: int) -> list:
    kernel = ConditionalFragmentMH(
        coupling=lambda b: coupling(b),
        mu_head=lambda y: mu(y),
        frag_phi=frag_phi,
        phi_mean=phi_mean_t,
        phi_std=phi_std_t,
        n_fragments=frag_phi.shape[0],
        beta=float(beta),
    )
    g = torch.Generator().manual_seed(int(seed))
    n = min(n, len(pool))
    seed_idxs = torch.randperm(len(pool), generator=g)[:n].tolist()
    y_template = torch.from_numpy(y_std.astype(np.float32)).to(device)
    samples = []
    stats = ConditionalMHStats(H_mean_history=[])
    for start in range(0, n, batch_size):
        idxs = seed_idxs[start:start + batch_size]
        data_list = [pool[i] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)
        B = int(batch.num_graphs)
        y = y_template.unsqueeze(0).expand(B, -1).contiguous()
        _, _ = kernel.run(batch, y, n_steps=mh_steps, stats=stats)
        batch_cpu = batch.cpu()
        for j, local in enumerate(batch_cpu.to_data_list()):
            samples.append({
                "frag_id": local.frag_id.tolist(),
                "edge_index": local.edge_index.t().tolist() if local.edge_index.numel() else [],
                "bond_type": local.bond_type.tolist(),
                "init_idx": idxs[j],
                "seed_smiles": getattr(local, "smiles", None),
            })
    return samples, stats.accept_rate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results/checkpoints/joint_final.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--library", type=Path,
                   default=Path("data/processed/fragment_library.parquet"))
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--mh-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--betas", nargs="*", type=float,
                   default=[0.1, 0.3, 1.0, 3.0, 10.0])
    p.add_argument("--target-axis", default="qed",
                   help="Name of the property axis whose hit rate we measure.")
    p.add_argument("--target-value", type=float, default=0.7,
                   help="Raw target value on --target-axis.")
    p.add_argument("--hit-sigma", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase5"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    cfg = load_config(args.config) if args.config.exists() else None
    data_path = args.data or Path(cfg["data"]["conditional"])

    coupling, mu, ckpt_cfg = _load_joint(args.checkpoint, device)
    n_fragments = int(ckpt_cfg["model"]["coupling"]["n_fragments"])
    properties = list(ckpt_cfg["model"]["external_field"]["properties"])
    pool = ZINCFragmentDataset(data_path, split="train")
    phi_mean_np = pool.phi_mean; phi_std_np = pool.phi_std
    phi_mean_t = torch.from_numpy(phi_mean_np.astype(np.float32)).to(device)
    phi_std_t  = torch.from_numpy(phi_std_np.astype(np.float32)).to(device)

    frag_phi_np = build_frag_phi_table(args.library, properties)[:n_fragments]
    frag_phi = torch.from_numpy(frag_phi_np).to(device)

    if args.target_axis not in properties:
        raise SystemExit(f"unknown axis {args.target_axis}; valid={properties}")
    axis = properties.index(args.target_axis)
    y_raw = phi_mean_np.astype(np.float32).copy()
    y_raw[axis] = float(args.target_value)
    y_std_vec = (y_raw - phi_mean_np) / phi_std_np
    logger.info("target %s=%.3f  y_std=%s", args.target_axis, y_raw[axis],
                y_std_vec.round(2).tolist())

    lib = FragmentLibraryIndex.from_parquet(args.library)
    rows = []
    for beta in args.betas:
        logger.info("=== β = %.3f ===", beta)
        t0 = time.time()
        samples, accept = sample_at_beta(
            coupling, mu, frag_phi, phi_mean_t, phi_std_t, pool,
            n=args.n, beta=beta, mh_steps=args.mh_steps,
            batch_size=args.batch_size, y_std=y_std_vec,
            device=device, seed=0)
        dt = time.time() - t0
        # Decode.
        results = decode_pool(samples, lib)
        smiles = [r.smiles for r in results if r.smiles is not None]
        n_valid = len(smiles)
        # Metrics.
        diversity = pairwise_tanimoto_distance(smiles)
        hit_rate, n_parse = target_hit_rate(
            smiles, y_raw, axis, phi_mean_np, phi_std_np,
            properties=properties, hit_sigma=args.hit_sigma)
        logger.info("  dt=%.1fs  accept=%.3f  n_valid=%d/%d  diversity=%s  hit_rate@%s=%.3f (n=%d)",
                    dt, accept, n_valid, len(samples),
                    f"{diversity:.3f}" if diversity is not None else "NA",
                    f"±{args.hit_sigma}σ", hit_rate, n_parse)
        rows.append({"beta": float(beta),
                     "accept_rate": float(accept),
                     "n_samples": int(len(samples)),
                     "n_valid": int(n_valid),
                     "diversity": float(diversity) if diversity is not None else None,
                     "hit_rate": float(hit_rate),
                     "hit_rate_n": int(n_parse),
                     "runtime_s": float(dt)})

    # Save.
    report = {
        "target_axis":  args.target_axis,
        "target_value": float(args.target_value),
        "hit_sigma":    float(args.hit_sigma),
        "properties":   properties,
        "rows":         rows,
    }
    report_path = args.out_dir / "fig3_temperature.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("report → %s", report_path)

    # Plot.
    if any(r["diversity"] is not None for r in rows):
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
        betas = np.array([r["beta"] for r in rows], dtype=float)
        diversity = np.array([r["diversity"] if r["diversity"] is not None else np.nan
                              for r in rows], dtype=float)
        hits = np.array([r["hit_rate"] for r in rows], dtype=float)

        ax = axes[0]
        ax.plot(betas, diversity, "o-", color="#d62728", lw=2.0, ms=8)
        ax.set_xscale("log")
        ax.set_xlabel(r"inverse temperature $\beta$")
        ax.set_ylabel(r"pairwise Tanimoto distance $\langle 1 - s\rangle$")
        ax.set_title("Fig 3a — exploration collapses as β rises")

        ax = axes[1]
        ax.plot(betas, hits, "o-", color="#1f77b4", lw=2.0, ms=8)
        ax.set_xscale("log")
        ax.set_xlabel(r"inverse temperature $\beta$")
        ax.set_ylabel(f"target hit rate  ({args.target_axis}={args.target_value:g} ±{args.hit_sigma}σ)")
        ax.set_title("Fig 3b — property-target concentration grows with β")
        fig.tight_layout()
        fig.savefig(args.out_dir / "fig3_temperature.png", dpi=150)
        logger.info("fig → %s", args.out_dir / "fig3_temperature.png")


if __name__ == "__main__":
    main()
