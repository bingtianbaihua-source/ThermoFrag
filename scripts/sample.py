"""Phase-4 conditional sampler: draw fragment-graph samples from the joint model.

Loads the Phase-3 ``joint_final.pt`` checkpoint (CouplingMuModule: V + μ), builds
a per-fragment property lookup table from the fragment library, then runs
:class:`ConditionalFragmentMH` for ``--mh-steps`` sweeps per chain using
conditional Hamiltonian

    H(m; y) = V^couple(m) - mu(y) . phi(m).

Seed graphs are drawn uniformly from a ZINC fragment LMDB (same convention as
``scripts/sample_unconditional.py``) and the generated fragment-id assignments
are written to a ``.pkl`` in the same schema Phase 2 evaluation consumes, plus
two extra fields per sample:

    y_raw     : the raw conditioning target (pre-standardization)
    y_std     : the standardized conditioning target the μ head saw

The y vector can be supplied three ways:
  * ``--y "logP=2.5,qed=0.7,..."`` comma-separated raw values (missing keys default to the training mean of that property, i.e. y_std=0 on that axis).
  * ``--y-file path.npy`` a length-K numpy array of raw y values.
  * default: y_raw = phi_mean (i.e. y_std = 0 — the "no-condition" draw).

Usage::

    python scripts/sample.py \
        --checkpoint results/checkpoints/joint_final.pt \
        --config configs/phase3.yaml \
        --data data/processed/chembl_conditional.lmdb \
        --library data/processed/fragment_library.parquet \
        --y "logP=3.0,qed=0.8" \
        --n 200 --mh-steps 60 \
        --out results/eval/phase4/samples_default.pkl
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
from thermofrag.potentials.external_field import (
    ChemicalPotentialHead,
    PocketConditionalChemicalPotentialHead,
)
from thermofrag.potentials.pocket_coupling import PocketLigandCoupling
from thermofrag.sampling.conditional_mh import (
    ConditionalFragmentMH,
    ConditionalMHStats,
    build_frag_phi_table,
)
from thermofrag.utils.config import load_config


def _parse_y_string(s: str, properties: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in s.split(","):
        if not tok.strip():
            continue
        k, v = tok.split("=", 1)
        k = k.strip()
        if k not in properties:
            raise SystemExit(f"[sample] unknown property '{k}'; valid: {properties}")
        out[k] = float(v)
    return out


def _resolve_y(
    args: argparse.Namespace,
    properties: list[str],
    phi_mean: np.ndarray,
) -> np.ndarray:
    """Return the raw (pre-standardization) y vector, shape [K]."""
    K = len(properties)
    if args.y_file:
        arr = np.load(args.y_file)
        if arr.shape != (K,):
            raise SystemExit(f"[sample] --y-file must be shape ({K},); got {arr.shape}")
        return arr.astype(np.float32)
    if args.y:
        overrides = _parse_y_string(args.y, properties)
        y = phi_mean.astype(np.float32).copy()
        for k, v in overrides.items():
            y[properties.index(k)] = v
        return y
    return phi_mean.astype(np.float32).copy()


def _load_joint_checkpoint(ckpt_path: Path, device: str):
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" not in blob or "cfg" not in blob:
        raise ValueError(f"{ckpt_path} is not a Trainer checkpoint")
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
    coupling_sd = {k[len("coupling."):]: v for k, v in sd.items() if k.startswith("coupling.")}
    mu_sd = {k[len("mu."):]: v for k, v in sd.items() if k.startswith("mu.")}
    coupling.load_state_dict(coupling_sd, strict=True)
    mu.load_state_dict(mu_sd, strict=True)
    coupling.to(device).eval()
    mu.to(device).eval()
    return coupling, mu, cfg


def _load_pocket_mu_head(
    pocket_ckpt: Path,
    n_properties: int,
    hidden: int,
    device: str,
) -> tuple[PocketConditionalChemicalPotentialHead, int]:
    """Load a TF-pocket μ-only checkpoint saved by scripts/train_pocket_variant.py.

    The checkpoint holds ``state_dict`` of the pocket head plus ``pocket_dim``.
    Returns (loaded_head, pocket_dim). ``n_properties`` / ``hidden`` must match
    the joint checkpoint's TF-base μ head so the sampler's coupling + μ stay
    shape-compatible (same φ dimension, same hidden).
    """
    blob = torch.load(str(pocket_ckpt), map_location=device, weights_only=False)
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise ValueError(f"{pocket_ckpt} is not a TF-pocket checkpoint")
    pocket_dim = int(blob.get("pocket_dim") or blob.get("cfg", {}).get("model", {}).get("external_field", {}).get("pocket_dim", 0))
    if pocket_dim <= 0:
        raise ValueError(f"pocket_dim missing or non-positive in {pocket_ckpt}")
    head = PocketConditionalChemicalPotentialHead(
        n_properties=n_properties, pocket_dim=pocket_dim, hidden=hidden,
    )
    missing, unexpected = head.load_state_dict(blob["state_dict"], strict=False)
    if missing:
        print(f"[sample]   pocket head: missing keys {missing}")
    if unexpected:
        print(f"[sample]   pocket head: unexpected keys {unexpected}")
    head.to(device).eval()
    return head, pocket_dim


def _load_pocket_embed(path: Path, expected_dim: int, device: str) -> torch.Tensor:
    arr = np.load(str(path))
    if arr.ndim != 1 or arr.shape[0] != expected_dim:
        raise ValueError(
            f"pocket embed at {path} has shape {arr.shape}; expected ({expected_dim},) "
            f"to match TF-pocket checkpoint pocket_dim"
        )
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def _load_v_pocket(
    ckpt_path: Path,
    phi_dim: int,
    pocket_dim: int,
    device: str,
) -> PocketLigandCoupling:
    """Load a V^pocket(m, p) checkpoint saved by scripts/train_pocket_variant.py.

    The checkpoint persists ``state_dict`` of ``PocketLigandCoupling`` plus
    ``phi_dim`` / ``pocket_dim`` / ``pocket_hidden`` / ``mlp_hidden``. Raises
    if the shapes don't match the sampler-side phi / pocket dimensions.
    """
    blob = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(blob, dict) or "v_pocket_state_dict" not in blob:
        raise ValueError(f"{ckpt_path} has no v_pocket_state_dict")
    sd = blob["v_pocket_state_dict"]
    meta = blob.get("v_pocket_meta", {})
    head = PocketLigandCoupling(
        phi_dim=phi_dim,
        pocket_dim=pocket_dim,
        pocket_hidden=int(meta.get("pocket_hidden", 64)),
        mlp_hidden=int(meta.get("mlp_hidden", 128)),
    )
    missing, unexpected = head.load_state_dict(sd, strict=False)
    if missing:
        print(f"[sample]   V^pocket: missing keys {missing}")
    if unexpected:
        print(f"[sample]   V^pocket: unexpected keys {unexpected}")
    head.to(device).eval()
    print(
        f"[sample] V^pocket loaded from {ckpt_path}  "
        f"vina_mean={float(head.vina_mean):.3f}  "
        f"vina_scale={float(head.vina_scale):.3f}  "
        f"params={sum(p.numel() for p in head.parameters())/1e6:.3f}M"
    )
    return head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"),
                   help="config used to resolve default data paths when CLI flags are absent")
    p.add_argument("--data", type=Path, default=None,
                   help="seed pool LMDB; defaults to cfg.data.conditional")
    p.add_argument("--library", type=Path, default=Path("data/processed/fragment_library.parquet"))
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--y", type=str, default=None, help="comma-separated raw property overrides, e.g. 'logP=3.0,qed=0.7'")
    p.add_argument("--y-file", type=Path, default=None, help=".npy of raw y, length = n_properties")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--mh-steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pocket-ckpt", type=Path, default=None,
                   help="TF-pocket μ-only checkpoint (tf_pocket_final.pt). Replaces "
                        "the coupling checkpoint's μ head with a pocket-conditional one.")
    p.add_argument("--pocket-embed", type=Path, default=None,
                   help=".npy of the conditioning pocket embedding (required with --pocket-ckpt). "
                        "E.g. data/processed/pocket_embeds/litpcba/VDR.npy")
    p.add_argument("--v-pocket-ckpt", type=Path, default=None,
                   help="TF-pocket-v2 V^pocket(m, p) checkpoint. Adds the pocket-ligand "
                        "coupling term to the MH Hamiltonian so the sampler rewards molecules "
                        "the network predicts will dock well. Requires --pocket-embed.")
    p.add_argument("--v-pocket-weight", type=float, default=1.0,
                   help="scalar multiplier on V^pocket in H. 1.0 uses the calibrated kcal/mol "
                        "output directly; set lower to soften the pocket pull.")
    args = p.parse_args()
    if (args.pocket_ckpt is None) != (args.pocket_embed is None):
        p.error("--pocket-ckpt and --pocket-embed must be passed together")
    if args.v_pocket_ckpt is not None and args.pocket_embed is None:
        p.error("--v-pocket-ckpt requires --pocket-embed")

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config) if args.config.exists() else None
    data_path = args.data or Path(cfg["data"]["conditional"])
    print(f"[sample] data={data_path} split={args.split}")

    # --- Load model ---------------------------------------------------------
    coupling, mu, ckpt_cfg = _load_joint_checkpoint(args.checkpoint, device)
    n_fragments = int(ckpt_cfg["model"]["coupling"]["n_fragments"])
    properties = list(ckpt_cfg["model"]["external_field"]["properties"])
    hidden = int(ckpt_cfg["model"]["external_field"]["hidden"])
    print(
        f"[sample] ckpt n_fragments={n_fragments}  properties={properties}  "
        f"params_V={sum(p.numel() for p in coupling.parameters())/1e6:.2f}M  "
        f"params_mu={sum(p.numel() for p in mu.parameters())/1e6:.2f}M"
    )

    # --- TF-pocket branch: swap μ for the pocket-conditional head ----------
    pocket_mu: PocketConditionalChemicalPotentialHead | None = None
    pocket_vec: torch.Tensor | None = None
    v_pocket: PocketLigandCoupling | None = None
    pocket_dim = 0
    if args.pocket_ckpt is not None:
        pocket_mu, pocket_dim = _load_pocket_mu_head(
            args.pocket_ckpt, n_properties=len(properties), hidden=hidden, device=device,
        )
        pocket_vec = _load_pocket_embed(args.pocket_embed, pocket_dim, device)
        print(
            f"[sample] TF-pocket μ loaded from {args.pocket_ckpt}  "
            f"pocket_dim={pocket_dim}  pocket={args.pocket_embed.name}  "
            f"norm={float(torch.linalg.norm(pocket_vec)):.2f}  "
            f"params_mu={sum(p.numel() for p in pocket_mu.parameters())/1e6:.2f}M"
        )
    if args.v_pocket_ckpt is not None:
        # If V^pocket alone is passed, we still need pocket_dim for the embed check.
        if pocket_vec is None:
            # Peek at the checkpoint to learn pocket_dim.
            _blob = torch.load(str(args.v_pocket_ckpt), map_location="cpu", weights_only=False)
            _sd = _blob.get("v_pocket_state_dict", {})
            _w = _sd.get("pocket_proj.0.weight")
            if _w is None:
                raise SystemExit(f"V^pocket ckpt missing pocket_proj.0.weight")
            pocket_dim = int(_w.shape[1])
            pocket_vec = _load_pocket_embed(args.pocket_embed, pocket_dim, device)
            del _blob
        v_pocket = _load_v_pocket(
            args.v_pocket_ckpt, phi_dim=len(properties), pocket_dim=pocket_dim, device=device,
        )

    # --- Load seed pool -----------------------------------------------------
    pool = ZINCFragmentDataset(data_path, split=args.split)
    if pool.n_fragments != n_fragments:
        raise ValueError(
            f"dataset n_fragments={pool.n_fragments} != ckpt n_fragments={n_fragments}"
        )
    phi_mean_np = pool.phi_mean
    phi_std_np = pool.phi_std
    if phi_mean_np is None:
        raise ValueError("seed LMDB missing phi_mean/phi_std; use the conditional LMDB")
    print(f"[sample] seed pool={len(pool)}  phi_mean={phi_mean_np.round(3).tolist()}")

    # --- Build frag-phi table ----------------------------------------------
    print(f"[sample] building per-fragment phi table from {args.library}")
    t0 = time.time()
    frag_phi_np = build_frag_phi_table(args.library, properties)  # [V, K]
    if frag_phi_np.shape[0] < n_fragments:
        raise RuntimeError(
            f"fragment library has {frag_phi_np.shape[0]} entries but model expects {n_fragments}; rebuild the library"
        )
    frag_phi_np = frag_phi_np[:n_fragments]
    print(f"[sample]   built in {time.time()-t0:.1f}s  shape={frag_phi_np.shape}")

    frag_phi = torch.from_numpy(frag_phi_np).to(device)
    phi_mean = torch.from_numpy(np.asarray(phi_mean_np, dtype=np.float32)).to(device)
    phi_std = torch.from_numpy(np.asarray(phi_std_np, dtype=np.float32)).to(device)

    # --- Resolve conditioning target y -------------------------------------
    y_raw = _resolve_y(args, properties, phi_mean_np)  # [K]
    y_std = (y_raw - phi_mean_np) / phi_std_np
    print(f"[sample] y_raw = {dict(zip(properties, y_raw.round(3).tolist()))}")
    print(f"[sample] y_std = {y_std.round(3).tolist()}")

    y_tensor_template = torch.from_numpy(y_std).to(device).float()  # [K]

    # --- Run sampler -------------------------------------------------------
    if pocket_mu is not None:
        # All chains share the same pocket (same LIT-PCBA target per run). Bake it
        # into the closure so the MH kernel keeps its plain ``mu(y)`` signature.
        _pocket_for_mu = pocket_vec  # captured

        def mu_head_fn(y: torch.Tensor) -> torch.Tensor:
            p = _pocket_for_mu.unsqueeze(0).expand(y.shape[0], -1)
            return pocket_mu(y, p)
    else:
        def mu_head_fn(y: torch.Tensor) -> torch.Tensor:
            return mu(y)

    v_pocket_fn = None
    if v_pocket is not None:
        _v_pocket_for_H = v_pocket  # captured
        _pocket_for_V = pocket_vec  # captured
        _w = float(args.v_pocket_weight)

        def v_pocket_fn(phi_z: torch.Tensor) -> torch.Tensor:
            p = _pocket_for_V.unsqueeze(0).expand(phi_z.shape[0], -1)
            return _w * _v_pocket_for_H(phi_z, p)

    kernel = ConditionalFragmentMH(
        coupling=lambda b: coupling(b),
        mu_head=mu_head_fn,
        frag_phi=frag_phi,
        phi_mean=phi_mean,
        phi_std=phi_std,
        n_fragments=n_fragments,
        beta=float(args.beta),
        v_pocket_fn=v_pocket_fn,
    )

    g = torch.Generator().manual_seed(int(args.seed))
    n = min(int(args.n), len(pool))
    seed_idxs = torch.randperm(len(pool), generator=g)[:n].tolist()
    print(f"[sample] n_chains={n}  mh_steps={args.mh_steps}  batch_size={args.batch_size}  beta={args.beta}")

    outputs: list[dict] = []
    stats = ConditionalMHStats(H_mean_history=[])
    t0 = time.time()
    for start in range(0, n, args.batch_size):
        idxs = seed_idxs[start : start + args.batch_size]
        data_list = [pool[i] for i in idxs]
        batch = Batch.from_data_list(data_list).to(device)
        B = int(batch.num_graphs)
        y = y_tensor_template.unsqueeze(0).expand(B, -1).contiguous()
        _, _ = kernel.run(batch, y, n_steps=int(args.mh_steps), stats=stats)

        batch_cpu = batch.cpu()
        for j, local in enumerate(batch_cpu.to_data_list()):
            outputs.append({
                "frag_id": local.frag_id.tolist(),
                "edge_index": local.edge_index.t().tolist() if local.edge_index.numel() else [],
                "bond_type": local.bond_type.tolist(),
                "init_idx": idxs[j],
                "seed_smiles": getattr(local, "smiles", None),
            })
        done = min(start + args.batch_size, n)
        dt = time.time() - t0
        print(
            f"[sample]   {done}/{n}  {done/max(dt,1e-6):.1f} chain/s  "
            f"accept_rate={stats.accept_rate:.3f}  "
            f"H_last={stats.H_mean_history[-1] if stats.H_mean_history else float('nan'):.2f}"
        )

    payload = {
        "n_chains": len(outputs),
        "mh_steps": args.mh_steps,
        "beta": args.beta,
        "accept_rate": stats.accept_rate,
        "y_raw": y_raw.tolist(),
        "y_std": y_std.tolist(),
        "properties": properties,
        "checkpoint": str(args.checkpoint),
        "pocket_ckpt": str(args.pocket_ckpt) if args.pocket_ckpt else None,
        "pocket_embed": str(args.pocket_embed) if args.pocket_embed else None,
        "v_pocket_ckpt": str(args.v_pocket_ckpt) if args.v_pocket_ckpt else None,
        "v_pocket_weight": float(args.v_pocket_weight) if v_pocket is not None else None,
        "variant": (
            "tf_pocket_v2"
            if v_pocket is not None
            else ("tf_pocket" if pocket_mu is not None else "tf_base")
        ),
        "H_history": stats.H_mean_history,
        "samples": outputs,
    }
    with open(args.out, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"[sample] wrote {len(outputs)} samples -> {args.out}  accept_rate={stats.accept_rate:.3f}")


if __name__ == "__main__":
    main()
