"""Train ThermoFrag.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/tiny.yaml --max-steps 100 \
        --data-qm-train data/processed/spice_shards
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch.nn as nn

from thermofrag.potentials.coupling import CouplingPotential
from thermofrag.potentials.external_field import ChemicalPotentialHead
from thermofrag.sampling.fragment_mh import FragmentNodeFlipMH
from thermofrag.training.pcd import PCDBuffer
from thermofrag.training.trainer import Trainer, build_qm_head, build_spice_loaders
from thermofrag.utils.config import load_config
from thermofrag.utils.seed import seed_everything


class CouplingMuModule(nn.Module):
    """Wraps Coupling + Mu so Trainer can see both submodules and optimize jointly."""

    def __init__(self, coupling: CouplingPotential, mu: ChemicalPotentialHead):
        super().__init__()
        self.coupling = coupling
        self.mu = mu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="override run.device")
    p.add_argument("--data-qm-train", type=str, default=None, help="override data.qm_train shard dir")
    p.add_argument("--data-qm-val", type=str, default=None, help="override data.qm_val shard dir")
    p.add_argument("--data-unconditional", type=str, default=None, help="override data.unconditional LMDB path")
    p.add_argument("--pool-size", type=int, default=None, help="limit Phase-2 data pool size (smoke runs)")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.data_qm_train:
        cfg["data"]["qm_train"] = args.data_qm_train
    if args.data_qm_val:
        cfg["data"]["qm_val"] = args.data_qm_val
    if args.data_unconditional:
        cfg["data"]["unconditional"] = args.data_unconditional
    if args.device:
        cfg["run"]["device"] = args.device

    seed_everything(cfg["run"]["seed"])
    phase = cfg["training"]["phase"]
    print(f"[train] phase={phase} device={cfg['run']['device']} precision={cfg['run'].get('precision','fp32')}")

    if phase == "pretrain_qm":
        model = build_qm_head(cfg)
        train_loader, val_loader = build_spice_loaders(cfg)
        print(
            f"[train] QM params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
            f"train_size={len(train_loader.dataset)}  "
            f"val_size={len(val_loader.dataset) if val_loader else 0}"
        )
        trainer = Trainer(cfg, model, train_loader, val_loader, device=cfg["run"]["device"])
        out = trainer.fit(max_steps=args.max_steps)

    elif phase == "pretrain_coupling":
        out = _run_pretrain_coupling(cfg, max_steps=args.max_steps, pool_size=args.pool_size)
        trainer = out.pop("_trainer")  # used below for metrics/ckpt paths

    elif phase == "joint_conditional":
        out = _run_joint_conditional(cfg, max_steps=args.max_steps, pool_size=args.pool_size)
        trainer = out.pop("_trainer")

    else:
        raise SystemExit(f"phase '{phase}' not wired yet (joint is Phase 3 milestone)")

    print(f"[train] done. final step={out['step']} loss_ema={out.get('loss_ema')}")
    print(f"[train] metrics -> {trainer.metrics_path}")
    print(f"[train] checkpoints -> {trainer.ckpt_dir}")


def _run_pretrain_coupling(cfg: dict, *, max_steps: int | None, pool_size: int | None) -> dict:
    """Assemble CouplingPotential + PCD + MH and run Trainer._fit_coupling.

    Returns the trainer's ``fit`` output dict plus ``_trainer`` for caller-side logging.
    """
    from thermofrag.data.zinc_fragments import ZINCFragmentDataset

    lmdb_path = cfg["data"]["unconditional"]
    print(f"[train] loading ZINC fragment dataset from {lmdb_path}")
    train_ds = ZINCFragmentDataset(lmdb_path, split="train")
    if pool_size is not None:
        # Trainer indexes data_pool by integer; a plain slice-view keeps it indexable
        # while limiting LMDB reads to the requested prefix.
        from torch.utils.data import Subset
        train_ds_used: object = Subset(train_ds, range(min(pool_size, len(train_ds))))
    else:
        train_ds_used = train_ds
    print(f"[train]   n_train={len(train_ds)}  pool_used={len(train_ds_used)}  n_fragments={train_ds.n_fragments}")

    # Resolve vocabulary size (prefer dataset meta over config override).
    mc = cfg["model"]["coupling"]
    n_fragments = int(mc.get("n_fragments") or train_ds.n_fragments)
    if n_fragments < train_ds.n_fragments:
        raise ValueError(
            f"coupling.n_fragments={n_fragments} is smaller than the dataset vocabulary "
            f"{train_ds.n_fragments}; data contains ids beyond the embedding range."
        )
    mc["n_fragments"] = n_fragments  # record the resolved value

    model = CouplingPotential(
        n_fragments=n_fragments,
        n_bond_types=int(mc.get("n_bond_types", 8)),
        hidden=int(mc["hidden"]),
        num_layers=int(mc["num_layers"]),
    )
    cws = mc.get("warm_start")
    if cws:
        import torch
        sd = torch.load(cws, map_location="cpu")
        model.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] CouplingPotential params={n_params/1e6:.2f}M  n_fragments={n_fragments}")

    trainer = Trainer(cfg, model, train_loader=None, val_loader=None, device=cfg["run"]["device"])

    # Seed the PCD buffer from the training data pool.
    pcd_cfg = cfg["training"]["pcd"]
    buffer = PCDBuffer(
        size=int(pcd_cfg["buffer_size"]),
        refresh_frac=float(pcd_cfg["refresh_frac"]),
        seed=int(cfg["run"].get("seed", 0)),
    )
    buffer.init_from_dataset(train_ds_used)

    # Wrap the coupling model for MH acceptance: it needs .to(device) and no autocast.
    mh_kernel = FragmentNodeFlipMH(
        coupling=lambda b: trainer.model(b),
        n_fragments=n_fragments,
        beta=float(mc.get("mh_beta", 1.0)),
    )

    trainer.attach_pcd(buffer, data_pool=train_ds_used, mh_kernel=mh_kernel)

    out = trainer.fit(max_steps=max_steps)
    out["_trainer"] = trainer
    return out


def _run_joint_conditional(cfg: dict, *, max_steps: int | None, pool_size: int | None) -> dict:
    """Phase 3: CouplingPotential + ChemicalPotentialHead joint fine-tune.

    Uses the conditional LMDB (cfg.data.conditional). Warm-starts coupling from
    cfg.model.coupling.warm_start if present (e.g. results/checkpoints/coupling_final.pt).
    """
    import torch
    from thermofrag.data.zinc_fragments import ZINCFragmentDataset

    lmdb_path = cfg["data"]["conditional"]
    print(f"[train] loading conditional LMDB from {lmdb_path}")
    train_ds = ZINCFragmentDataset(lmdb_path, split="train")
    if pool_size is not None:
        from torch.utils.data import Subset
        train_ds_used: object = Subset(train_ds, range(min(pool_size, len(train_ds))))
    else:
        train_ds_used = train_ds
    print(
        f"[train]   n_train={len(train_ds)}  pool_used={len(train_ds_used)}  "
        f"n_fragments={train_ds.n_fragments}  phi_dim={train_ds.phi_dim}  "
        f"phi_properties={train_ds.phi_properties}"
    )

    mc = cfg["model"]["coupling"]
    n_fragments = int(mc.get("n_fragments") or train_ds.n_fragments)
    if n_fragments < train_ds.n_fragments:
        raise ValueError(
            f"coupling.n_fragments={n_fragments} < dataset vocabulary {train_ds.n_fragments}"
        )
    mc["n_fragments"] = n_fragments

    coupling = CouplingPotential(
        n_fragments=n_fragments,
        n_bond_types=int(mc.get("n_bond_types", 8)),
        hidden=int(mc["hidden"]),
        num_layers=int(mc["num_layers"]),
    )
    cws = mc.get("warm_start")
    if cws:
        sd = torch.load(cws, map_location="cpu")
        if "state_dict" in sd:
            sd = sd["state_dict"]
        coupling.load_state_dict(sd, strict=False)
        print(f"[train] coupling warm-started from {cws}")

    me = cfg["model"]["external_field"]
    if len(me["properties"]) != train_ds.phi_dim:
        raise ValueError(
            f"config external_field.properties has {len(me['properties'])} entries but "
            f"dataset phi_dim={train_ds.phi_dim}; rebuild conditional LMDB or fix config"
        )
    mu = ChemicalPotentialHead(n_properties=train_ds.phi_dim, hidden=int(me["hidden"]))

    model = CouplingMuModule(coupling, mu)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] CouplingMu params={n_params/1e6:.2f}M (coupling+mu)")

    trainer = Trainer(cfg, model, train_loader=None, val_loader=None, device=cfg["run"]["device"])

    pcd_cfg = cfg["training"]["pcd"]
    buffer = PCDBuffer(
        size=int(pcd_cfg["buffer_size"]),
        refresh_frac=float(pcd_cfg["refresh_frac"]),
        seed=int(cfg["run"].get("seed", 0)),
    )
    buffer.init_from_dataset(train_ds_used)

    mh_kernel = FragmentNodeFlipMH(
        coupling=lambda b: trainer.model.coupling(b),
        n_fragments=n_fragments,
        beta=float(mc.get("mh_beta", 1.0)),
    )

    import torch
    phi_mean_np = train_ds.phi_mean
    phi_std_np = train_ds.phi_std
    if phi_mean_np is None or phi_std_np is None:
        raise RuntimeError("conditional LMDB meta missing phi_mean/phi_std; rebuild with the latest script")
    phi_mean = torch.as_tensor(phi_mean_np, dtype=torch.float32)
    phi_std = torch.as_tensor(phi_std_np, dtype=torch.float32)

    trainer.attach_joint_conditional(
        buffer, train_ds_used, mh_kernel, phi_mean=phi_mean, phi_std=phi_std
    )

    out = trainer.fit(max_steps=max_steps)
    out["_trainer"] = trainer
    return out


if __name__ == "__main__":
    main()
