"""Train TF-pocket v3: EGNN-over-Cα pocket encoder + μ(y, p) head.

Differences from ``scripts/train_pocket_variant.py`` (v1/v2):

* Pocket branch is **trainable**. We instantiate ``EGNNPocketEncoder`` over
  padded pocket Cα geometry loaded from ``CrossDockedPocketGeomDataset``.
  Its output plays the role of the ``.npy`` pocket vector in v1/v2 and
  feeds into the pre-existing ``PocketConditionalChemicalPotentialHead``.
* QM + coupling are still frozen (this script never touches them; μ +
  EGNN are the only trainable parameters).
* Warm-start: when ``--warm-start`` is passed, the μ head's ``y``-slice is
  copied from ``joint_final.pt`` as before. The EGNN is randomly
  initialised from scratch; to keep step-0 behaviour close to TF-base we
  zero-init ``out_proj``, so ``pocket_vec == 0`` at step 0 and
  ``mu(y, p=0)`` equals the TF-base μ.

Sampler-side contract: after training finishes we ship one ``.npy`` per
LIT-PCBA target via ``scripts/dump_pocket_v3_litpcba.py`` so
``scripts/sample.py --pocket-ckpt <v3_final.pt> --pocket-embed <.npy>``
Just Works without v3-specific sampler changes.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from thermofrag.data.crossdocked import (
    CrossDockedPocketGeomDataset,
    collate_pocket_geom_batch,
)
from thermofrag.potentials.external_field import (
    ChemicalPotentialHead,
    PocketConditionalChemicalPotentialHead,
)
from thermofrag.potentials.pocket_egnn import EGNNPocketEncoder
from thermofrag.utils.config import load_config
from thermofrag.utils.seed import seed_everything


def build_heads(cfg: dict, phi_dim: int) -> tuple[EGNNPocketEncoder, PocketConditionalChemicalPotentialHead]:
    me = cfg["model"]["external_field"]
    pe = cfg["model"]["pocket_encoder"]
    if len(me["properties"]) != phi_dim:
        raise ValueError(
            f"config external_field.properties has {len(me['properties'])} entries but "
            f"dataset phi_dim={phi_dim}"
        )
    if int(pe["embed_dim"]) != int(me["pocket_dim"]):
        raise ValueError(
            f"pocket_encoder.embed_dim={pe['embed_dim']} != external_field.pocket_dim={me['pocket_dim']}"
        )
    encoder = EGNNPocketEncoder(
        embed_dim=int(pe["embed_dim"]),
        n_layers=int(pe["n_layers"]),
        n_rbf=int(pe.get("n_rbf", 16)),
        cutoff_a=float(pe.get("cutoff_a", 10.0)),
    )
    head = PocketConditionalChemicalPotentialHead(
        n_properties=phi_dim,
        pocket_dim=int(me["pocket_dim"]),
        hidden=int(me["hidden"]),
    )
    return encoder, head


def maybe_warm_start(
    head: PocketConditionalChemicalPotentialHead,
    encoder: EGNNPocketEncoder,
    ckpt_path: Path,
) -> None:
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    mu_sd = {}
    for k, v in sd.items():
        if k.startswith("mu."):
            mu_sd[k[len("mu.") :]] = v
    if not mu_sd:
        raise RuntimeError(f"no mu.* keys in {ckpt_path}; is this a TF-base joint checkpoint?")
    base = ChemicalPotentialHead(n_properties=head.n_properties, hidden=head.hidden)
    missing, unexpected = base.load_state_dict(mu_sd, strict=False)
    if missing:
        print(f"[tf-pocket-v3 warm-start] base head missing {missing}")
    if unexpected:
        print(f"[tf-pocket-v3 warm-start] base head unexpected {unexpected}")
    head.warm_start_from_base(base)
    # Do NOT zero the EGNN's out_proj. The μ head's warm_start_from_base
    # already zeros the trunk[0] pocket-channel columns, which makes
    # ``mu(y, p_any) == mu_base(y)`` at step 0 regardless of the pocket
    # vector's value. Zeroing out_proj on top creates a dead saddle point
    # where gradients through the pocket branch are identically zero
    # (W_p = 0 means dL/dpocket = 0, and pocket = 0 means dL/dW_p = 0).
    # Keeping the EGNN at its default init breaks the symmetry.
    print(f"[tf-pocket-v3 warm-start] copied μ from {ckpt_path}; EGNN kept at scratch init")


def contrastive_loss(mu_pred_ij: torch.Tensor, phi_z: torch.Tensor) -> torch.Tensor:
    B = mu_pred_ij.shape[0]
    scores = (mu_pred_ij * phi_z.unsqueeze(1)).sum(dim=-1)  # [B, B]
    target = torch.arange(B, device=scores.device)
    return F.cross_entropy(scores, target)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--warm-start", type=Path, default=None,
                   help="TF-base joint_final.pt for μ warm-start (EGNN stays scratch)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pool-size", type=int, default=None,
                   help="limit training records (smoke runs)")
    p.add_argument("--ckpt-tag", type=str, default="tf_pocket_v3",
                   help="checkpoint basename")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["run"]["device"] = args.device
    device = cfg["run"].get("device", "cuda")
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    seed_everything(cfg["run"].get("seed", 42))

    lmdb_path = cfg["data"]["conditional"]
    geom_dir = cfg["data"]["pocket_geom"]
    print(f"[tf-pocket-v3] dataset={lmdb_path}")
    print(f"[tf-pocket-v3] pocket_geom={geom_dir}")

    ds_full = CrossDockedPocketGeomDataset(lmdb_path, pocket_geom_dir=geom_dir, split="train")
    ds = ds_full if args.pool_size is None else Subset(ds_full, range(min(args.pool_size, len(ds_full))))
    val_ds = CrossDockedPocketGeomDataset(lmdb_path, pocket_geom_dir=geom_dir, split="val")
    print(f"[tf-pocket-v3] n_train={len(ds)} n_val={len(val_ds)} phi_dim={ds_full.phi_dim}")

    t = cfg["training"]
    loader = DataLoader(
        ds,
        batch_size=int(t["batch_size"]),
        shuffle=True,
        num_workers=int(t.get("num_workers", 0)),
        collate_fn=collate_pocket_geom_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(t["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pocket_geom_batch,
    )

    encoder, head = build_heads(cfg, phi_dim=ds_full.phi_dim)
    encoder.to(device)
    head.to(device)
    if args.warm_start is not None:
        maybe_warm_start(head, encoder, args.warm_start)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"[tf-pocket-v3] EGNN params={n_enc/1e6:.3f}M  μ head params={n_head/1e6:.3f}M")

    phi_mean = torch.as_tensor(ds_full.phi_mean, dtype=torch.float32, device=device)
    phi_std = torch.as_tensor(ds_full.phi_std, dtype=torch.float32, device=device)

    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(
        params,
        lr=float(t["lr"]),
        weight_decay=float(t["weight_decay"]),
    )

    lw = t.get("loss_weights", {})
    lam_mu = float(lw.get("mu", 1.0))
    lam_contrast = float(lw.get("contrast", 0.5))
    beta_mu = float(t.get("mu_beta", 1.0))
    y_noise = float(t.get("mu_y_noise", 0.3))
    grad_clip = float(t.get("grad_clip", 5.0))
    log_every = int(t.get("log_every", 50))
    ckpt_every = int(t.get("ckpt_every", 500))
    eval_every = int(t.get("eval_every", 0))

    out_root = Path(cfg["run"]["out_dir"])
    log_dir = out_root / "logs" / cfg["run"]["name"]
    ckpt_dir = out_root / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"

    def log(d: dict) -> None:
        with metrics_path.open("a") as f:
            f.write(json.dumps(d) + "\n")

    def save(tag: str) -> Path:
        path = ckpt_dir / f"{tag}.pt"
        payload = {
            "state_dict": head.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "cfg": cfg,
            "phi_mean": phi_mean.detach().cpu(),
            "phi_std": phi_std.detach().cpu(),
            "pocket_dim": int(head.pocket_dim),
            "encoder_cfg": {
                "embed_dim": encoder.embed_dim,
                "n_layers": encoder.n_layers,
                "n_rbf": encoder.n_rbf,
                "cutoff_a": encoder.cutoff_a,
            },
        }
        torch.save(payload, path)
        return path

    def encode_pockets(batch: dict) -> torch.Tensor:
        coords = batch["pocket_coords"].to(device, non_blocking=True)
        aa = batch["pocket_aa"].to(device, non_blocking=True)
        mask = batch["pocket_mask"].to(device, non_blocking=True)
        return encoder(coords, aa, mask)

    @torch.no_grad()
    def evaluate() -> dict:
        encoder.eval(); head.eval()
        n = 0
        L_mu_sum = 0.0
        L_contrast_sum = 0.0
        acc_sum = 0.0
        for batch in val_loader:
            phi = batch["phi"].to(device)
            pocket = encode_pockets(batch)
            phi_z = (phi - phi_mean) / phi_std
            y = phi_z + y_noise * torch.randn_like(phi_z)
            B = phi.shape[0]

            mu_diag = head(y, pocket)
            y_exp = y.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
            p_exp = pocket.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
            mu_all = head(y_exp, p_exp).view(B, B, -1)

            mu_target = beta_mu * phi_z
            L_mu_sum += F.mse_loss(mu_diag, mu_target, reduction="sum").item()
            scores = (mu_all * phi_z.unsqueeze(1)).sum(dim=-1)
            tgt = torch.arange(B, device=device)
            L_contrast_sum += F.cross_entropy(scores, tgt, reduction="sum").item()
            acc_sum += (scores.argmax(dim=-1) == tgt).float().sum().item()
            n += B
        encoder.train(); head.train()
        return {
            "val/L_mu_per_sample": L_mu_sum / max(n, 1),
            "val/L_contrast_per_sample": L_contrast_sum / max(n, 1),
            "val/pocket_acc@B": acc_sum / max(n, 1),
            "val/n": n,
        }

    encoder.train(); head.train()
    step = 0
    t0 = time.time()
    loss_ema: float | None = None
    best_val = math.inf
    max_steps = args.max_steps
    ckpt_tag = str(args.ckpt_tag)

    for epoch in range(int(t.get("epochs", 100))):
        for batch in loader:
            phi = batch["phi"].to(device)
            pocket = encode_pockets(batch)
            phi_z = (phi - phi_mean) / phi_std
            y = phi_z + y_noise * torch.randn_like(phi_z)

            B = phi.shape[0]
            mu_diag = head(y, pocket)

            y_exp = y.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
            p_exp = pocket.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
            mu_all = head(y_exp, p_exp).view(B, B, -1)

            mu_target = beta_mu * phi_z
            L_mu = F.mse_loss(mu_diag, mu_target)
            L_contrast = contrastive_loss(mu_all, phi_z)
            loss = lam_mu * L_mu + lam_contrast * L_contrast

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()

            loss_val = loss.item()
            loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
            step += 1

            if step % log_every == 0:
                with torch.no_grad():
                    scores = (mu_all * phi_z.unsqueeze(1)).sum(dim=-1)
                    acc = (scores.argmax(dim=-1) == torch.arange(B, device=device)).float().mean().item()
                log({
                    "step": step, "epoch": epoch,
                    "loss": loss_val, "loss_ema": loss_ema,
                    "L_mu": L_mu.item(), "L_contrast": L_contrast.item(),
                    "pocket_acc@B": acc,
                    "lr": opt.param_groups[0]["lr"],
                    "samples_per_sec": (step * B) / max(time.time() - t0, 1e-6),
                })
            if step % ckpt_every == 0:
                save(f"{ckpt_tag}_last")
            if eval_every > 0 and step % eval_every == 0:
                val_m = evaluate()
                log({"step": step, "epoch": epoch, **val_m})
                if val_m["val/L_mu_per_sample"] < best_val:
                    best_val = val_m["val/L_mu_per_sample"]
                    save(f"{ckpt_tag}_best")
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    save(f"{ckpt_tag}_final")
    print(f"[tf-pocket-v3] training done: step={step} loss_ema={loss_ema}")

    # Laplace over μ head only (EGNN is outside the posterior, as in v1).
    try:
        n_y = min(4096, len(ds_full))
        idxs = torch.randperm(len(ds_full))[:n_y].tolist()
        yp_batches = []
        encoder.eval()
        for start in range(0, n_y, 256):
            sub = [ds_full[i] for i in idxs[start : start + 256]]
            batch = collate_pocket_geom_batch(sub)
            phi = batch["phi"].to(device)
            with torch.no_grad():
                pocket = encode_pockets(batch)
            phi_z = (phi - phi_mean) / phi_std
            yp_batches.append((phi_z.detach(), pocket.detach()))
        head.fit_laplace(iter(yp_batches))
        save(f"{ckpt_tag}_final")
        log({"step": step, "laplace_fitted": True, "n_y": n_y})
        print(f"[tf-pocket-v3] Laplace fitted on {n_y} ys")
    except Exception as e:
        log({"step": step, "laplace_fit_error": str(e)})
        print(f"[tf-pocket-v3] Laplace fit failed: {e}")


if __name__ == "__main__":
    main()
