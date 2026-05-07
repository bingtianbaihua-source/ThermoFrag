"""Fine-tune the TF-pocket variant: μ(y, pocket) on CrossDocked2020.

Protocol (docs/TF_POCKET.md):

1. Load the CrossDocked conditional LMDB + precomputed pocket embeddings.
2. Instantiate PocketConditionalChemicalPotentialHead. If a TF-base
   ``joint_final.pt`` is passed (``--warm-start``), copy μ(y)'s weights
   via ``warm_start_from_base`` so step-0 behavior matches TF-base.
3. Freeze QM + coupling (they stay at their Phase-3 values and are not
   touched here; the TF-pocket variant only changes the external field).
4. Optimize L = L_μ + λ_c · L_contrast where
     L_μ        = MSE(μ(y_i, p_i), β φ_z_i)          [identity per
                                                      thermodynamic id]
     L_contrast = CE(−μ(y_i, p_j) · φ_z_i,  j=i)     [pocket-aware]
   The contrastive term uses the other pockets in the batch as negatives,
   forcing μ to depend on the pocket.
5. Fit diagonal Laplace on μ's mean_head weights for C5 / OOD AUROC.
6. Save ``results/checkpoints/tf_pocket_mu.pt`` plus a metrics jsonl.

Usage::

    python scripts/train_pocket_variant.py --config configs/tf_pocket.yaml \\
        [--warm-start results/checkpoints/joint_final.pt] \\
        [--max-steps 10000]
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

from thermofrag.data.crossdocked import CrossDockedConditionalDataset, collate_pocket_batch
from thermofrag.potentials.external_field import (
    ChemicalPotentialHead,
    PocketConditionalChemicalPotentialHead,
)
from thermofrag.potentials.pocket_coupling import PocketLigandCoupling
from thermofrag.utils.config import load_config
from thermofrag.utils.seed import seed_everything


def build_mu_head(cfg: dict, phi_dim: int, pocket_dim: int) -> PocketConditionalChemicalPotentialHead:
    me = cfg["model"]["external_field"]
    if len(me["properties"]) != phi_dim:
        raise ValueError(
            f"config external_field.properties has {len(me['properties'])} entries but "
            f"dataset phi_dim={phi_dim}"
        )
    return PocketConditionalChemicalPotentialHead(
        n_properties=phi_dim,
        pocket_dim=pocket_dim,
        hidden=int(me["hidden"]),
    )


def maybe_warm_start(head: PocketConditionalChemicalPotentialHead, ckpt_path: Path) -> None:
    """Warm-start μ from a TF-base joint_final.pt checkpoint.

    The saved state dict has keys like ``mu.trunk.0.weight`` (wrapped in
    CouplingMuModule). We extract the ``mu.*`` subtree into a plain
    ChemicalPotentialHead and call ``head.warm_start_from_base``.
    """
    sd = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    mu_sd = {}
    for k, v in sd.items():
        if k.startswith("mu."):
            mu_sd[k[len("mu.") :]] = v
    if not mu_sd:
        raise RuntimeError(f"no mu.* keys in {ckpt_path}; is this a TF-base joint checkpoint?")
    base = ChemicalPotentialHead(
        n_properties=head.n_properties,
        hidden=head.hidden,
    )
    missing, unexpected = base.load_state_dict(mu_sd, strict=False)
    if missing:
        print(f"[warm-start] base head missing {missing}")
    if unexpected:
        print(f"[warm-start] base head unexpected {unexpected}")
    head.warm_start_from_base(base)
    print(f"[warm-start] copied μ weights from {ckpt_path}")


def contrastive_loss(mu_pred_ij: torch.Tensor, phi_z: torch.Tensor) -> torch.Tensor:
    """Symmetric softmax over pockets in a batch.

    ``mu_pred_ij`` is [B, B, K] where index (i, j) = μ(y_i, p_j).
    Scores are ``s_ij = μ(y_i, p_j) · φ_z_i``; correct class is j=i.
    """
    B = mu_pred_ij.shape[0]
    scores = (mu_pred_ij * phi_z.unsqueeze(1)).sum(dim=-1)  # [B, B]
    target = torch.arange(B, device=scores.device)
    return F.cross_entropy(scores, target)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--warm-start", type=Path, default=None,
                   help="TF-base joint_final.pt for μ warm-start")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pool-size", type=int, default=None,
                   help="limit training records (smoke runs)")
    p.add_argument("--enable-v-pocket", action="store_true",
                   help="Jointly train V^pocket(m, p) to regress the "
                        "crossdocked Vina dock labels. Adds a scalar "
                        "ligand-pocket coupling module; saved alongside μ.")
    p.add_argument("--v-pocket-weight", type=float, default=0.5,
                   help="loss weight on the V^pocket MSE term")
    p.add_argument("--v-pocket-hidden", type=int, default=64)
    p.add_argument("--v-pocket-mlp-hidden", type=int, default=128)
    p.add_argument("--ckpt-tag", type=str, default="tf_pocket",
                   help="checkpoint basename; defaults to tf_pocket. Set to "
                        "tf_pocket_v2 when training with --enable-v-pocket.")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg["run"]["device"] = args.device
    device = cfg["run"].get("device", "cuda")
    device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
    seed_everything(cfg["run"].get("seed", 42))

    lmdb_path = cfg["data"]["conditional"]
    pocket_dir = cfg["data"]["pocket_embeds"]
    print(f"[tf-pocket] dataset={lmdb_path}")
    print(f"[tf-pocket] pockets={pocket_dir}")

    ds_full = CrossDockedConditionalDataset(lmdb_path, pocket_embeds_dir=pocket_dir, split="train")
    ds = ds_full if args.pool_size is None else Subset(ds_full, range(min(args.pool_size, len(ds_full))))
    val_ds = CrossDockedConditionalDataset(lmdb_path, pocket_embeds_dir=pocket_dir, split="val")
    print(f"[tf-pocket] n_train={len(ds)} n_val={len(val_ds)} "
          f"phi_dim={ds_full.phi_dim} pocket_dim={ds_full.pocket_dim}")

    t = cfg["training"]
    loader = DataLoader(
        ds,
        batch_size=int(t["batch_size"]),
        shuffle=True,
        num_workers=int(t.get("num_workers", 0)),
        collate_fn=collate_pocket_batch,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(t["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pocket_batch,
    )

    head = build_mu_head(cfg, phi_dim=ds_full.phi_dim, pocket_dim=ds_full.pocket_dim).to(device)
    if args.warm_start is not None:
        maybe_warm_start(head, args.warm_start)

    n_params = sum(p.numel() for p in head.parameters())
    print(f"[tf-pocket] μ head params={n_params/1e6:.3f}M")

    phi_mean = torch.as_tensor(ds_full.phi_mean, dtype=torch.float32, device=device)
    phi_std = torch.as_tensor(ds_full.phi_std, dtype=torch.float32, device=device)

    # V^pocket scaffolding: optional, but when enabled we estimate (mean, std) of
    # vina_dock from the first ~5000 records so the regression head learns on a
    # standardized target. TargetDiff's preprocessed CrossDocked carries dock
    # scores directly on each record; bad rows fall through as NaN and are
    # skipped by the MSE mask.
    v_pocket: PocketLigandCoupling | None = None
    if args.enable_v_pocket:
        vd_vals: list[float] = []
        for i in range(min(5000, len(ds_full))):
            v = float(ds_full[i].get("vina_dock", float("nan")))
            if v == v and v != 0.0:  # NaN check; skip exact-zero placeholders
                vd_vals.append(v)
        if len(vd_vals) < 100:
            raise RuntimeError(
                f"vina_dock values present in only {len(vd_vals)}/5000 sampled records; "
                "the upstream LMDB does not appear to carry dock labels"
            )
        vd_arr = torch.tensor(vd_vals, dtype=torch.float32)
        vina_mean = float(vd_arr.mean().item())
        vina_scale = float(vd_arr.std().clamp_min(1e-3).item())
        v_pocket = PocketLigandCoupling(
            phi_dim=ds_full.phi_dim,
            pocket_dim=ds_full.pocket_dim,
            pocket_hidden=int(args.v_pocket_hidden),
            mlp_hidden=int(args.v_pocket_mlp_hidden),
            vina_scale=vina_scale,
            vina_mean=vina_mean,
        ).to(device)
        print(
            f"[tf-pocket] V^pocket enabled: vina_mean={vina_mean:.3f} "
            f"vina_scale={vina_scale:.3f}  params="
            f"{sum(p.numel() for p in v_pocket.parameters())/1e6:.3f}M"
        )

    params = list(head.parameters())
    if v_pocket is not None:
        params += list(v_pocket.parameters())
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
            "cfg": cfg,
            "phi_mean": phi_mean.detach().cpu(),
            "phi_std": phi_std.detach().cpu(),
            "pocket_dim": ds_full.pocket_dim,
        }
        if v_pocket is not None:
            payload["v_pocket_state_dict"] = v_pocket.state_dict()
            payload["v_pocket_meta"] = {
                "phi_dim": int(ds_full.phi_dim),
                "pocket_dim": int(ds_full.pocket_dim),
                "pocket_hidden": int(args.v_pocket_hidden),
                "mlp_hidden": int(args.v_pocket_mlp_hidden),
                "vina_mean": float(v_pocket.vina_mean),
                "vina_scale": float(v_pocket.vina_scale),
            }
        torch.save(payload, path)
        return path

    @torch.no_grad()
    def evaluate() -> dict:
        head.eval()
        if v_pocket is not None:
            v_pocket.eval()
        n = 0
        L_mu_sum = 0.0
        L_contrast_sum = 0.0
        acc_sum = 0.0
        L_vp_sum = 0.0
        n_vp = 0
        vp_pred_sum = 0.0
        vp_tgt_sum = 0.0
        vp_pred_sq = 0.0
        vp_tgt_sq = 0.0
        vp_cross = 0.0
        for batch in val_loader:
            phi = batch["phi"].to(device)
            pocket = batch["pocket"].to(device)
            phi_z = (phi - phi_mean) / phi_std
            y = phi_z + y_noise * torch.randn_like(phi_z)

            mu_diag = head(y, pocket)
            B = phi.shape[0]
            y_exp = y.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
            p_exp = pocket.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
            mu_all = head(y_exp, p_exp).view(B, B, -1)
            mu_target = beta_mu * phi_z
            L_mu = F.mse_loss(mu_diag, mu_target, reduction="sum").item()
            scores = (mu_all * phi_z.unsqueeze(1)).sum(dim=-1)
            tgt = torch.arange(B, device=device)
            L_c = F.cross_entropy(scores, tgt, reduction="sum").item()
            acc = (scores.argmax(dim=-1) == tgt).float().sum().item()
            L_mu_sum += L_mu
            L_contrast_sum += L_c
            acc_sum += acc
            n += B

            if v_pocket is not None:
                vd = batch["vina_dock"].to(device)
                mask = torch.isfinite(vd) & (vd != 0.0)
                if int(mask.sum().item()) > 0:
                    phi_z_m = phi_z[mask]
                    pocket_m = pocket[mask]
                    vd_m = vd[mask]
                    vp_std_pred = v_pocket.forward_standardized(phi_z_m, pocket_m)
                    vp_kcal = vp_std_pred * v_pocket.vina_scale + v_pocket.vina_mean
                    vd_std = (vd_m - v_pocket.vina_mean) / v_pocket.vina_scale
                    L_vp_sum += F.mse_loss(vp_std_pred, vd_std, reduction="sum").item()
                    n_vp += int(mask.sum().item())
                    vp_pred_sum += float(vp_kcal.sum().item())
                    vp_tgt_sum += float(vd_m.sum().item())
                    vp_pred_sq += float((vp_kcal * vp_kcal).sum().item())
                    vp_tgt_sq += float((vd_m * vd_m).sum().item())
                    vp_cross += float((vp_kcal * vd_m).sum().item())
        head.train()
        if v_pocket is not None:
            v_pocket.train()
        out = {
            "val/L_mu_per_sample": L_mu_sum / max(n, 1),
            "val/L_contrast_per_sample": L_contrast_sum / max(n, 1),
            "val/pocket_acc@B": acc_sum / max(n, 1),
            "val/n": n,
        }
        if v_pocket is not None and n_vp > 0:
            out["val/L_vpocket_std_per_sample"] = L_vp_sum / n_vp
            out["val/n_vpocket"] = n_vp
            mean_p = vp_pred_sum / n_vp
            mean_t = vp_tgt_sum / n_vp
            var_p = max(vp_pred_sq / n_vp - mean_p * mean_p, 1e-12)
            var_t = max(vp_tgt_sq / n_vp - mean_t * mean_t, 1e-12)
            cov = vp_cross / n_vp - mean_p * mean_t
            out["val/vpocket_pearson"] = cov / ((var_p * var_t) ** 0.5)
            out["val/vpocket_mae_kcal"] = (L_vp_sum / n_vp) ** 0.5 * float(v_pocket.vina_scale)
        return out

    head.train()
    if v_pocket is not None:
        v_pocket.train()
    step = 0
    t0 = time.time()
    loss_ema: float | None = None
    best_val = math.inf
    max_steps = args.max_steps
    ckpt_tag = str(args.ckpt_tag)
    lam_vp = float(args.v_pocket_weight) if v_pocket is not None else 0.0

    for epoch in range(int(t.get("epochs", 100))):
        for batch in loader:
            phi = batch["phi"].to(device)
            pocket = batch["pocket"].to(device)
            phi_z = (phi - phi_mean) / phi_std
            y = phi_z + y_noise * torch.randn_like(phi_z)

            B = phi.shape[0]
            mu_diag = head(y, pocket)  # [B, K]

            y_exp = y.unsqueeze(1).expand(B, B, -1).reshape(B * B, -1)
            p_exp = pocket.unsqueeze(0).expand(B, B, -1).reshape(B * B, -1)
            mu_all = head(y_exp, p_exp).view(B, B, -1)

            mu_target = beta_mu * phi_z
            L_mu = F.mse_loss(mu_diag, mu_target)
            L_contrast = contrastive_loss(mu_all, phi_z)
            loss = lam_mu * L_mu + lam_contrast * L_contrast

            L_vp_val: float | None = None
            if v_pocket is not None:
                vd = batch["vina_dock"].to(device)
                mask = torch.isfinite(vd) & (vd != 0.0)
                if int(mask.sum().item()) >= 2:
                    phi_z_m = phi_z[mask]
                    pocket_m = pocket[mask]
                    vd_m = vd[mask]
                    vp_std_pred = v_pocket.forward_standardized(phi_z_m, pocket_m)
                    vd_std = (vd_m - v_pocket.vina_mean) / v_pocket.vina_scale
                    L_vp = F.mse_loss(vp_std_pred, vd_std)
                    loss = loss + lam_vp * L_vp
                    L_vp_val = float(L_vp.item())

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
                rec = {
                    "step": step, "epoch": epoch,
                    "loss": loss_val, "loss_ema": loss_ema,
                    "L_mu": L_mu.item(), "L_contrast": L_contrast.item(),
                    "pocket_acc@B": acc,
                    "lr": opt.param_groups[0]["lr"],
                    "samples_per_sec": (step * B) / max(time.time() - t0, 1e-6),
                }
                if L_vp_val is not None:
                    rec["L_vpocket_std"] = L_vp_val
                log(rec)
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
    print(f"[tf-pocket] training done: step={step} loss_ema={loss_ema}")

    # Fit Laplace over the final μ head — use a bounded subset of the training pool.
    try:
        n_y = min(4096, len(ds_full))
        idxs = torch.randperm(len(ds_full))[:n_y].tolist()
        yp_batches = []
        for start in range(0, n_y, 256):
            sub = [ds_full[i] for i in idxs[start : start + 256]]
            phi = torch.stack([s["phi"] for s in sub], dim=0).to(device)
            pocket = torch.stack([s["pocket"] for s in sub], dim=0).to(device)
            phi_z = (phi - phi_mean) / phi_std
            yp_batches.append((phi_z, pocket))
        head.fit_laplace(iter(yp_batches))
        save(f"{ckpt_tag}_final")  # re-save with Laplace buffer populated
        log({"step": step, "laplace_fitted": True, "n_y": n_y})
        print(f"[tf-pocket] Laplace fitted on {n_y} ys")
    except Exception as e:
        log({"step": step, "laplace_fit_error": str(e)})
        print(f"[tf-pocket] Laplace fit failed: {e}")


if __name__ == "__main__":
    main()
