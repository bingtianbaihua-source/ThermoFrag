"""TF-pocket-v4: fine-tune v2's V^pocket on sampler-distributed phase4 data.

Design rationale (2026-04-24 diagnostic): v2's V^pocket(m, p), trained on
CrossDocked2020 with CD-standardized phi_z, reaches Pearson 0.83 on its own
val split. But at sample time the sampler feeds ChEMBL-standardized phi_z,
which is OOD for v2, and the effective Pearson on 1442 phase4 samples drops
to 0.343 pooled / 0.417 per-target-mean. That is non-zero but too weak to
reliably bias the MH kernel toward good binders. v4 fine-tunes v2 on the
phase4 pool -- drawn from exactly the distribution the sampler explores at
inference -- so the predictor is re-aligned to the sampler manifold.

Protocol
--------
* Load v2 checkpoint (``results/checkpoints/tf_pocket_v2_final.pt``).
* Build a (phi_z, pocket, vina) dataset from existing phase4 artefacts:
  - phi_raw(m) = sum_i frag_phi[frag_id_i]   (matches sampler's
    _phi_of_batch convention)
  - phi_z = (phi_raw - ChEMBL phi_mean) / ChEMBL phi_std
  - pocket = data/processed/pocket_embeds/litpcba/<target>.npy (ESM-2 1280d)
  - vina = ground-truth AutoDock Vina score from phase4/vina/<target>.parquet
* 5-fold cross-target CV (15 targets, 3 per val fold). For each fold, save
  a checkpoint that was *never* trained on val targets.
* Loss: Huber on (vina - train_vina_mean) / train_vina_std.
* Early stop on val Pearson.

Output
------
* results/checkpoints/tf_v_vina_v4_fold{0..4}.pt -- each holds state_dict,
  vina_mean/std calibration, and list of held-out targets.
* results/eval/v4_fold_pearson.csv -- per-fold val Pearson + per-target
  Pearson for the fold's val targets.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from thermofrag.data.zinc_fragments import ZINCFragmentDataset
from thermofrag.potentials.pocket_coupling import PocketLigandCoupling
from thermofrag.sampling.conditional_mh import build_frag_phi_table


LITPCBA = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1", "GBA", "IDH1",
    "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]


@dataclass
class Phase4Sample:
    target: str
    phi_z: np.ndarray      # (K,)
    pocket: np.ndarray     # (D,)
    vina: float


def _phi_from_frag_ids(frag_ids: list[int], frag_phi: np.ndarray) -> np.ndarray:
    if not frag_ids:
        return np.zeros(frag_phi.shape[1], dtype=np.float32)
    return frag_phi[frag_ids].sum(axis=0)


def load_phase4_dataset(
    samples_dir: Path,
    vina_dir: Path,
    pocket_embeds: Path,
    frag_phi: np.ndarray,
    phi_mean: np.ndarray,
    phi_std: np.ndarray,
) -> list[Phase4Sample]:
    out: list[Phase4Sample] = []
    for target in LITPCBA:
        samp_pkl = samples_dir / f"{target}.pkl"
        vina_pq = vina_dir / f"{target}.parquet"
        pkt_npy = pocket_embeds / f"{target}.npy"
        if not (samp_pkl.exists() and vina_pq.exists() and pkt_npy.exists()):
            print(f"[v4] skip {target} (missing files)")
            continue
        with open(samp_pkl, "rb") as f:
            d = pickle.load(f)
        chain_samples = d["samples"]
        vdf = pd.read_parquet(vina_pq)
        vdf = vdf[vdf.status == "ok"][["chain_idx", "vina_score"]]
        pocket = np.load(pkt_npy).astype(np.float32)
        for _, row in vdf.iterrows():
            ci = int(row.chain_idx)
            fi = chain_samples[ci]["frag_id"]
            phi_raw = _phi_from_frag_ids(fi, frag_phi)
            phi_z = (phi_raw - phi_mean) / phi_std
            out.append(
                Phase4Sample(target=target, phi_z=phi_z, pocket=pocket,
                             vina=float(row.vina_score))
            )
    print(f"[v4] loaded {len(out)} total samples across {len(set(s.target for s in out))} targets")
    return out


def make_folds(n_folds: int, seed: int = 0) -> list[list[str]]:
    """Return ``n_folds`` lists of target names (val targets per fold)."""
    rng = np.random.default_rng(seed)
    order = list(LITPCBA)
    rng.shuffle(order)
    # 15 / 5 = 3 per fold
    folds = []
    per_fold = len(order) // n_folds
    for i in range(n_folds):
        start, end = i * per_fold, (i + 1) * per_fold
        folds.append(order[start:end])
    # If division not exact, any leftovers go into last fold.
    remainder = order[n_folds * per_fold:]
    if remainder:
        folds[-1].extend(remainder)
    return folds


def _make_tensors(
    samples: list[Phase4Sample], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi = torch.from_numpy(np.stack([s.phi_z for s in samples], axis=0)).float().to(device)
    pkt = torch.from_numpy(np.stack([s.pocket for s in samples], axis=0)).float().to(device)
    v = torch.from_numpy(np.asarray([s.vina for s in samples], dtype=np.float32)).to(device)
    return phi, pkt, v


def load_v_pocket_v2(ckpt: Path, device: str) -> PocketLigandCoupling:
    blob = torch.load(str(ckpt), map_location=device, weights_only=False)
    sd = blob["v_pocket_state_dict"]
    meta = blob.get("v_pocket_meta", {})
    pocket_hidden = int(meta.get("pocket_hidden", 64))
    mlp_hidden = int(meta.get("mlp_hidden", 128))
    pocket_dim = int(sd["pocket_proj.0.weight"].shape[1])
    phi_dim = int(sd["mlp.0.weight"].shape[1] - pocket_hidden)
    head = PocketLigandCoupling(
        phi_dim=phi_dim, pocket_dim=pocket_dim,
        pocket_hidden=pocket_hidden, mlp_hidden=mlp_hidden,
    )
    head.load_state_dict(sd, strict=True)
    head.to(device)
    return head


def train_one_fold(
    fold_idx: int,
    val_targets: list[str],
    samples: list[Phase4Sample],
    v2_ckpt: Path,
    device: str,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    out_dir: Path,
) -> dict:
    val_tset = set(val_targets)
    train_s = [s for s in samples if s.target not in val_tset]
    val_s = [s for s in samples if s.target in val_tset]
    print(f"[v4] fold {fold_idx}: val targets={val_targets}  train_n={len(train_s)} val_n={len(val_s)}")
    if len(train_s) < 64 or len(val_s) < 10:
        raise RuntimeError(f"fold {fold_idx} pool too small")

    head = load_v_pocket_v2(v2_ckpt, device)
    # Recalibrate vina_mean/std to train fold (v2's was CrossDocked's).
    train_vina = np.asarray([s.vina for s in train_s], dtype=np.float32)
    tmean, tstd = float(train_vina.mean()), float(train_vina.std() + 1e-6)
    head.set_calibration(tmean, tstd)

    phi_tr, pkt_tr, v_tr = _make_tensors(train_s, device)
    phi_va, pkt_va, v_va = _make_tensors(val_s, device)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()

    # Baseline Pearson (v2 out-of-the-box, no fine-tune) for context.
    head.eval()
    with torch.no_grad():
        pred0 = head(phi_va, pkt_va).cpu().numpy()
    base_pearson = float(np.corrcoef(v_va.cpu().numpy(), pred0)[0, 1])
    print(f"[v4]   fold {fold_idx} baseline val Pearson (v2 pre-ft): {base_pearson:+.3f}")

    best_pearson = base_pearson
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    best_epoch = 0

    ds_tr = TensorDataset(phi_tr, pkt_tr, v_tr)
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(ds_tr), device=device)
        losses: list[float] = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            phi_b, pkt_b, v_b = phi_tr[idx], pkt_tr[idx], v_tr[idx]
            pred_std = head.forward_standardized(phi_b, pkt_b)
            target_std = (v_b - tmean) / tstd
            loss = loss_fn(pred_std, target_std)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        head.eval()
        with torch.no_grad():
            pred = head(phi_va, pkt_va).cpu().numpy()
        pr = float(np.corrcoef(v_va.cpu().numpy(), pred)[0, 1])
        mean_loss = float(np.mean(losses))
        tag = ""
        if pr > best_pearson:
            best_pearson = pr
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            best_epoch = ep
            tag = "  *"
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[v4]   fold {fold_idx} ep{ep:3d}  loss={mean_loss:.4f}  "
                  f"val Pearson={pr:+.3f}{tag}")

    # Restore best.
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred_best = head(phi_va, pkt_va).cpu().numpy()
    val_truth = v_va.cpu().numpy()

    # Per-val-target breakdown
    per_target = []
    for t in val_targets:
        mask = np.asarray([s.target == t for s in val_s], dtype=bool)
        if mask.sum() < 3:
            continue
        pr_t = float(np.corrcoef(val_truth[mask], pred_best[mask])[0, 1])
        per_target.append({"target": t, "n": int(mask.sum()), "pearson": pr_t})
        print(f"[v4]   fold {fold_idx} val {t:12s}  n={int(mask.sum()):3d}  "
              f"Pearson={pr_t:+.3f}")

    # Persist fold checkpoint.
    out_path = out_dir / f"tf_v_vina_v4_fold{fold_idx}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "v_pocket_state_dict": head.state_dict(),
        "v_pocket_meta": {
            "pocket_hidden": head.pocket_hidden,
            "mlp_hidden": head.mlp_hidden,
            "phi_dim": head.phi_dim,
            "pocket_dim": head.pocket_dim,
        },
        "fold_idx": fold_idx,
        "val_targets": val_targets,
        "best_epoch": best_epoch,
        "baseline_pearson": base_pearson,
        "best_pearson": best_pearson,
        "per_target_pearson": per_target,
        "vina_train_mean": tmean,
        "vina_train_std": tstd,
    }, str(out_path))
    print(f"[v4]   fold {fold_idx} saved -> {out_path}  "
          f"best_pearson={best_pearson:+.3f} at ep{best_epoch}")

    return {
        "fold_idx": fold_idx,
        "val_targets": val_targets,
        "train_n": len(train_s),
        "val_n": len(val_s),
        "baseline_pearson": base_pearson,
        "best_pearson": best_pearson,
        "best_epoch": best_epoch,
        "per_target": per_target,
        "ckpt": str(out_path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2-ckpt", type=Path,
                   default=Path("results/checkpoints/tf_pocket_v2_final.pt"))
    p.add_argument("--samples-dir", type=Path,
                   default=Path("results/eval/phase4/samples"))
    p.add_argument("--vina-dir", type=Path,
                   default=Path("results/eval/phase4/vina"))
    p.add_argument("--pocket-embeds", type=Path,
                   default=Path("data/processed/pocket_embeds/litpcba"))
    p.add_argument("--library", type=Path,
                   default=Path("data/processed/fragment_library.parquet"))
    p.add_argument("--chembl-lmdb", type=Path,
                   default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/checkpoints"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fold-seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[v4] device={device}")

    # Properties + ChEMBL phi_mean/std (sampler-consistent).
    chembl = ZINCFragmentDataset(str(args.chembl_lmdb))
    properties = chembl.phi_properties
    phi_mean = np.asarray(chembl.meta["phi_mean"], dtype=np.float32)
    phi_std = np.asarray(chembl.meta["phi_std"], dtype=np.float32)
    print(f"[v4] properties={properties}")
    print(f"[v4] phi_mean (ChEMBL)={phi_mean.round(3).tolist()}")

    print("[v4] building frag_phi table...")
    frag_phi = build_frag_phi_table(args.library, properties)

    print("[v4] loading phase4 samples...")
    samples = load_phase4_dataset(
        args.samples_dir, args.vina_dir, args.pocket_embeds,
        frag_phi, phi_mean, phi_std,
    )

    folds = make_folds(args.n_folds, seed=args.fold_seed)
    print(f"[v4] folds: {folds}")

    rows = []
    for fi, val_targets in enumerate(folds):
        row = train_one_fold(
            fi, val_targets, samples, args.v2_ckpt,
            device=device, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, weight_decay=args.weight_decay,
            out_dir=args.out_dir,
        )
        rows.append(row)

    # Aggregate summary.
    mean_baseline = float(np.mean([r["baseline_pearson"] for r in rows]))
    mean_best = float(np.mean([r["best_pearson"] for r in rows]))
    print()
    print(f"[v4] ====== 5-fold summary ======")
    print(f"[v4] mean baseline val Pearson (v2 pre-ft):   {mean_baseline:+.3f}")
    print(f"[v4] mean best     val Pearson (v4 post-ft):  {mean_best:+.3f}")
    print(f"[v4] delta:                                    {mean_best - mean_baseline:+.3f}")

    # CSV
    flat = []
    for r in rows:
        for pt in r["per_target"]:
            flat.append({
                "fold": r["fold_idx"],
                "val_targets": ",".join(r["val_targets"]),
                "target": pt["target"],
                "n": pt["n"],
                "pearson": pt["pearson"],
                "fold_baseline_pearson": r["baseline_pearson"],
                "fold_best_pearson": r["best_pearson"],
            })
    out_csv = Path("results/eval/v4_fold_pearson.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(out_csv, index=False)
    print(f"[v4] per-target CSV -> {out_csv}")

    # Gate reporting
    gate = 0.40
    print(f"[v4] gate: mean val Pearson > {gate} => {mean_best > gate}")


if __name__ == "__main__":
    main()
