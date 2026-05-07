#!/usr/bin/env python
"""μ-matrix cross-validation driver (Phase-7, Task 7).

Spec: ``docs/validation/07_mu_matrix_crossval.md``.

Sub-tasks (selectable via ``--subtask``):

  chembl_corr  Pearson + Spearman 8x8 of φ on raw ChEMBL conditional LMDB,
               plus partial-corr(HBA, HBD | controls). Decisive test for
               Cat-E HBA-HBD trade-off (see MU_MATRIX_FINDINGS.md).
  laplace      Sample Laplace last-layer posterior of trained μ head and
               compute per-entry M[i,j] CIs.
  drugbank     Re-run partial-corr(HBA, HBD | controls) on independent
               population (DrugBank approved-drug SMILES).

Sub-task ``seed_stability`` retrains the μ head from scratch across
multiple random seeds using the same implemented L_mu target as Phase 3:
``mu(phi_z + noise) -> phi_z``. ``feature_swap`` (Tier-3) remains
deferred because it requires regenerating the conditional LMDB with
alternative property definitions.

Outputs land under ``results/eval/phase7/mu_crossval/``. Each sub-task
appends a block to ``AGGREGATE/07_mu_crossval_summary.json``.

Conventions follow ``docs/validation/00_shared_infrastructure.md``:

* ``--seed`` defaults 42; manifest.json records git SHA + args.
* Idempotent: re-runs reuse parquet outputs unless ``--force``.
* All paths absolute or anchored to CLI flags.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PHI_ORDER = ["logP", "qed", "sa", "tpsa", "mw", "hba", "hbd", "rotb"]
NOISE_FLOOR = 0.05  # |M| > 0.05 counts as "strong" off-diagonal


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, args: argparse.Namespace, inputs: dict[str, Path], wall: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "git_sha": _git_sha(Path(args.repo_root)),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.executable,
        "wall_seconds": wall,
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": {name: {"path": str(p), "sha256": _sha256(p) if p.exists() and p.is_file() else None}
                   for name, p in inputs.items()},
    }
    out_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2))


def load_M_baseline(c2_report: Path) -> np.ndarray:
    rep = json.loads(c2_report.read_text())
    return np.asarray(rep["response_matrix"])


def load_phi_from_lmdb(lmdb_path: Path, max_records: int | None = None) -> np.ndarray:
    """Load φ vectors from a single-file ChEMBL conditional LMDB.

    Returns array of shape [N, 8] in PHI_ORDER. The LMDB is the one built
    by ``scripts/build_conditional_lmdb.py``; each record is a dict with
    keys including ``phi`` (a length-8 list) and ``phi_properties`` (the
    axis-name list, expected to equal PHI_ORDER).
    """
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    rows = []
    with env.begin() as txn:
        cur = txn.cursor()
        for i, (_, v) in enumerate(cur):
            rec = pickle.loads(v)
            phi = rec.get("phi")
            if phi is None:
                continue
            if rec.get("phi_properties") and rec["phi_properties"] != PHI_ORDER:
                raise RuntimeError(f"phi_properties mismatch: got {rec['phi_properties']}")
            rows.append(phi)
            if max_records is not None and len(rows) >= max_records:
                break
    env.close()
    return np.asarray(rows, dtype=np.float64)


def load_smiles_phi_from_lmdb(lmdb_path: Path, max_records: int | None = None) -> tuple[list[str], np.ndarray]:
    """Load SMILES and φ vectors from the conditional LMDB."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    smiles: list[str] = []
    rows = []
    with env.begin() as txn:
        cur = txn.cursor()
        for _, v in cur:
            rec = pickle.loads(v)
            phi = rec.get("phi")
            smi = rec.get("smiles")
            if phi is None or smi is None:
                continue
            if rec.get("phi_properties") and rec["phi_properties"] != PHI_ORDER:
                raise RuntimeError(f"phi_properties mismatch: got {rec['phi_properties']}")
            smiles.append(str(smi))
            rows.append(phi)
            if max_records is not None and len(rows) >= max_records:
                break
    env.close()
    return smiles, np.asarray(rows, dtype=np.float32)


def load_phi_meta_from_lmdb(lmdb_path: Path) -> dict:
    """Return the LMDB metadata block written by the conditional preprocessor."""
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, subdir=False)
    try:
        with env.begin() as txn:
            raw = txn.get(b"__meta__")
            if raw is None:
                raise RuntimeError(f"{lmdb_path} has no __meta__ record")
            meta = pickle.loads(raw)
    finally:
        env.close()
    return meta


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> tuple[float, float]:
    """Pearson partial-corr of x and y after linearly regressing out controls.

    Returns (r, p_value). Both x and y must be 1-D length-N; controls is
    [N, K]. Implementation: residualize x and y on controls separately,
    Pearson the residuals.
    """
    # Add intercept column
    A = np.column_stack([np.ones(len(controls)), controls])
    # Least-squares for x ~ A, y ~ A
    bx, *_ = np.linalg.lstsq(A, x, rcond=None)
    by, *_ = np.linalg.lstsq(A, y, rcond=None)
    rx = x - A @ bx
    ry = y - A @ by
    r, p = pearsonr(rx, ry)
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Sub-task: chembl_corr (Tier-1 decisive)
# ---------------------------------------------------------------------------


def run_chembl_corr(args: argparse.Namespace, out_root: Path) -> dict:
    out_dir = out_root
    sub_dir = out_dir
    sub_dir.mkdir(parents=True, exist_ok=True)
    corr_path = sub_dir / "chembl_corr_matrix.parquet"
    pcorr_path = sub_dir / "chembl_partial_corr.parquet"

    if (corr_path.exists() and pcorr_path.exists()) and not args.force:
        print(f"[skip] {corr_path} and {pcorr_path} already exist; use --force to rebuild")
        corr_df = pd.read_parquet(corr_path)
        pcorr_df = pd.read_parquet(pcorr_path)
        return _summarize_chembl_corr(corr_df, pcorr_df, args)

    print(f"[load] reading φ vectors from {args.lmdb}")
    phi = load_phi_from_lmdb(args.lmdb, max_records=args.max_records)
    print(f"[load] phi shape = {phi.shape}")

    # 1. Pearson + Spearman 8×8
    K = phi.shape[1]
    pearson = np.zeros((K, K))
    spearman = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            pearson[i, j], _ = pearsonr(phi[:, i], phi[:, j])
            spearman[i, j], _ = spearmanr(phi[:, i], phi[:, j])

    rows = []
    for i, name_i in enumerate(PHI_ORDER):
        for j, name_j in enumerate(PHI_ORDER):
            rows.append({
                "i": i, "j": j, "pushed": name_i, "responded": name_j,
                "pearson": pearson[i, j], "spearman": spearman[i, j],
            })
    corr_df = pd.DataFrame(rows)
    corr_df.to_parquet(corr_path, index=False)
    print(f"[write] {corr_path}")

    # 2. Partial correlations for the candidate-novel HBA↔HBD finding (Cat E).
    hba = PHI_ORDER.index("hba")
    hbd = PHI_ORDER.index("hbd")
    tpsa = PHI_ORDER.index("tpsa")
    mw = PHI_ORDER.index("mw")
    logp = PHI_ORDER.index("logP")
    sa = PHI_ORDER.index("sa")

    pc_specs = [
        ("hba_hbd", "controls=tpsa,mw", phi[:, hba], phi[:, hbd], phi[:, [tpsa, mw]]),
        ("hba_hbd", "controls=tpsa", phi[:, hba], phi[:, hbd], phi[:, [tpsa]]),
        ("hba_hbd", "controls=mw", phi[:, hba], phi[:, hbd], phi[:, [mw]]),
        ("hba_hbd", "controls=tpsa,mw,logp,sa", phi[:, hba], phi[:, hbd],
         phi[:, [tpsa, mw, logp, sa]]),
        # marginal corr for reference (controls = empty)
        ("hba_hbd", "marginal", phi[:, hba], phi[:, hbd], np.zeros((len(phi), 0))),
    ]

    pc_rows = []
    for pair, ctrl_name, x, y, ctrl in pc_specs:
        if ctrl.shape[1] == 0:
            r, _ = pearsonr(x, y)
            p = float(_pearson_p(r, len(x)))
        else:
            r, p = partial_corr(x, y, ctrl)
        pc_rows.append({"pair": pair, "controls": ctrl_name, "r": float(r), "p": float(p),
                        "n": int(len(x))})
        print(f"[partial_corr] {pair} | {ctrl_name:32s}  r={r:+.4f}  p={p:.2e}")

    # Also: Cat-F orthogonality of RotB. Marginal + partial vs each compositional axis.
    rotb = PHI_ORDER.index("rotb")
    for axis_name, axis_idx in [("logP", logp), ("hba", hba), ("hbd", hbd), ("tpsa", tpsa)]:
        # Partial corr of RotB and axis controlling for MW (size confound)
        ctrl = phi[:, [mw]]
        r, p = partial_corr(phi[:, rotb], phi[:, axis_idx], ctrl)
        pc_rows.append({"pair": f"rotb_{axis_name}", "controls": "controls=mw",
                        "r": float(r), "p": float(p), "n": int(len(phi))})

    pcorr_df = pd.DataFrame(pc_rows)
    pcorr_df.to_parquet(pcorr_path, index=False)
    print(f"[write] {pcorr_path}")

    return _summarize_chembl_corr(corr_df, pcorr_df, args)


def _pearson_p(r: float, n: int) -> float:
    """Two-sided p-value for Pearson r with n samples (no ties)."""
    from scipy.stats import t as student_t
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    tstat = r * np.sqrt(df / (1 - r * r))
    return float(2 * (1 - student_t.cdf(abs(tstat), df=df)))


def _summarize_chembl_corr(corr_df: pd.DataFrame, pcorr_df: pd.DataFrame, args: argparse.Namespace) -> dict:
    K = len(PHI_ORDER)
    M = load_M_baseline(Path(args.c2_report))
    # Reshape pearson/spearman back into 8×8 for sign comparison
    P = np.zeros((K, K))
    for _, row in corr_df.iterrows():
        P[int(row["i"]), int(row["j"])] = row["pearson"]

    off = ~np.eye(K, dtype=bool)
    strong = np.abs(M) > NOISE_FLOOR
    strong_off = strong & off
    n_strong_off = int(strong_off.sum())
    sign_match = (np.sign(M) == np.sign(P)) & strong_off
    n_sign_match = int(sign_match.sum())

    # Decisive HBA-HBD partial-corr test
    hba_hbd_partial_main = pcorr_df.query(
        "pair == 'hba_hbd' and controls == 'controls=tpsa,mw'"
    ).iloc[0]
    hba_hbd_marginal = pcorr_df.query(
        "pair == 'hba_hbd' and controls == 'marginal'"
    ).iloc[0]

    print()
    print("=" * 70)
    print("Sub-task 1 summary")
    print("=" * 70)
    print(f"  Strong off-diag entries (|M|>{NOISE_FLOOR}): {n_strong_off}")
    print(f"  Sign-agreement vs ChEMBL Pearson:           {n_sign_match}/{n_strong_off}")
    print(f"  Marginal corr(HBA, HBD):                    r={hba_hbd_marginal['r']:+.4f}")
    print(f"  Partial   corr(HBA, HBD | TPSA, MW):        r={hba_hbd_partial_main['r']:+.4f}")
    print(f"  Cat-E DECISIVE test (partial < 0):          {'PASS' if hba_hbd_partial_main['r'] < 0 else 'FAIL'}")
    print()

    summary = {
        "subtask": "chembl_corr",
        "n_strong_off": n_strong_off,
        "n_sign_match": n_sign_match,
        "sign_match_threshold": 18,
        "sign_match_pass": bool(n_sign_match >= 18),
        "hba_hbd_marginal_r": float(hba_hbd_marginal["r"]),
        "hba_hbd_partial_tpsa_mw_r": float(hba_hbd_partial_main["r"]),
        "hba_hbd_partial_tpsa_mw_p": float(hba_hbd_partial_main["p"]),
        "cat_e_decisive_pass": bool(hba_hbd_partial_main["r"] < 0),
        "n_records": int(corr_df["pearson"].count()),
    }
    return summary


# ---------------------------------------------------------------------------
# Sub-task: laplace (Tier-1 cheap)
# ---------------------------------------------------------------------------


def run_laplace(args: argparse.Namespace, out_root: Path) -> dict:
    """Sample Laplace posterior of μ head and compute per-entry M CIs.

    Strategy: load ``ChemicalPotentialHead`` from the joint_final.pt
    checkpoint, extract its ``laplace_diag`` posterior variance over
    ``mean_head`` weights, then sample N draws of mean_head W and compute
    M[i, j] for each draw.
    """
    import torch
    sys.path.insert(0, str(Path(args.repo_root) / "src"))
    from thermofrag.potentials.external_field import ChemicalPotentialHead  # type: ignore

    out_dir = out_root / "laplace_posterior"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "M_with_ci.parquet"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} already exists; use --force to rebuild")
        return _summarize_laplace(pd.read_parquet(out_path), args)

    device = "cpu"  # μ head is tiny; no need for GPU
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    # μ head is stored under prefix `mu.` in joint_final.pt (see trainer.py)
    mu_keys = {k[len("mu."):]: v for k, v in state_dict.items() if k.startswith("mu.")}
    if not mu_keys:
        # Fallback: some older configs used `chempot.`
        mu_keys = {k[len("chempot."):]: v for k, v in state_dict.items()
                   if k.startswith("chempot.")}
    if not mu_keys:
        raise RuntimeError("no 'mu.*' or 'chempot.*' keys in checkpoint")

    head = ChemicalPotentialHead(n_properties=8, hidden=256).to(device)
    missing, unexpected = head.load_state_dict(mu_keys, strict=False)
    if missing:
        print(f"[warn] missing keys when loading μ head: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    head.eval()

    laplace_diag = head.laplace_diag.detach().cpu().numpy().reshape(8, 256)  # [P, H]
    print(f"[laplace] mean post_var = {laplace_diag.mean():.3e}, "
          f"min = {laplace_diag.min():.3e}, max = {laplace_diag.max():.3e}")

    rng = np.random.default_rng(args.seed)
    W_mean = head.mean_head.weight.detach().cpu().numpy()  # [P, H]
    b_mean = head.mean_head.bias.detach().cpu().numpy()    # [P]
    W_std = np.sqrt(laplace_diag)  # diagonal Laplace, per-weight std

    # Pre-compute trunk(0) and trunk(e_i) features (these are deterministic
    # because trunk is held fixed; only mean_head is sampled).
    with torch.no_grad():
        y0 = torch.zeros(1, 8, device=device)
        feat0 = head.trunk(y0).cpu().numpy()[0]  # [H]
        feat_unit = []
        for i in range(8):
            y = torch.zeros(1, 8, device=device)
            y[0, i] = 1.0
            feat_unit.append(head.trunk(y).cpu().numpy()[0])  # [H]
        feat_unit = np.stack(feat_unit, axis=0)  # [P, H]

    n_samples = args.n_laplace
    M_samples = np.zeros((n_samples, 8, 8))
    for s in range(n_samples):
        eps = rng.standard_normal(W_mean.shape)
        W_s = W_mean + W_std * eps  # [P, H]
        # base = (W_s @ feat0) + b
        base = W_s @ feat0 + b_mean  # [P]
        for i in range(8):
            mu_at_e_i = W_s @ feat_unit[i] + b_mean  # [P]
            M_samples[s, i, :] = mu_at_e_i - base

    M_mean = M_samples.mean(0)
    M_std = M_samples.std(0)
    M_lo = np.quantile(M_samples, 0.025, axis=0)
    M_hi = np.quantile(M_samples, 0.975, axis=0)

    rows = []
    for i in range(8):
        for j in range(8):
            rows.append({
                "i": i, "j": j, "pushed": PHI_ORDER[i], "responded": PHI_ORDER[j],
                "M_post_mean": float(M_mean[i, j]),
                "M_post_std": float(M_std[i, j]),
                "ci_lo_2_5": float(M_lo[i, j]),
                "ci_hi_97_5": float(M_hi[i, j]),
            })
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"[write] {out_path}")

    # Sanity: deterministic M from c2_report should be inside CI for ~95% of entries
    M_det = load_M_baseline(Path(args.c2_report))
    inside = (M_det >= M_lo) & (M_det <= M_hi)
    print(f"[sanity] deterministic M inside 95% CI for {inside.sum()}/64 entries")

    return _summarize_laplace(df, args)


def _summarize_laplace(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    K = len(PHI_ORDER)
    M_det = load_M_baseline(Path(args.c2_report))
    off = ~np.eye(K, dtype=bool)
    strong = np.abs(M_det) > NOISE_FLOOR
    strong_off = strong & off
    n_strong_off = int(strong_off.sum())

    n_robust = 0
    for _, row in df.iterrows():
        i, j = int(row["i"]), int(row["j"])
        if not strong_off[i, j]:
            continue
        lo, hi = row["ci_lo_2_5"], row["ci_hi_97_5"]
        m_det = M_det[i, j]
        if (lo > 0 and m_det > 0) or (hi < 0 and m_det < 0):
            n_robust += 1

    print()
    print("=" * 70)
    print("Sub-task 3 (Laplace) summary")
    print("=" * 70)
    print(f"  Strong off-diag entries:                 {n_strong_off}")
    print(f"  Strong entries with 95% CI excluding 0: {n_robust}/{n_strong_off}")
    print()

    return {
        "subtask": "laplace",
        "n_strong_off": n_strong_off,
        "n_robust": n_robust,
        "robust_threshold": 18,
        "robust_pass": bool(n_robust >= 18),
        "n_samples": args.n_laplace,
    }


# ---------------------------------------------------------------------------
# Sub-task: seed_stability (Tier-2)
# ---------------------------------------------------------------------------


def run_seed_stability(args: argparse.Namespace, out_root: Path) -> dict:
    """Retrain μ-only heads across seeds and summarize M stability.

    This intentionally isolates the chemical-potential head: the Phase-3
    implementation already trains μ with

        y = phi_z + sigma * N(0, I),     target = beta * phi_z

    so the seed-stability question can be answered without running the
    coupling PCD loop again. That makes this a direct test of training
    stochasticity in the μ field rather than a confounded full-model retrain.
    """
    import torch
    sys.path.insert(0, str(Path(args.repo_root) / "src"))
    from thermofrag.potentials.external_field import ChemicalPotentialHead  # type: ignore

    out_dir = out_root / "seed_stability"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / "M_per_seed.parquet"
    summary_path = out_dir / "M_mean_std.parquet"

    if per_seed_path.exists() and summary_path.exists() and not args.force:
        print(f"[skip] {per_seed_path} and {summary_path} already exist; use --force to rebuild")
        return _summarize_seed_stability(pd.read_parquet(summary_path), args)

    meta = load_phi_meta_from_lmdb(args.lmdb)
    if meta.get("phi_properties") and meta["phi_properties"] != PHI_ORDER:
        raise RuntimeError(f"phi_properties mismatch in LMDB meta: {meta['phi_properties']}")

    phi = load_phi_from_lmdb(args.lmdb, max_records=args.max_records).astype(np.float32)
    if phi.shape[1] != len(PHI_ORDER):
        raise RuntimeError(f"expected {len(PHI_ORDER)} phi columns, got {phi.shape[1]}")
    phi_mean = np.asarray(meta.get("phi_mean") or phi.mean(axis=0), dtype=np.float32)
    phi_std = np.asarray(meta.get("phi_std") or phi.std(axis=0), dtype=np.float32)
    phi_std = np.where(phi_std == 0, 1.0, phi_std).astype(np.float32)
    phi_z_np = (phi - phi_mean) / phi_std
    print(f"[seed_stability] phi_z shape = {phi_z_np.shape}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"[seed_stability] device={device} n_seeds={args.n_seeds} steps={args.seed_steps}")

    phi_z = torch.as_tensor(phi_z_np, dtype=torch.float32)
    n = phi_z.shape[0]
    batch_size = int(args.seed_batch_size)
    n_props = len(PHI_ORDER)
    seed_rows = []
    metrics_rows = []

    for seed_offset in range(args.n_seeds):
        seed = int(args.seed + seed_offset)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        head = ChemicalPotentialHead(n_properties=n_props, hidden=int(args.seed_hidden)).to(device)
        opt = torch.optim.AdamW(
            head.parameters(),
            lr=float(args.seed_lr),
            weight_decay=float(args.seed_weight_decay),
        )
        loss_ema = None
        head.train()
        t_seed = time.time()
        for step in range(1, int(args.seed_steps) + 1):
            idx = rng.integers(0, n, size=batch_size)
            target = phi_z[idx].to(device, non_blocking=True)
            y = target + float(args.seed_y_noise) * torch.randn_like(target)
            pred = head(y)
            loss = ((pred - float(args.seed_mu_beta) * target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(args.seed_grad_clip))
            opt.step()
            loss_val = float(loss.detach().cpu())
            loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
            if args.seed_log_every and step % int(args.seed_log_every) == 0:
                print(f"[seed {seed:03d}] step={step:05d} loss={loss_val:.5f} ema={loss_ema:.5f}")

        head.eval()
        with torch.no_grad():
            y0 = torch.zeros(1, n_props, device=device)
            base = head(y0).detach().cpu().numpy()[0]
            M = np.zeros((n_props, n_props), dtype=np.float64)
            for i in range(n_props):
                y = torch.zeros(1, n_props, device=device)
                y[0, i] = 1.0
                M[i, :] = head(y).detach().cpu().numpy()[0] - base

        ckpt_path = ckpt_dir / f"mu_seed{seed}.pt"
        torch.save({
            "seed": seed,
            "state_dict": head.state_dict(),
            "phi_properties": PHI_ORDER,
            "training": {
                "steps": int(args.seed_steps),
                "batch_size": batch_size,
                "lr": float(args.seed_lr),
                "weight_decay": float(args.seed_weight_decay),
                "y_noise": float(args.seed_y_noise),
                "mu_beta": float(args.seed_mu_beta),
            },
        }, ckpt_path)

        for i in range(n_props):
            for j in range(n_props):
                seed_rows.append({
                    "seed": seed,
                    "i": i,
                    "j": j,
                    "pushed": PHI_ORDER[i],
                    "responded": PHI_ORDER[j],
                    "M": float(M[i, j]),
                })
        metrics_rows.append({
            "seed": seed,
            "final_loss": loss_val,
            "loss_ema": float(loss_ema if loss_ema is not None else loss_val),
            "wall_seconds": time.time() - t_seed,
            "ckpt": str(ckpt_path),
        })
        print(f"[seed {seed:03d}] done loss_ema={loss_ema:.5f} ckpt={ckpt_path}")

    per_seed_df = pd.DataFrame(seed_rows)
    per_seed_df.to_parquet(per_seed_path, index=False)
    pd.DataFrame(metrics_rows).to_parquet(out_dir / "seed_metrics.parquet", index=False)
    print(f"[write] {per_seed_path}")

    M_det = load_M_baseline(Path(args.c2_report))
    summary_rows = []
    for i in range(n_props):
        for j in range(n_props):
            vals = per_seed_df.query("i == @i and j == @j")["M"].to_numpy(dtype=np.float64)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            ci_lo = float(np.quantile(vals, 0.025))
            ci_hi = float(np.quantile(vals, 0.975))
            det = float(M_det[i, j])
            strong_off = bool((i != j) and abs(det) > NOISE_FLOOR)
            ci_excludes_zero = bool(ci_lo > 0 or ci_hi < 0)
            sign_matches_det = bool(np.sign(mean) == np.sign(det)) if det != 0 else True
            robust = bool(strong_off and ci_excludes_zero and sign_matches_det)
            summary_rows.append({
                "i": i,
                "j": j,
                "pushed": PHI_ORDER[i],
                "responded": PHI_ORDER[j],
                "M_det": det,
                "M_seed_mean": mean,
                "M_seed_std": std,
                "ci_lo_2_5": ci_lo,
                "ci_hi_97_5": ci_hi,
                "strong_offdiag": strong_off,
                "ci_excludes_zero": ci_excludes_zero,
                "sign_matches_det": sign_matches_det,
                "robust": robust,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(summary_path, index=False)
    print(f"[write] {summary_path}")

    return _summarize_seed_stability(summary_df, args)


def _summarize_seed_stability(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    strong = df[df["strong_offdiag"]]
    n_strong = int(len(strong))
    n_robust = int(strong["robust"].sum())
    n_sign_match = int(strong["sign_matches_det"].sum())
    print()
    print("=" * 70)
    print("Sub-task 2 (seed stability) summary")
    print("=" * 70)
    print(f"  Strong off-diag entries:                    {n_strong}")
    print(f"  Seed mean sign matches deterministic M:     {n_sign_match}/{n_strong}")
    print(f"  Strong entries with seed CI excluding zero: {n_robust}/{n_strong}")
    print()
    return {
        "subtask": "seed_stability",
        "n_strong_off": n_strong,
        "n_sign_match": n_sign_match,
        "n_robust": n_robust,
        "robust_threshold": 18,
        "robust_pass": bool(n_robust >= 18),
        "sign_match_pass": bool(n_sign_match >= min(18, n_strong)),
        "n_seeds": int(args.n_seeds),
        "steps": int(args.seed_steps),
    }


# ---------------------------------------------------------------------------
# Sub-task: feature_swap (Tier-3, offline proxy perturbations)
# ---------------------------------------------------------------------------


FEATURE_SWAP_DEFINITIONS = {
    "logp_slogp_vsa": (
        "Replace Crippen MolLogP with a surface-binned SlogP_VSA average "
        "proxy: sum((bin_index-6.5)*SlogP_VSA_i)/sum(SlogP_VSA_i)."
    ),
    "sa_bertz": (
        "Replace Ertl SA with RDKit BertzCT graph-complexity score. This is "
        "a synthetic-complexity proxy, not RAscore."
    ),
    "tpsa_sp": "Replace RDKit TPSA with CalcTPSA(includeSandP=True).",
    "qed_unweighted": "Replace Bickerton WEIGHT_MEAN QED with RDKit QED.WEIGHT_NONE.",
}


def _compute_feature_swap_matrices(
    args: argparse.Namespace,
    out_dir: Path,
    swaps: list[str],
) -> dict[str, np.ndarray]:
    """Return swapped φ matrices for requested swaps, caching them as npz."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, rdMolDescriptors

    cache_dir = out_dir / "phi_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    missing = []
    for swap in swaps:
        cache_path = cache_dir / f"phi_{swap}.npz"
        if cache_path.exists() and not args.force:
            out[swap] = np.load(cache_path)["phi"].astype(np.float32)
        else:
            missing.append(swap)
    if not missing:
        return out

    smiles, base_phi = load_smiles_phi_from_lmdb(args.lmdb, max_records=args.max_records)
    mats = {swap: base_phi.copy() for swap in missing}
    logp_idx = PHI_ORDER.index("logP")
    qed_idx = PHI_ORDER.index("qed")
    sa_idx = PHI_ORDER.index("sa")
    tpsa_idx = PHI_ORDER.index("tpsa")
    skipped = {swap: 0 for swap in missing}
    print(f"[feature_swap] computing {missing} for {len(smiles)} molecules")

    for row_idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            for swap in missing:
                skipped[swap] += 1
            continue
        try:
            if "logp_slogp_vsa" in missing:
                bins = [getattr(Descriptors, f"SlogP_VSA{i}")(mol) for i in range(1, 13)]
                denom = float(sum(bins))
                if denom > 0:
                    mats["logp_slogp_vsa"][row_idx, logp_idx] = float(
                        sum((i - 6.5) * b for i, b in enumerate(bins, start=1)) / denom
                    )
                else:
                    skipped["logp_slogp_vsa"] += 1
            if "sa_bertz" in missing:
                mats["sa_bertz"][row_idx, sa_idx] = float(Descriptors.BertzCT(mol))
            if "tpsa_sp" in missing:
                mats["tpsa_sp"][row_idx, tpsa_idx] = float(rdMolDescriptors.CalcTPSA(mol, includeSandP=True))
            if "qed_unweighted" in missing:
                mats["qed_unweighted"][row_idx, qed_idx] = float(QED.qed(mol, w=QED.WEIGHT_NONE))
        except Exception:
            for swap in missing:
                skipped[swap] += 1
        if args.feature_progress_every and (row_idx + 1) % int(args.feature_progress_every) == 0:
            print(f"[feature_swap] computed {row_idx + 1}/{len(smiles)}")

    for swap in missing:
        cache_path = cache_dir / f"phi_{swap}.npz"
        np.savez_compressed(cache_path, phi=mats[swap], skipped=skipped[swap])
        out[swap] = mats[swap].astype(np.float32)
        print(f"[write] {cache_path} skipped={skipped[swap]}")
    return out


def _train_mu_matrix_from_phi(
    phi_np: np.ndarray,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    hidden: int,
    lr: float,
    weight_decay: float,
    y_noise: float,
    mu_beta: float,
    grad_clip: float,
    device: str,
    log_prefix: str,
    log_every: int = 0,
) -> tuple[np.ndarray, dict, object]:
    import torch
    sys.path.insert(0, str(Path.cwd() / "src"))
    from thermofrag.potentials.external_field import ChemicalPotentialHead  # type: ignore

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    phi_mean = phi_np.mean(axis=0, dtype=np.float64).astype(np.float32)
    phi_std = phi_np.std(axis=0, dtype=np.float64).astype(np.float32)
    phi_std = np.where(phi_std == 0, 1.0, phi_std).astype(np.float32)
    phi_z_np = ((phi_np.astype(np.float32) - phi_mean) / phi_std).astype(np.float32)
    phi_z = torch.as_tensor(phi_z_np, dtype=torch.float32)
    n = phi_z.shape[0]
    n_props = phi_z.shape[1]

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    head = ChemicalPotentialHead(n_properties=n_props, hidden=hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_ema = None
    loss_val = float("nan")
    t0 = time.time()
    head.train()
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=batch_size)
        target = phi_z[idx].to(device, non_blocking=True)
        y = target + y_noise * torch.randn_like(target)
        pred = head(y)
        loss = ((pred - mu_beta * target) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        opt.step()
        loss_val = float(loss.detach().cpu())
        loss_ema = loss_val if loss_ema is None else 0.98 * loss_ema + 0.02 * loss_val
        if log_every and step % log_every == 0:
            print(f"[{log_prefix}] step={step:05d} loss={loss_val:.5f} ema={loss_ema:.5f}")

    head.eval()
    with torch.no_grad():
        y0 = torch.zeros(1, n_props, device=device)
        base = head(y0).detach().cpu().numpy()[0]
        M = np.zeros((n_props, n_props), dtype=np.float64)
        for i in range(n_props):
            y = torch.zeros(1, n_props, device=device)
            y[0, i] = 1.0
            M[i, :] = head(y).detach().cpu().numpy()[0] - base
    metrics = {
        "final_loss": loss_val,
        "loss_ema": float(loss_ema if loss_ema is not None else loss_val),
        "wall_seconds": time.time() - t0,
        "seed": seed,
        "steps": steps,
    }
    return M, metrics, head


def run_feature_swap(args: argparse.Namespace, out_root: Path) -> dict:
    out_dir = out_root / "feature_perturbation"
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.parquet"

    swaps = args.feature_swaps
    if len(swaps) == 1 and swaps[0] == "all":
        swaps = list(FEATURE_SWAP_DEFINITIONS)
    for swap in swaps:
        if swap not in FEATURE_SWAP_DEFINITIONS:
            raise ValueError(f"unknown feature swap '{swap}'. Valid: {sorted(FEATURE_SWAP_DEFINITIONS)}")

    if summary_path.exists() and all((out_dir / f"M_{s}.parquet").exists() for s in swaps) and not args.force:
        print(f"[skip] feature swap outputs exist; use --force to rebuild")
        return _summarize_feature_swap(pd.read_parquet(summary_path), args)

    matrices = _compute_feature_swap_matrices(args, out_dir, swaps)
    M_det = load_M_baseline(Path(args.c2_report))
    n_props = len(PHI_ORDER)
    strong_off = (np.abs(M_det) > NOISE_FLOOR) & ~np.eye(n_props, dtype=bool)

    summary_rows = []
    for swap in swaps:
        print(f"[feature_swap] training μ head for {swap}: {FEATURE_SWAP_DEFINITIONS[swap]}")
        M, metrics, head = _train_mu_matrix_from_phi(
            matrices[swap],
            seed=int(args.feature_seed),
            steps=int(args.feature_steps),
            batch_size=int(args.feature_batch_size),
            hidden=int(args.seed_hidden),
            lr=float(args.seed_lr),
            weight_decay=float(args.seed_weight_decay),
            y_noise=float(args.seed_y_noise),
            mu_beta=float(args.seed_mu_beta),
            grad_clip=float(args.seed_grad_clip),
            device=args.device,
            log_prefix=f"feature:{swap}",
            log_every=int(args.seed_log_every),
        )
        torch_path = ckpt_dir / f"mu_{swap}.pt"
        try:
            import torch
            torch.save({
                "swap": swap,
                "swap_definition": FEATURE_SWAP_DEFINITIONS[swap],
                "state_dict": head.state_dict(),
                "phi_properties": PHI_ORDER,
                "metrics": metrics,
            }, torch_path)
        except Exception as e:
            print(f"[warn] checkpoint save failed for {swap}: {e}")

        rows = []
        for i in range(n_props):
            for j in range(n_props):
                sign_matches = bool(np.sign(M[i, j]) == np.sign(M_det[i, j])) if M_det[i, j] != 0 else True
                row = {
                    "swap": swap,
                    "swap_definition": FEATURE_SWAP_DEFINITIONS[swap],
                    "i": i,
                    "j": j,
                    "pushed": PHI_ORDER[i],
                    "responded": PHI_ORDER[j],
                    "M_det": float(M_det[i, j]),
                    "M_swap": float(M[i, j]),
                    "strong_offdiag": bool(strong_off[i, j]),
                    "sign_matches_det": sign_matches,
                    "final_loss": metrics["final_loss"],
                    "loss_ema": metrics["loss_ema"],
                    "ckpt": str(torch_path),
                }
                rows.append(row)
                summary_rows.append(row)
        swap_path = out_dir / f"M_{swap}.parquet"
        pd.DataFrame(rows).to_parquet(swap_path, index=False)
        print(f"[write] {swap_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(summary_path, index=False)
    print(f"[write] {summary_path}")
    return _summarize_feature_swap(summary_df, args)


def _summarize_feature_swap(df: pd.DataFrame, args: argparse.Namespace) -> dict:
    strong = df[df["strong_offdiag"]]
    per_swap = strong.groupby("swap")["sign_matches_det"].agg(["sum", "count"]).reset_index()
    pivot = strong.pivot_table(index=["i", "j", "pushed", "responded"], columns="swap", values="sign_matches_det", aggfunc="first")
    if len(pivot) == 0:
        n_all = 0
        n_strong = 0
    else:
        all_preserved = pivot.all(axis=1)
        n_all = int(all_preserved.sum())
        n_strong = int(len(all_preserved))
    print()
    print("=" * 70)
    print("Sub-task 4 (feature perturbation) summary")
    print("=" * 70)
    for row in per_swap.itertuples(index=False):
        print(f"  {row.swap:<18s} sign-preserved strong entries: {int(row.sum)}/{int(row.count)}")
    print(f"  Preserved across all swaps: {n_all}/{n_strong}")
    print()
    return {
        "subtask": "feature_swap",
        "n_strong_off": n_strong,
        "n_preserved_all_swaps": n_all,
        "preserve_threshold": 18,
        "preserve_pass": bool(n_all >= 18),
        "swaps": {
            str(row.swap): {"n_preserved": int(row.sum), "n_strong_off": int(row.count)}
            for row in per_swap.itertuples(index=False)
        },
        "swap_definitions": FEATURE_SWAP_DEFINITIONS,
        "note": "Offline proxy perturbations: true XLogP3/RAscore/3D-PSA were not available in the local environment.",
    }


# ---------------------------------------------------------------------------
# Sub-task: drugbank (Tier-1, independent population)
# ---------------------------------------------------------------------------


def run_drugbank(args: argparse.Namespace, out_root: Path) -> dict:
    """Replicate the partial-corr(HBA, HBD | TPSA, MW) test on an
    independent population (DrugBank approved drugs).

    Input expected at ``args.drugbank_smi`` (one SMILES per line, plain
    text or .smi). If absent, this sub-task is skipped honestly with a
    note. RDKit is used to recompute the 8-dim φ, so this does not depend
    on prior caching.
    """
    out_dir = out_root / "independent_population"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "drugbank_corr.parquet"

    smi_path = Path(args.drugbank_smi) if args.drugbank_smi else None
    if smi_path is None or not smi_path.exists():
        print(f"[skip] drugbank SMILES not found at {smi_path}; honest-skip")
        return {"subtask": "drugbank", "skipped": True, "reason": f"no SMILES at {smi_path}"}

    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; use --force to rebuild")
        return _summarize_drugbank(pd.read_parquet(out_path))

    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED
    try:
        from rdkit.Chem import RDConfig
        sa_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
        if sa_path.exists():
            sys.path.append(str(sa_path.parent))
            import sascorer  # type: ignore
            sa_fn = sascorer.calculateScore
        else:
            sa_fn = None
    except Exception:
        sa_fn = None

    smiles = [s.strip() for s in smi_path.read_text().splitlines() if s.strip()]
    print(f"[load] {len(smiles)} SMILES from {smi_path}")
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            phi = [
                Crippen.MolLogP(m),
                QED.qed(m),
                sa_fn(m) if sa_fn else float("nan"),
                Descriptors.TPSA(m),
                Descriptors.MolWt(m),
                Lipinski.NumHAcceptors(m),
                Lipinski.NumHDonors(m),
                Lipinski.NumRotatableBonds(m),
            ]
        except Exception:
            continue
        if any(np.isnan(phi)):
            continue
        rows.append(phi)
    phi = np.asarray(rows, dtype=np.float64)
    print(f"[parse] phi shape = {phi.shape}")

    # Same partial-corr suite as ChEMBL
    hba, hbd = PHI_ORDER.index("hba"), PHI_ORDER.index("hbd")
    tpsa, mw = PHI_ORDER.index("tpsa"), PHI_ORDER.index("mw")
    logp, sa = PHI_ORDER.index("logP"), PHI_ORDER.index("sa")

    specs = [
        ("hba_hbd", "marginal", np.zeros((len(phi), 0))),
        ("hba_hbd", "controls=tpsa,mw", phi[:, [tpsa, mw]]),
        ("hba_hbd", "controls=tpsa", phi[:, [tpsa]]),
        ("hba_hbd", "controls=mw", phi[:, [mw]]),
        ("hba_hbd", "controls=tpsa,mw,logp,sa", phi[:, [tpsa, mw, logp, sa]]),
    ]
    out_rows = []
    for pair, ctrl_name, ctrl in specs:
        x = phi[:, hba]; y = phi[:, hbd]
        if ctrl.shape[1] == 0:
            r, _ = pearsonr(x, y)
            p = _pearson_p(r, len(x))
        else:
            r, p = partial_corr(x, y, ctrl)
        out_rows.append({"pair": pair, "controls": ctrl_name, "r": float(r), "p": float(p),
                         "n": int(len(phi))})
        print(f"[drugbank] {pair} | {ctrl_name:32s} r={r:+.4f}  p={p:.2e}")

    df = pd.DataFrame(out_rows)
    df.to_parquet(out_path, index=False)
    print(f"[write] {out_path}")

    return _summarize_drugbank(df)


def _summarize_drugbank(df: pd.DataFrame) -> dict:
    main = df.query("pair == 'hba_hbd' and controls == 'controls=tpsa,mw'").iloc[0]
    print(f"  DrugBank partial corr(HBA, HBD | TPSA, MW): r={main['r']:+.4f}  p={main['p']:.2e}")
    print(f"  Cat-E corroboration on DrugBank:           {'PASS' if main['r'] < 0 else 'FAIL'}")
    return {
        "subtask": "drugbank",
        "hba_hbd_partial_tpsa_mw_r": float(main["r"]),
        "hba_hbd_partial_tpsa_mw_p": float(main["p"]),
        "cat_e_drugbank_pass": bool(main["r"] < 0),
        "n": int(main["n"]),
    }


# ---------------------------------------------------------------------------
# Sub-task: litpcba (Tier-1, independent population from ChEMBL)
# ---------------------------------------------------------------------------


def _phi_from_smiles(smiles_iter):
    """Vectorize 8-dim φ for a SMILES iterator using the same definitions
    as ``scripts/build_conditional_lmdb.py``. Returns float64 array.
    """
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, QED
    try:
        from rdkit.Chem import RDConfig
        sa_path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
        if sa_path.exists():
            sys.path.append(str(sa_path.parent))
            import sascorer  # type: ignore
            sa_fn = sascorer.calculateScore
        else:
            sa_fn = None
    except Exception:
        sa_fn = None

    rows = []
    skipped = 0
    for s in smiles_iter:
        m = Chem.MolFromSmiles(s)
        if m is None:
            skipped += 1
            continue
        try:
            phi = [
                Crippen.MolLogP(m),
                QED.qed(m),
                sa_fn(m) if sa_fn else float("nan"),
                Descriptors.TPSA(m),
                Descriptors.MolWt(m),
                Lipinski.NumHAcceptors(m),
                Lipinski.NumHDonors(m),
                Lipinski.NumRotatableBonds(m),
            ]
        except Exception:
            skipped += 1
            continue
        if any(np.isnan(phi)):
            skipped += 1
            continue
        rows.append(phi)
    return np.asarray(rows, dtype=np.float64), skipped


def run_litpcba(args: argparse.Namespace, out_root: Path) -> dict:
    """Replicate the partial-corr(HBA, HBD | TPSA, MW) test on LIT-PCBA
    actives (pooled across 15 targets). Independent population from the
    ChEMBL training set; deduplicates on SMILES.
    """
    out_dir = out_root / "independent_population"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "litpcba_actives_corr.parquet"

    if not args.litpcba_dir.exists():
        print(f"[skip] {args.litpcba_dir} not found")
        return {"subtask": "litpcba", "skipped": True, "reason": str(args.litpcba_dir)}

    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; use --force to rebuild")
        return _summarize_litpcba(pd.read_parquet(out_path))

    smiles_pool: set[str] = set()
    for smi_file in sorted(args.litpcba_dir.glob("*.smi")):
        for line in smi_file.read_text().splitlines():
            tok = line.strip().split()
            if not tok:
                continue
            smiles_pool.add(tok[0])
    smiles = sorted(smiles_pool)
    print(f"[load] {len(smiles)} unique SMILES from {args.litpcba_dir}")

    phi, skipped = _phi_from_smiles(smiles)
    print(f"[parse] phi shape = {phi.shape} (skipped {skipped})")

    hba, hbd = PHI_ORDER.index("hba"), PHI_ORDER.index("hbd")
    tpsa, mw = PHI_ORDER.index("tpsa"), PHI_ORDER.index("mw")
    logp, sa = PHI_ORDER.index("logP"), PHI_ORDER.index("sa")

    specs = [
        ("hba_hbd", "marginal", np.zeros((len(phi), 0))),
        ("hba_hbd", "controls=tpsa,mw", phi[:, [tpsa, mw]]),
        ("hba_hbd", "controls=tpsa", phi[:, [tpsa]]),
        ("hba_hbd", "controls=mw", phi[:, [mw]]),
        ("hba_hbd", "controls=tpsa,mw,logp,sa", phi[:, [tpsa, mw, logp, sa]]),
    ]
    out_rows = []
    for pair, ctrl_name, ctrl in specs:
        x = phi[:, hba]; y = phi[:, hbd]
        if ctrl.shape[1] == 0:
            r, _ = pearsonr(x, y)
            p = _pearson_p(r, len(x))
        else:
            r, p = partial_corr(x, y, ctrl)
        out_rows.append({"pair": pair, "controls": ctrl_name, "r": float(r), "p": float(p),
                         "n": int(len(phi))})
        print(f"[litpcba] {pair} | {ctrl_name:32s} r={r:+.4f}  p={p:.2e}")

    df = pd.DataFrame(out_rows)
    df.to_parquet(out_path, index=False)
    print(f"[write] {out_path}")
    return _summarize_litpcba(df)


def _summarize_litpcba(df: pd.DataFrame) -> dict:
    main = df.query("pair == 'hba_hbd' and controls == 'controls=tpsa,mw'").iloc[0]
    print(f"  LIT-PCBA partial corr(HBA, HBD | TPSA, MW): r={main['r']:+.4f}  p={main['p']:.2e}")
    print(f"  Cat-E corroboration on LIT-PCBA actives:    {'PASS' if main['r'] < 0 else 'FAIL'}")
    return {
        "subtask": "litpcba",
        "hba_hbd_partial_tpsa_mw_r": float(main["r"]),
        "hba_hbd_partial_tpsa_mw_p": float(main["p"]),
        "cat_e_litpcba_pass": bool(main["r"] < 0),
        "n": int(main["n"]),
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def write_aggregate(out_root: Path, all_summaries: list[dict], args: argparse.Namespace) -> None:
    agg_dir = Path(args.repo_root) / "results" / "eval" / "phase7" / "AGGREGATE"
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_path = agg_dir / "07_mu_crossval_summary.json"

    # Merge with any existing summary (to support running sub-tasks separately)
    existing = {}
    if agg_path.exists():
        try:
            existing = json.loads(agg_path.read_text())
        except Exception:
            pass

    by_sub = existing.get("subtasks", {})
    for s in all_summaries:
        by_sub[s["subtask"]] = s

    thresholds = {}
    if "chembl_corr" in by_sub:
        s = by_sub["chembl_corr"]
        thresholds["sign_agreement_vs_chembl_pearson"] = {
            "target": ">= 18 / 22 strong off-diag",
            "observed": f"{s['n_sign_match']}/{s['n_strong_off']}",
            "pass": bool(s.get("sign_match_pass", False)),
        }
        thresholds["cat_e_partial_corr_decisive"] = {
            "target": "partial_corr(HBA, HBD | TPSA, MW) < 0 on raw ChEMBL",
            "observed": f"r = {s['hba_hbd_partial_tpsa_mw_r']:+.4f} (p={s['hba_hbd_partial_tpsa_mw_p']:.2e})",
            "pass": bool(s.get("cat_e_decisive_pass", False)),
        }
    if "laplace" in by_sub:
        s = by_sub["laplace"]
        thresholds["laplace_robust_strong_offdiag"] = {
            "target": ">= 18 / strong with 95% CI excluding 0",
            "observed": f"{s['n_robust']}/{s['n_strong_off']}",
            "pass": bool(s.get("robust_pass", False)),
        }
    if "seed_stability" in by_sub:
        s = by_sub["seed_stability"]
        thresholds["seed_stability_robust_strong_offdiag"] = {
            "target": ">= 18 / strong with 95% seed CI excluding 0",
            "observed": f"{s['n_robust']}/{s['n_strong_off']}",
            "pass": bool(s.get("robust_pass", False)),
        }
        thresholds["seed_stability_sign_agreement"] = {
            "target": ">= 18 / strong seed-mean signs match deterministic M",
            "observed": f"{s['n_sign_match']}/{s['n_strong_off']}",
            "pass": bool(s.get("sign_match_pass", False)),
        }
    if "feature_swap" in by_sub:
        s = by_sub["feature_swap"]
        thresholds["feature_swap_sign_preservation"] = {
            "target": ">= 18 / strong off-diag preserve sign across feature swaps",
            "observed": f"{s['n_preserved_all_swaps']}/{s['n_strong_off']}",
            "pass": bool(s.get("preserve_pass", False)),
        }
    if "drugbank" in by_sub and not by_sub["drugbank"].get("skipped"):
        s = by_sub["drugbank"]
        thresholds["cat_e_drugbank_corroboration"] = {
            "target": "partial_corr(HBA, HBD | TPSA, MW) < 0 on DrugBank",
            "observed": f"r = {s['hba_hbd_partial_tpsa_mw_r']:+.4f} (p={s['hba_hbd_partial_tpsa_mw_p']:.2e})",
            "pass": bool(s.get("cat_e_drugbank_pass", False)),
        }
    if "litpcba" in by_sub and not by_sub["litpcba"].get("skipped"):
        s = by_sub["litpcba"]
        thresholds["cat_e_litpcba_corroboration"] = {
            "target": "partial_corr(HBA, HBD | TPSA, MW) < 0 on LIT-PCBA actives (pooled)",
            "observed": f"r = {s['hba_hbd_partial_tpsa_mw_r']:+.4f} (p={s['hba_hbd_partial_tpsa_mw_p']:.2e})",
            "pass": bool(s.get("cat_e_litpcba_pass", False)),
        }

    payload = {
        "task_id": "07_mu_crossval",
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "thresholds": thresholds,
        "subtasks": by_sub,
        "notes": "Tier-1: chembl_corr + laplace + independent population. "
                 "Tier-2: seed_stability retrains μ-only heads. "
                 "Tier-3 feature_swap uses offline proxy perturbations when external XLogP3/RAscore are unavailable.",
    }
    agg_path.write_text(json.dumps(payload, indent=2))
    print(f"[write] {agg_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subtask", required=True,
                   choices=[
                       "chembl_corr", "laplace", "seed_stability", "feature_swap",
                       "drugbank", "litpcba", "tier1", "tier2", "tier3",
                   ])
    p.add_argument("--lmdb", type=Path, default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--ckpt", type=Path, default=Path("results/checkpoints/joint_final.pt"))
    p.add_argument("--c2_report", type=Path, default=Path("results/eval/phase3/c2_report.json"))
    p.add_argument("--drugbank_smi", type=Path,
                   default=Path("data/external/drugbank_approved.smi"),
                   help="One SMILES per line; if missing the drugbank sub-task self-skips.")
    p.add_argument("--litpcba_dir", type=Path,
                   default=Path("data/external/litpcba_actives"),
                   help="Directory with per-target .smi files (pooled across targets).")
    p.add_argument("--out_root", type=Path,
                   default=Path("results/eval/phase7/mu_crossval"))
    p.add_argument("--repo_root", type=Path, default=Path("/home/zhao/code/ThermoFrag"))
    p.add_argument("--max_records", type=int, default=None,
                   help="Subsample LMDB for testing; default uses all.")
    p.add_argument("--n_laplace", type=int, default=1000,
                   help="Posterior samples for the Laplace sub-task.")
    p.add_argument("--n_seeds", type=int, default=10,
                   help="Number of μ-only retraining seeds for seed_stability.")
    p.add_argument("--seed_steps", type=int, default=3000,
                   help="Optimizer steps per seed for μ-only retraining.")
    p.add_argument("--seed_batch_size", type=int, default=1024,
                   help="Batch size for μ-only retraining.")
    p.add_argument("--seed_hidden", type=int, default=256,
                   help="Hidden size of the μ head for seed_stability.")
    p.add_argument("--seed_lr", type=float, default=1.0e-3,
                   help="Learning rate for μ-only retraining.")
    p.add_argument("--seed_weight_decay", type=float, default=1.0e-5,
                   help="Weight decay for μ-only retraining.")
    p.add_argument("--seed_y_noise", type=float, default=0.3,
                   help="Gaussian noise added to standardized y during μ retraining.")
    p.add_argument("--seed_mu_beta", type=float, default=1.0,
                   help="Scale factor in μ target = beta * phi_z.")
    p.add_argument("--seed_grad_clip", type=float, default=5.0,
                   help="Gradient clipping norm for μ-only retraining.")
    p.add_argument("--seed_log_every", type=int, default=0,
                   help="Log every N steps per seed; 0 disables step logs.")
    p.add_argument("--feature_swaps", nargs="+", default=["all"],
                   help="Feature swaps to run, or 'all'. Valid: logp_slogp_vsa sa_bertz tpsa_sp qed_unweighted.")
    p.add_argument("--feature_steps", type=int, default=3000,
                   help="Optimizer steps per feature-swap μ retrain.")
    p.add_argument("--feature_batch_size", type=int, default=1024,
                   help="Batch size for feature-swap μ retraining.")
    p.add_argument("--feature_seed", type=int, default=42,
                   help="Random seed for feature-swap μ retraining.")
    p.add_argument("--feature_progress_every", type=int, default=50000,
                   help="Progress interval while recomputing swapped features; 0 disables.")
    p.add_argument("--device", default="cuda",
                   help="Device for seed_stability retraining.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="Recompute outputs even if parquet exists.")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"[dry] subtask={args.subtask} out_root={out_root}")
        return

    inputs = {
        "lmdb": args.lmdb,
        "ckpt": args.ckpt,
        "c2_report": args.c2_report,
    }
    if args.drugbank_smi.exists():
        inputs["drugbank_smi"] = args.drugbank_smi

    t0 = time.time()
    summaries = []
    if args.subtask == "chembl_corr":
        summaries.append(run_chembl_corr(args, out_root))
    elif args.subtask == "laplace":
        summaries.append(run_laplace(args, out_root))
    elif args.subtask == "seed_stability":
        summaries.append(run_seed_stability(args, out_root))
    elif args.subtask == "feature_swap":
        summaries.append(run_feature_swap(args, out_root))
    elif args.subtask == "drugbank":
        summaries.append(run_drugbank(args, out_root))
    elif args.subtask == "litpcba":
        summaries.append(run_litpcba(args, out_root))
    elif args.subtask == "tier1":
        summaries.append(run_chembl_corr(args, out_root))
        summaries.append(run_laplace(args, out_root))
        summaries.append(run_drugbank(args, out_root))
        summaries.append(run_litpcba(args, out_root))
    elif args.subtask == "tier2":
        summaries.append(run_seed_stability(args, out_root))
    elif args.subtask == "tier3":
        summaries.append(run_feature_swap(args, out_root))

    write_aggregate(out_root, summaries, args)
    write_manifest(out_root, args, inputs, time.time() - t0)


if __name__ == "__main__":
    main()
