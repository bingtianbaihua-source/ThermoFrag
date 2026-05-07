#!/usr/bin/env python3
"""Comprehensive OOD AUROC breakdown by k (axes pushed) and r (push radius).

Extends `eval_ood_auroc.py` to provide:
  - mixed-cardinality reference (k random in [3, 8], r in [2.5, 4.0]) — the
    headline number reported in the paper
  - per-k AUROC for k in {3, 4, 5, 6, 7, 8} at the same r-range
  - per-r AUROC at fixed k=5 (median) for r in {2.5, 3.0, 3.5, 4.0}
  - single-axis-push AUROC (k=1) per individual phi axis as a sensitivity
    map (extreme case; not reported as headline because k=1 is too easy
    when the chosen axis is statistically rare in the training distribution)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_ood_auroc import (  # type: ignore
    collect_phi_z, load_mu_head, variance_norm, ZINCFragmentDataset,
    load_config,
)


def build_ood_fixed(phi_z_id, n_ood, k_fixed, r_lo=2.5, r_hi=4.0, seed=1):
    rng = np.random.default_rng(seed)
    N, K = phi_z_id.shape
    out = np.empty((n_ood, K), dtype=np.float32)
    for i in range(n_ood):
        y = phi_z_id[rng.integers(N)].copy()
        axes = rng.choice(K, size=k_fixed, replace=False)
        for a in axes:
            sign = rng.choice([-1.0, 1.0])
            r = rng.uniform(r_lo, r_hi)
            y[a] = sign * r
        out[i] = y
    return out


def build_ood_fixed_r(phi_z_id, n_ood, k_min=3, r_fixed=3.0, seed=1):
    rng = np.random.default_rng(seed)
    N, K = phi_z_id.shape
    out = np.empty((n_ood, K), dtype=np.float32)
    for i in range(n_ood):
        y = phi_z_id[rng.integers(N)].copy()
        k = rng.integers(k_min, K + 1)
        axes = rng.choice(K, size=k, replace=False)
        for a in axes:
            sign = rng.choice([-1.0, 1.0])
            y[a] = sign * r_fixed
        out[i] = y
    return out


def build_ood_single_axis(phi_z_id, n_ood, axis, r_lo=2.5, r_hi=4.0, seed=1):
    rng = np.random.default_rng(seed)
    N, K = phi_z_id.shape
    out = np.empty((n_ood, K), dtype=np.float32)
    for i in range(n_ood):
        y = phi_z_id[rng.integers(N)].copy()
        sign = rng.choice([-1.0, 1.0])
        r = rng.uniform(r_lo, r_hi)
        y[axis] = sign * r
        out[i] = y
    return out


def metrics(var_id, var_ood):
    y_true = np.concatenate([np.zeros(len(var_id)), np.ones(len(var_ood))])
    y_score = np.concatenate([var_id, var_ood])
    auroc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # FPR at TPR=0.95
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr_at_95 = float(fpr[idx])
    return {
        "auroc": auroc,
        "fpr_at_95tpr": fpr_at_95,
        "var_id_mean": float(var_id.mean()),
        "var_ood_mean": float(var_ood.mean()),
        "var_ratio_mean": float(var_ood.mean() / (var_id.mean() + 1e-12)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", type=Path,
                   default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--ckpt", type=Path,
                   default=Path("results/checkpoints/joint_final.pt"))
    p.add_argument("--config", type=Path, default=Path("configs/phase3.yaml"))
    p.add_argument("--out", type=Path,
                   default=Path("results/eval/phase5/ood_breakdown.json"))
    p.add_argument("--n-id", type=int, default=2000)
    p.add_argument("--n-ood", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    mu_hidden = int(cfg.get("joint_conditional", {}).get("mu_hidden", 256))
    dataset = ZINCFragmentDataset(args.lmdb, split="train")
    n_prop = dataset.phi_dim
    phi_props = dataset.phi_properties
    head = load_mu_head(args.ckpt, n_properties=n_prop, hidden=mu_hidden)

    phi_z_id = collect_phi_z(dataset, n=args.n_id, seed=args.seed)
    var_id = variance_norm(head, phi_z_id)

    out = {
        "n_id": args.n_id,
        "n_ood_per_setting": args.n_ood,
        "n_properties": n_prop,
        "phi_axes": phi_props,
        "var_id_mean": float(var_id.mean()),
    }

    # ------------------------------------------------------------------ headline
    from eval_ood_auroc import build_pareto_ood
    y_ref = build_pareto_ood(phi_z_id, args.n_ood, k_min=3, r_lo=2.5, r_hi=4.0,
                              seed=args.seed + 17)
    var_ref = variance_norm(head, y_ref)
    out["headline_mixed_3_to_8_axes_r_2.5_to_4.0"] = metrics(var_id, var_ref)

    # ------------------------------------------------------------------ per-k
    per_k = {}
    for k in range(1, n_prop + 1):
        y_k = build_ood_fixed(phi_z_id, args.n_ood, k_fixed=k,
                              r_lo=2.5, r_hi=4.0, seed=args.seed + 100 + k)
        var_k = variance_norm(head, y_k)
        per_k[f"k={k}"] = metrics(var_id, var_k)
    out["per_k_axes_pushed"] = per_k

    # ------------------------------------------------------------------ per-r
    per_r = {}
    for r in [2.5, 3.0, 3.5, 4.0]:
        y_r = build_ood_fixed_r(phi_z_id, args.n_ood, k_min=3, r_fixed=r,
                                 seed=args.seed + 200 + int(r * 10))
        var_r = variance_norm(head, y_r)
        per_r[f"r={r:.1f}"] = metrics(var_id, var_r)
    out["per_r_at_k_random_3_to_8"] = per_r

    # ------------------------------------------------------------------ single-axis
    per_axis = {}
    for ai, name in enumerate(phi_props):
        y_a = build_ood_single_axis(phi_z_id, args.n_ood, axis=ai,
                                     r_lo=2.5, r_hi=4.0,
                                     seed=args.seed + 300 + ai)
        var_a = variance_norm(head, y_a)
        per_axis[name] = metrics(var_id, var_a)
    out["per_single_axis_push"] = per_axis

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nReport saved to {args.out}")


if __name__ == "__main__":
    main()
