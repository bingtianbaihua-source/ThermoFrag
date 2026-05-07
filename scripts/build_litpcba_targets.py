"""Compute per-target conditioning vector y for each LIT-PCBA target.

For each target T, select the top ``--top-frac`` of rows by docking score
(most negative = best binder, same convention as the BBAR pipeline), compute
the 8-dim property vector phi per compound via ``thermofrag.data.properties``,
take the mean, and save the raw (pre-standardization) vector so that
``scripts/sample.py --y-file`` can consume it.

Outputs:
    --out/<target>/y_raw.npy        # shape [8] float32, raw property means
    --out/<target>/y_std.npy        # standardized by conditional-LMDB phi_mean/std
    --out/<target>/summary.json     # n_top, score_range, phi_mean, phi_std used
    --out/index.json                # list of all targets written

Usage::

    python scripts/build_litpcba_targets.py \
        --litpcba /tmp/litpcba_check/LIT-PCBA/data.csv \
        --lmdb data/processed/chembl_conditional.lmdb \
        --top-frac 0.01 \
        --out results/eval/phase4/litpcba_targets
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from thermofrag.data.properties import compute_phi

RDLogger.DisableLog("rdApp.*")

PROPERTIES = ["logP", "qed", "sa", "tpsa", "mw", "hba", "hbd", "rotb"]


def _load_phi_stats(lmdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    env = lmdb.open(str(lmdb_path), subdir=False, readonly=True, lock=False)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__meta__"))
    env.close()
    return (
        np.asarray(meta["phi_mean"], dtype=np.float32),
        np.asarray(meta["phi_std"], dtype=np.float32),
    )


def _compute_phi_batch(smiles_list: list[str]) -> np.ndarray:
    out = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            out.append(compute_phi(mol, PROPERTIES))
        except Exception:
            continue
    if not out:
        return np.zeros((0, len(PROPERTIES)), dtype=np.float32)
    return np.stack(out, axis=0).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--litpcba", type=Path, required=True, help="LIT-PCBA data.csv")
    p.add_argument("--lmdb", type=Path, required=True, help="conditional LMDB for phi_mean/std")
    p.add_argument("--top-frac", type=float, default=0.01, help="top fraction of hits per target (by dock score)")
    p.add_argument("--min-top-n", type=int, default=200, help="floor on n per target if top_frac rounds too low")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--targets", default="all", help="comma-sep list, or 'all'")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    phi_mean, phi_std = _load_phi_stats(args.lmdb)
    print(f"[litpcba] phi_mean={phi_mean.round(3).tolist()}")
    print(f"[litpcba] phi_std ={phi_std.round(3).tolist()}")

    df = pd.read_csv(args.litpcba)
    TARGETS = [c for c in df.columns if c not in ("KEY", "SMILES", "QED")]
    if args.targets != "all":
        requested = [t.strip() for t in args.targets.split(",")]
        TARGETS = [t for t in TARGETS if t in requested]
        if not TARGETS:
            raise SystemExit(f"[litpcba] none of {requested} in LIT-PCBA columns")
    print(f"[litpcba] processing {len(TARGETS)} targets: {TARGETS}")

    index = {"targets": [], "phi_mean": phi_mean.tolist(), "phi_std": phi_std.tolist()}
    t0 = time.time()
    for tgt in TARGETS:
        vals = df[tgt].values
        n_top = max(int(round(len(vals) * args.top_frac)), int(args.min_top_n))
        n_top = min(n_top, len(vals))
        # Lower (more negative) = better binder.
        top_idx = np.argsort(vals)[:n_top]
        top_smiles = df["SMILES"].iloc[top_idx].tolist()
        top_scores = vals[top_idx]

        phi = _compute_phi_batch(top_smiles)
        if len(phi) < 10:
            print(f"[litpcba] {tgt}: too few valid phi ({len(phi)}); skipping")
            continue
        y_raw = phi.mean(axis=0)
        y_std = (y_raw - phi_mean) / phi_std

        tgt_dir = args.out / tgt
        tgt_dir.mkdir(exist_ok=True)
        np.save(tgt_dir / "y_raw.npy", y_raw.astype(np.float32))
        np.save(tgt_dir / "y_std.npy", y_std.astype(np.float32))
        summary = {
            "target": tgt,
            "n_top_requested": n_top,
            "n_phi_computed": int(len(phi)),
            "score_min": float(top_scores.min()),
            "score_max": float(top_scores.max()),
            "score_mean": float(top_scores.mean()),
            "y_raw": y_raw.tolist(),
            "y_std": y_std.tolist(),
            "phi_std_of_top": phi.std(axis=0).tolist(),
            "top_frac": args.top_frac,
        }
        (tgt_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        y_std_norm = float(np.linalg.norm(y_std))
        index["targets"].append({
            "target": tgt,
            "n_top": summary["n_phi_computed"],
            "score_mean": summary["score_mean"],
            "y_std_norm": y_std_norm,
        })
        print(
            f"[litpcba] {tgt:>12s}  n_top={summary['n_phi_computed']:4d}  "
            f"score_mean={summary['score_mean']:.2f}  "
            f"y_std_norm={y_std_norm:.2f}  "
            f"t={time.time()-t0:.1f}s"
        )

    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[litpcba] wrote {len(index['targets'])} targets -> {args.out}")


if __name__ == "__main__":
    main()
