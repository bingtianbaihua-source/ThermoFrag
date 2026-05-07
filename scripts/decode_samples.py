"""Batch-decode Phase-4 fragment-graph samples into SMILES.

Reads every ``results/eval/phase4/samples/*.pkl`` produced by
``scripts/sample.py`` and writes, per target::

    results/eval/phase4/decoded/<target>.parquet
        target, chain_idx, init_idx, n_flips, mode, smiles  (one row per sample)
    results/eval/phase4/decoded/summary.csv
        one row per target with mode counts + yield

Only ``identical`` and ``leaf_flip`` samples carry a non-null SMILES; other
modes are recorded as-is for downstream diagnostics.
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import pandas as pd

from thermofrag.sampling.decoder import FragmentLibraryIndex, decode_pool


def decode_target(pkl_path: Path, lib: FragmentLibraryIndex) -> tuple[pd.DataFrame, dict]:
    with open(pkl_path, "rb") as fh:
        pool = pickle.load(fh)
    samples = pool["samples"]
    results = decode_pool(samples, lib)
    rows = []
    for chain_idx, (s, r) in enumerate(zip(samples, results)):
        rows.append(
            {
                "chain_idx": chain_idx,
                "init_idx": int(s["init_idx"]),
                "seed_smiles": s["seed_smiles"],
                "n_flips": int(r.n_flips),
                "mode": r.mode,
                "smiles": r.smiles,
            }
        )
    df = pd.DataFrame(rows)
    mode_counts = Counter(df["mode"])
    n_valid = int(df["smiles"].notna().sum())
    summary = {
        "total": len(df),
        "valid": n_valid,
        "yield": n_valid / len(df) if len(df) else 0.0,
        **{f"mode_{k}": v for k, v in mode_counts.items()},
    }
    return df, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-dir", type=Path,
                   default=Path("results/eval/phase4/samples"))
    p.add_argument("--lib", type=Path,
                   default=Path("data/processed/fragment_library.parquet"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase4/decoded"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lib = FragmentLibraryIndex.from_parquet(args.lib)

    pkls = sorted(args.samples_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"[decode] no .pkl files in {args.samples_dir}")

    summary_rows = []
    for pkl in pkls:
        target = pkl.stem
        df, summary = decode_target(pkl, lib)
        df.insert(0, "target", target)
        out_path = args.out_dir / f"{target}.parquet"
        df.to_parquet(out_path)
        summary["target"] = target
        summary_rows.append(summary)
        print(
            f"[decode] {target:12s}  total={summary['total']:4d}  "
            f"valid={summary['valid']:3d}  yield={summary['yield']:.3f}  "
            f"→ {out_path}"
        )

    s_df = pd.DataFrame(summary_rows).set_index("target").sort_index()
    s_df = s_df.fillna(0).astype({c: int for c in s_df.columns if c != "yield"})
    s_df.to_csv(args.out_dir / "summary.csv")
    print(f"[decode] summary → {args.out_dir / 'summary.csv'}")
    print(s_df[["total", "valid", "yield"]])


if __name__ == "__main__":
    main()
