"""Phase-4 / claim-C4 strain audit.

Reads every ``results/eval/phase4/decoded/<target>.parquet``, pulls valid
SMILES (``smiles.notna()``), computes GAFF strain energy
(``E_GAFF(x_MMFF94_min) - E_GAFF(x_GAFF_min)``) per molecule, and writes:

    results/eval/phase4/strain/<target>.parquet
        target, chain_idx, smiles, e_mmff, e_gaff, strain, n_atoms, status
    results/eval/phase4/strain/summary.csv
        per-target counts + mean/median/std of strain (kcal/mol)

Uses a process pool — each SMILES takes seconds to minutes, dominated by
AM1-BCC charge computation.
"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("strain")


def _worker(args):
    idx, smiles = args
    from thermofrag.eval.openmm_strain import compute_strain
    r = compute_strain(smiles)
    return idx, r


def run_target(df: pd.DataFrame, n_workers: int) -> pd.DataFrame:
    valid = df[df["smiles"].notna()].reset_index(drop=True)
    if len(valid) == 0:
        return pd.DataFrame(columns=["chain_idx", "smiles", "e_mmff",
                                     "e_gaff", "strain", "n_atoms", "status"])

    jobs = [(i, s) for i, s in enumerate(valid["smiles"].tolist())]
    rows = [None] * len(jobs)

    if n_workers <= 1:
        for i, smi in jobs:
            _, r = _worker((i, smi))
            rows[i] = r
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = [pool.submit(_worker, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                idx, r = fut.result()
                rows[idx] = r
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    logger.info("  %d / %d done", done, len(jobs))

    out = pd.DataFrame({
        "chain_idx": valid["chain_idx"].astype(int).tolist(),
        "smiles":    [r.smiles for r in rows],
        "e_mmff":    [r.e_mmff for r in rows],
        "e_gaff":    [r.e_gaff for r in rows],
        "strain":    [r.strain for r in rows],
        "n_atoms":   [r.n_atoms for r in rows],
        "status":    [r.status for r in rows],
    })
    return out


def summarize(df: pd.DataFrame) -> dict:
    ok = df[df["status"] == "ok"]
    n_total = len(df)
    n_ok = len(ok)
    if n_ok == 0:
        return {"n_total": n_total, "n_ok": 0, "mean": None, "median": None,
                "std": None, "p90": None}
    s = ok["strain"].to_numpy()
    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "mean":   float(np.mean(s)),
        "median": float(np.median(s)),
        "std":    float(np.std(s)),
        "p90":    float(np.percentile(s, 90)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-dir", type=Path,
                   default=Path("results/eval/phase4/decoded"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase4/strain"))
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--targets", nargs="*",
                   help="Optional subset of target names to evaluate "
                        "(default: every parquet in --decoded-dir).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of SMILES per target for smoke tests.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pqs = sorted(args.decoded_dir.glob("*.parquet"))
    if args.targets:
        keep = set(args.targets)
        pqs = [q for q in pqs if q.stem in keep]
    if not pqs:
        raise SystemExit(f"[strain] no parquets in {args.decoded_dir}")

    summary_rows = []
    for pq in pqs:
        target = pq.stem
        df = pd.read_parquet(pq)
        valid_n = int(df["smiles"].notna().sum())
        logger.info("=== %s (%d valid) ===", target, valid_n)
        if args.limit and valid_n > args.limit:
            valid_mask = df["smiles"].notna()
            idx_valid = df.index[valid_mask][:args.limit]
            df = df.loc[idx_valid].reset_index(drop=True)

        result_df = run_target(df, args.workers)
        result_df.insert(0, "target", target)
        out_pq = args.out_dir / f"{target}.parquet"
        result_df.to_parquet(out_pq)
        summ = summarize(result_df)
        summ["target"] = target
        summary_rows.append(summ)
        logger.info(
            "[strain] %s → n_ok=%d/%d  mean=%s  median=%s  p90=%s",
            target, summ["n_ok"], summ["n_total"],
            f"{summ['mean']:.2f}" if summ["mean"] is not None else "NA",
            f"{summ['median']:.2f}" if summ["median"] is not None else "NA",
            f"{summ['p90']:.2f}" if summ["p90"] is not None else "NA",
        )

    s_df = pd.DataFrame(summary_rows).set_index("target").sort_index()
    s_df.to_csv(args.out_dir / "summary.csv")
    logger.info("summary → %s", args.out_dir / "summary.csv")
    print(s_df)


if __name__ == "__main__":
    main()
