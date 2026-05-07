"""Dock the no-μ ablation pool against each of the 15 LIT-PCBA receptors.

Reads ``results/eval/phase5/nomu_samples/decoded.parquet`` (produced by
``scripts/sample_nomu_ablation.py`` + the decoder call in it), then for each
target dockings only the chain_idx that ALSO appear in the ThermoFrag pool
for that target (both pools share the same seed permutation, so matching
chain_idx means matching init_idx / seed). This reduces the dock set to the
paired intersection and saves ~4-5× compute vs docking every valid no-μ
ligand against every receptor.

Output::

    results/eval/phase5/nomu_vina/<target>.parquet   (chain_idx, smiles, vina_score, status)

Usage::

    python scripts/dock_nomu_ablation.py --workers 6 --exhaustiveness 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd


def _import_sibling(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_dock_vina = _import_sibling(
    "_dock_vina",
    Path(__file__).resolve().parent / "dock_vina.py",
)
dock_target = _dock_vina.dock_target
summarize = _dock_vina.summarize


logger = logging.getLogger("nomu_dock")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded", type=Path,
                   default=Path("results/eval/phase5/nomu_samples/decoded.parquet"))
    p.add_argument("--receptors", type=Path,
                   default=Path("data/external/receptors"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase5/nomu_vina"))
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap the number of SMILES (for smoke tests).")
    p.add_argument("--targets", nargs="*",
                   help="Subset of target names (default: everything in receptors/)")
    p.add_argument("--tf-decoded-dir", type=Path,
                   default=Path("results/eval/phase4/decoded"),
                   help="ThermoFrag decoded parquets; used to filter nomu dock set "
                        "to chain_idx ∩ TF valid per target (paired comparison).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    targets = args.targets or sorted(
        d.name for d in args.receptors.iterdir() if (d / "receptor.pdbqt").exists()
    )
    if not targets:
        raise SystemExit("no receptors found under " + str(args.receptors))

    nomu_full = pd.read_parquet(args.decoded)
    nomu_valid = nomu_full[nomu_full["smiles"].notna()].reset_index(drop=True)
    logger.info("nomu pool: %d valid SMILES", len(nomu_valid))

    summary_rows = []
    for t in targets:
        rec_dir = args.receptors / t
        out_pq = args.out_dir / f"{t}.parquet"
        logger.info("=== %s ===", t)
        # Filter to TF ∩ nomu chain_idx for paired comparison.
        tf_pq = args.tf_decoded_dir / f"{t}.parquet"
        if tf_pq.exists():
            tf_df = pd.read_parquet(tf_pq)
            tf_valid_cidx = set(tf_df[tf_df["smiles"].notna()]["chain_idx"].tolist())
            nomu_sub = nomu_valid[nomu_valid["chain_idx"].isin(tf_valid_cidx)]
            logger.info("  %s paired subset: %d ligands (TF valid=%d, nomu valid=%d)",
                        t, len(nomu_sub), len(tf_valid_cidx), len(nomu_valid))
        else:
            logger.warning("TF parquet missing for %s — using full nomu pool", t)
            nomu_sub = nomu_valid
        # Write a transient filtered parquet for dock_target to consume.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tf_tmp:
            nomu_sub.to_parquet(tf_tmp.name)
            tmp_path = Path(tf_tmp.name)
        try:
            df = dock_target(t, tmp_path, rec_dir, out_pq,
                             workers=args.workers,
                             exhaustiveness=args.exhaustiveness,
                             limit=args.limit)
        finally:
            tmp_path.unlink(missing_ok=True)
        s = summarize(df)
        s["target"] = t
        summary_rows.append(s)
        logger.info(
            "[nomu_dock] %s: n_ok=%d/%d mean=%s top10=%s",
            t, s["n_ok"], s["n_total"],
            f"{s['mean']:.2f}" if s["mean"] is not None else "NA",
            f"{s['top10_mean']:.2f}" if s["top10_mean"] is not None else "NA",
        )

    s_df = pd.DataFrame(summary_rows).set_index("target").sort_index()
    s_df.to_csv(args.out_dir / "summary.csv")
    logger.info("summary → %s", args.out_dir / "summary.csv")
    print(s_df)


if __name__ == "__main__":
    main()
