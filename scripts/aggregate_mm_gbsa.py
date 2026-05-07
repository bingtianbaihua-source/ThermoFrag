"""Phase-7 task 1 aggregator — produce 01_mm_gbsa_summary.json.

Reads `results/eval/phase7/mm_gbsa/<gen>/<target>.parquet` for every
generator × target, applies the pre-registered thresholds from
`docs/validation/01_mm_gbsa.md`, and writes the summary JSON.

Pre-registered thresholds
-------------------------

* Spearman(mm_gbsa_total, vina_score) on the pooled top-10 across all
  4 generators × 15 targets must be ≥ 0.5.
* TF MM-GBSA top-10 mean must be **better** (more negative) than
  TargetDiff's on ≥ 10/15 targets at paired Wilcoxon p < 0.05.
* Per-ligand failure rate (status != ok) < 30 %.

Outputs
-------

::

    results/eval/phase7/AGGREGATE/01_mm_gbsa_summary.json

Usage
-----

::

    python scripts/aggregate_mm_gbsa.py
        [--in_root  results/eval/phase7/mm_gbsa]
        [--out      results/eval/phase7/AGGREGATE/01_mm_gbsa_summary.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("aggregate_mm_gbsa")

ALL_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]
ALL_GENERATORS = ["thermofrag", "targetdiff", "rxnflow", "bbar"]


def _load_target(in_root: Path, gen: str, target: str) -> pd.DataFrame:
    pq = in_root / gen / f"{target}.parquet"
    if not pq.exists():
        return pd.DataFrame()
    return pd.read_parquet(pq)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--in_root", type=Path,
                   default=Path("results/eval/phase7/mm_gbsa"))
    p.add_argument("--out", type=Path,
                   default=Path("results/eval/phase7/AGGREGATE/01_mm_gbsa_summary.json"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # -------- Load every parquet, build a long-format frame ---------
    rows = []
    n_total = 0
    n_ok = 0
    for gen in ALL_GENERATORS:
        for tgt in ALL_TARGETS:
            df = _load_target(args.in_root, gen, tgt)
            if df.empty:
                continue
            df = df.copy()
            df["generator"] = gen
            df["target"] = tgt
            n_total += len(df)
            n_ok += int((df["status"] == "ok").sum())
            rows.append(df)
    if not rows:
        raise SystemExit(f"no parquets found under {args.in_root}")
    long = pd.concat(rows, ignore_index=True)
    ok = long[long["status"] == "ok"].copy()
    failure_rate = 1.0 - (n_ok / max(n_total, 1))
    logger.info("loaded %d rows (%d ok); failure_rate=%.3f",
                n_total, n_ok, failure_rate)

    # -------- Threshold 1: Spearman(MM-GBSA, Vina) on pooled data ---
    spearman_rho, spearman_p = stats.spearmanr(
        ok["mm_gbsa_total"], ok["vina_score"])

    # -------- Threshold 2: TF vs TargetDiff per target Wilcoxon -----
    per_target = {}
    sig_wins = 0
    sig_targets = []
    for tgt in ALL_TARGETS:
        tf = ok[(ok["generator"] == "thermofrag") &
                (ok["target"] == tgt)]["mm_gbsa_total"].values
        td = ok[(ok["generator"] == "targetdiff") &
                (ok["target"] == tgt)]["mm_gbsa_total"].values
        if len(tf) < 3 or len(td) < 3:
            per_target[tgt] = {
                "n_tf": int(len(tf)), "n_td": int(len(td)),
                "tf_mean": float(np.mean(tf)) if len(tf) else None,
                "td_mean": float(np.mean(td)) if len(td) else None,
                "delta": None, "wilcoxon_p": None, "tf_better": None,
                "note": "insufficient samples for Wilcoxon",
            }
            continue
        # Paired Wilcoxon needs equal-length samples; pool rank-comparison
        # is not paired across generators (different ligands).  Use
        # Mann-Whitney instead (consistent with C3 generator-vs-generator
        # convention but two-sided).
        u_stat, u_p = stats.mannwhitneyu(tf, td, alternative="two-sided")
        # Direction: lower MM-GBSA = better.  TF wins if its median is lower.
        tf_better = float(np.median(tf)) < float(np.median(td))
        is_sig = (u_p < 0.05) and tf_better
        if is_sig:
            sig_wins += 1
            sig_targets.append(tgt)
        per_target[tgt] = {
            "n_tf": int(len(tf)), "n_td": int(len(td)),
            "tf_mean": float(np.mean(tf)),
            "td_mean": float(np.mean(td)),
            "tf_median": float(np.median(tf)),
            "td_median": float(np.median(td)),
            "delta_mean": float(np.mean(tf) - np.mean(td)),
            "delta_median": float(np.median(tf) - np.median(td)),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p": float(u_p),
            "tf_better": bool(tf_better),
            "tf_sig_better": bool(is_sig),
        }

    # Per-generator pooled means for the report.
    per_generator = {}
    for gen in ALL_GENERATORS:
        sub = ok[ok["generator"] == gen]
        per_generator[gen] = {
            "n_ok": int(len(sub)),
            "mm_gbsa_mean": float(sub["mm_gbsa_total"].mean()) if len(sub) else None,
            "mm_gbsa_median": float(sub["mm_gbsa_total"].median()) if len(sub) else None,
            "vina_mean": float(sub["vina_score"].mean()) if len(sub) else None,
        }

    summary = {
        "task_id": "01_mm_gbsa",
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "thresholds": {
            "spearman_vs_vina": {
                "target": ">= 0.5",
                "observed": float(spearman_rho),
                "p": float(spearman_p),
                "pass": bool(spearman_rho >= 0.5),
            },
            "tf_vs_targetdiff_sig_wins": {
                "target": ">= 10/15 sig wins (p<0.05)",
                "observed": f"{sig_wins}/15",
                "sig_targets": sig_targets,
                "pass": bool(sig_wins >= 10),
            },
            "max_failure_rate": {
                "target": "< 0.30",
                "observed": float(failure_rate),
                "pass": bool(failure_rate < 0.30),
            },
        },
        "per_generator": per_generator,
        "per_target": per_target,
        "n_total": int(n_total),
        "n_ok": int(n_ok),
        "n_failed": int(n_total - n_ok),
        "notes": (
            "MM-GBSA single-frame on docked Vina poses, igb=5 (OBC2), "
            "saltcon=0.150, ff14SB + GAFF2, AM1-BCC charges (Gasteiger "
            "fallback). Receptor + ligand minimized 200 cycles with "
            "heavy-atom restraints (10 kcal/mol/Å²) on protein heavy "
            "atoms before the single-point. TF-vs-TD significance test "
            "is Mann-Whitney U (unpaired across distinct ligand sets) "
            "rather than paired Wilcoxon — paired test inappropriate "
            "because TF and TargetDiff produce different molecules. "
            "tf_better = TF top-K median MM-GBSA more negative."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s — sig_wins=%d/15 spearman=%.3f failure_rate=%.3f",
                args.out, sig_wins, spearman_rho, failure_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
