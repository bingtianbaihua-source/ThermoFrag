"""Phase-5 / claim C3: compare ThermoFrag Vina scores to the LIT-PCBA reference.

For each of the 15 LIT-PCBA targets, compare:

  - ThermoFrag docked Vina scores (``results/eval/phase4/vina/<target>.parquet``).
  - The LIT-PCBA screening library's per-target Vina scores, straight out of
    ``data/external/LIT-PCBA.tar.gz`` (246k compounds, columns named after each
    target).

We report, per target:
  - ThermoFrag mean / median / top-10 mean (best 10 by affinity).
  - LIT-PCBA reference: overall mean, and mean of the top 1% (these are the
    "active" compounds the Phase-4 y vectors were drawn from).
  - Paired metrics: percentile rank of ThermoFrag's mean inside the library,
    and Mann-Whitney U (two-sided, comparing ThermoFrag vs library).

Output::

    results/eval/phase5/c3_vs_litpcba.json
    results/eval/phase5/c3_vs_litpcba.csv
    results/eval/phase5/c3_vs_litpcba.png  (box plot per target)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, percentileofscore


logger = logging.getLogger("c3_eval")

TARGETS = ["ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1", "GBA",
           "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG",
           "TP53", "VDR"]


def load_litpcba(tar_path: Path) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as tf:
        names = [n for n in tf.getnames() if n.endswith("data.csv")]
        if not names:
            raise SystemExit(f"no data.csv in {tar_path}")
        member = tf.getmember(names[0])
        fh = tf.extractfile(member)
        if fh is None:
            raise SystemExit(f"could not extract {names[0]}")
        return pd.read_csv(io.BytesIO(fh.read()))


def analyse_target(thermofrag_pq: Path, litpcba_col: pd.Series) -> dict | None:
    if not thermofrag_pq.exists():
        return None
    tf_df = pd.read_parquet(thermofrag_pq)
    tf_ok = tf_df[tf_df["status"] == "ok"]
    if tf_ok.empty:
        return None
    tf = tf_ok["vina_score"].to_numpy()
    ref = litpcba_col.to_numpy(dtype=np.float32)
    ref = ref[np.isfinite(ref)]

    top1_cut = np.quantile(ref, 0.01)  # best 1%: most-negative scores
    ref_top1 = ref[ref <= top1_cut]
    tf_top10 = np.sort(tf)[:10]

    mw_stat, mw_p = mannwhitneyu(tf, ref, alternative="less")
    perc = float(percentileofscore(ref, float(np.mean(tf)), kind="mean"))
    return {
        "n_thermofrag":   int(len(tf)),
        "n_litpcba":      int(len(ref)),
        "tf_mean":        float(np.mean(tf)),
        "tf_median":      float(np.median(tf)),
        "tf_top10_mean":  float(np.mean(tf_top10)),
        "ref_mean":       float(np.mean(ref)),
        "ref_median":     float(np.median(ref)),
        "ref_top1_mean":  float(np.mean(ref_top1)),
        "ref_top1_cut":   float(top1_cut),
        "mean_percentile_in_ref": perc,       # lower = more negative (better) than ref
        "mannwhitney_u":  float(mw_stat),
        "mannwhitney_p":  float(mw_p),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--thermofrag-dir", type=Path,
                   default=Path("results/eval/phase4/vina"))
    p.add_argument("--litpcba-tar", type=Path,
                   default=Path("data/external/LIT-PCBA.tar.gz"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase5"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading LIT-PCBA reference library from %s", args.litpcba_tar)
    ref_df = load_litpcba(args.litpcba_tar)
    logger.info("  %d compounds, targets=%s", len(ref_df),
                [c for c in ref_df.columns if c in TARGETS])

    rows = []
    tf_scores_per_target = {}
    ref_scores_per_target = {}
    for t in TARGETS:
        tf_pq = args.thermofrag_dir / f"{t}.parquet"
        if t not in ref_df.columns:
            logger.warning("no LIT-PCBA column for %s — skip", t)
            continue
        result = analyse_target(tf_pq, ref_df[t])
        if result is None:
            logger.info("[%s] no ThermoFrag Vina parquet yet", t)
            continue
        result["target"] = t
        rows.append(result)
        tf_df = pd.read_parquet(tf_pq)
        tf_scores_per_target[t] = tf_df[tf_df["status"] == "ok"]["vina_score"].to_numpy()
        ref_scores_per_target[t] = ref_df[t].to_numpy(dtype=np.float32)
        logger.info(
            "%s: TF mean=%.2f / top10=%.2f | REF mean=%.2f / top1%%=%.2f  MW-p=%.2e  pct=%.2f",
            t, result["tf_mean"], result["tf_top10_mean"],
            result["ref_mean"], result["ref_top1_mean"],
            result["mannwhitney_p"], result["mean_percentile_in_ref"],
        )

    if not rows:
        logger.warning("no ThermoFrag Vina results available yet; aborting")
        return

    df = pd.DataFrame(rows).set_index("target").sort_index()
    csv_path = args.out_dir / "c3_vs_litpcba.csv"
    df.to_csv(csv_path)

    report = {
        "n_targets": int(len(df)),
        "n_targets_tf_beats_ref_mean": int((df["tf_mean"] < df["ref_mean"]).sum()),
        "n_targets_tf_beats_ref_top1": int((df["tf_mean"] < df["ref_top1_mean"]).sum()),
        "n_targets_tf_top10_beats_ref_top1": int((df["tf_top10_mean"] < df["ref_top1_mean"]).sum()),
        "n_targets_significant": int((df["mannwhitney_p"] < 0.01).sum()),
        "summary_mean_diff_kcal_per_mol": float((df["tf_mean"] - df["ref_mean"]).mean()),
        "per_target": df.to_dict(orient="index"),
    }
    report_path = args.out_dir / "c3_vs_litpcba.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("report → %s", report_path)

    # Box plot: for each target, ThermoFrag vs LIT-PCBA reference.
    targets_ready = list(tf_scores_per_target.keys())
    fig, ax = plt.subplots(1, 1, figsize=(max(8, 0.7 * len(targets_ready) + 2), 4.5))
    x = np.arange(len(targets_ready))
    w = 0.35
    tf_data = [tf_scores_per_target[t] for t in targets_ready]
    ref_data = [ref_scores_per_target[t] for t in targets_ready]

    bp1 = ax.boxplot(tf_data, positions=x - w / 2, widths=w * 0.9, patch_artist=True,
                     showfliers=False, boxprops={"facecolor": "#d62728"})
    bp2 = ax.boxplot(ref_data, positions=x + w / 2, widths=w * 0.9, patch_artist=True,
                     showfliers=False, boxprops={"facecolor": "#1f77b4"})
    ax.set_xticks(x)
    ax.set_xticklabels(targets_ready, rotation=45, ha="right")
    ax.set_ylabel("Vina score (kcal/mol)")
    ax.set_title("ThermoFrag (red) vs LIT-PCBA reference library (blue)")
    ax.invert_yaxis()  # more-negative = better on top
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]],
              ["ThermoFrag", "LIT-PCBA library"], loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_dir / "c3_vs_litpcba.png", dpi=150)
    logger.info("fig → %s", args.out_dir / "c3_vs_litpcba.png")


if __name__ == "__main__":
    main()
