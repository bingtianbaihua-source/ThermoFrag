"""Paired generator-vs-generator stats for C3 (Vina) and C4 (OpenMM strain).

Consumes (per BASELINES.md):

- ThermoFrag:   results/eval/phase4/{vina,strain}/<target>.parquet
- Baseline b:   results/eval/phase4_baselines/<b>/{vina,strain}/<target>.parquet

For each (target, baseline) pair, on the ``status == 'ok'`` rows:

C3 (Vina, most-negative = best):
    * Rank-sort each pool ascending by vina_score.
    * Take the top-10 of each.
    * Wilcoxon signed-rank between the paired top-10 sequences (TF top-k vs b top-k).
    * Report tf_top10_mean, baseline_top10_mean, tf_minus_baseline, wilcoxon_p.

C4 (strain, lower = less strain):
    * On status='ok' rows, compute mean strain per pool.
    * Cohen's d for paired effect size.
    * Report tf_mean_strain, baseline_mean_strain, cohens_d.

Outputs::

    results/eval/phase5/c3_vs_generators.csv
    results/eval/phase5/c4_vs_generators.csv
    results/eval/phase5/c3_c4_bars.png

Acceptance (from BASELINES.md):
    * C3: TF beats each baseline on ≥10/15 targets at paired Wilcoxon p<0.01.
    * C4: Cohen's d > 0.3 lower strain for TF.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("eval_gvg")

DEFAULT_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA",   "IDH1",  "KAT2A",   "MAPK1",     "MTORC1",
    "OPRK1", "PKM2",  "PPARG",   "TP53",      "VDR",
]
DEFAULT_BASELINES = ["rxnflow", "bbar", "targetdiff"]


def _load_ok(pq: Path, score_col: str,
             max_pool: Optional[int] = None) -> Optional[pd.Series]:
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    # Pool-size fairness: restrict to the first ``max_pool`` chain indices,
    # so top-k is drawn from a pool comparable to TF's ~100-160 docked ligands.
    if max_pool is not None and "chain_idx" in df.columns:
        df = df[df["chain_idx"] < max_pool]
    ok = df[(df["status"] == "ok") & df[score_col].notna()]
    if len(ok) == 0:
        return None
    return ok[score_col].reset_index(drop=True)


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    from scipy.stats import wilcoxon
    try:
        stat = wilcoxon(a, b, zero_method="wilcox", alternative="less")
        return float(stat.pvalue)
    except Exception:
        return None


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    # Paired Cohen's d on the mean-difference.  a - b; negative = a smaller.
    if len(a) == 0 or len(b) == 0:
        return None
    sa, sb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = (sa + sb) / 2.0
    if pooled <= 0:
        return None
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def compute_c3(tf_root: Path, base_root: Path, target: str, k: int,
               max_baseline_pool: Optional[int] = None) -> dict:
    tf = _load_ok(tf_root / f"{target}.parquet", "vina_score")
    bl = _load_ok(base_root / f"{target}.parquet", "vina_score",
                  max_pool=max_baseline_pool)
    out = {"target": target}
    if tf is None or bl is None:
        out.update({
            "tf_n_ok": 0 if tf is None else int(len(tf)),
            "baseline_n_ok": 0 if bl is None else int(len(bl)),
            "tf_top10_mean": None, "baseline_top10_mean": None,
            "tf_minus_baseline": None, "wilcoxon_p": None,
        })
        return out
    tf_top = np.sort(tf.to_numpy())[:k]
    bl_top = np.sort(bl.to_numpy())[:k]
    pad = min(len(tf_top), len(bl_top))
    if pad < 3:
        p = None
    else:
        p = _wilcoxon(tf_top[:pad], bl_top[:pad])
    out.update({
        "tf_n_ok": int(len(tf)),
        "baseline_n_ok": int(len(bl)),
        "tf_top10_mean": float(np.mean(tf_top)),
        "baseline_top10_mean": float(np.mean(bl_top)),
        "tf_minus_baseline": float(np.mean(tf_top) - np.mean(bl_top)),
        "wilcoxon_p": p,
    })
    return out


def compute_c4(tf_root: Path, base_root: Path, target: str,
               max_baseline_pool: Optional[int] = None) -> dict:
    tf = _load_ok(tf_root / f"{target}.parquet", "strain")
    bl = _load_ok(base_root / f"{target}.parquet", "strain",
                  max_pool=max_baseline_pool)
    out = {"target": target}
    if tf is None or bl is None:
        out.update({
            "tf_n_ok": 0 if tf is None else int(len(tf)),
            "baseline_n_ok": 0 if bl is None else int(len(bl)),
            "tf_mean_strain": None, "baseline_mean_strain": None,
            "cohens_d": None,
        })
        return out
    a = tf.to_numpy()
    b = bl.to_numpy()
    out.update({
        "tf_n_ok": int(len(a)),
        "baseline_n_ok": int(len(b)),
        "tf_mean_strain": float(np.mean(a)),
        "baseline_mean_strain": float(np.mean(b)),
        "cohens_d": _cohens_d(a, b),
    })
    return out


def plot_bars(c3: pd.DataFrame, c4: pd.DataFrame, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baselines = sorted(c3["baseline"].unique())
    targets = sorted(c3["target"].unique())
    x = np.arange(len(targets))
    width = 0.8 / (1 + len(baselines))

    # Explicit palette so baselines don't collide with TF's tab:blue.
    baseline_colors = {
        "bbar": "tab:orange",
        "rxnflow": "tab:green",
        "targetdiff": "tab:red",
    }

    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.6 * len(targets)), 7),
                             sharex=True)

    # C3: top-10 Vina mean
    tf_vals = []
    for t in targets:
        rows = c3[c3["target"] == t]
        if len(rows):
            tf_vals.append(rows["tf_top10_mean"].iloc[0])
        else:
            tf_vals.append(np.nan)
    axes[0].bar(x - 0.8 / 2 + width / 2, tf_vals, width, label="ThermoFrag",
                color="tab:blue")
    for i, b in enumerate(baselines):
        vals = []
        for t in targets:
            row = c3[(c3["target"] == t) & (c3["baseline"] == b)]
            vals.append(row["baseline_top10_mean"].iloc[0] if len(row) else np.nan)
        axes[0].bar(x - 0.8 / 2 + width / 2 + (i + 1) * width, vals, width,
                    label=b, color=baseline_colors.get(b))
    axes[0].set_ylabel("top-10 Vina mean (kcal/mol, lower=better)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].legend()

    # C4: mean strain
    tf_vals = []
    for t in targets:
        rows = c4[c4["target"] == t]
        if len(rows):
            tf_vals.append(rows["tf_mean_strain"].iloc[0])
        else:
            tf_vals.append(np.nan)
    axes[1].bar(x - 0.8 / 2 + width / 2, tf_vals, width, label="ThermoFrag",
                color="tab:blue")
    for i, b in enumerate(baselines):
        vals = []
        for t in targets:
            row = c4[(c4["target"] == t) & (c4["baseline"] == b)]
            vals.append(row["baseline_mean_strain"].iloc[0] if len(row) else np.nan)
        axes[1].bar(x - 0.8 / 2 + width / 2 + (i + 1) * width, vals, width,
                    label=b, color=baseline_colors.get(b))
    axes[1].set_ylabel("mean GAFF strain (kcal/mol, lower=better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(targets, rotation=35, ha="right")
    axes[1].legend()

    fig.suptitle("ThermoFrag vs generator baselines (C3 top-10 Vina, C4 strain)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tf-root", type=Path,
                   default=Path("results/eval/phase4"))
    p.add_argument("--baselines-root", type=Path,
                   default=Path("results/eval/phase4_baselines"))
    p.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES)
    p.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase5"))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-baseline-pool", type=int, default=None,
                   help="If set, restrict each baseline's per-target pool to "
                        "the first N chain indices before computing top-k. "
                        "Used to match baseline pool size to TF's ~100-160 "
                        "docked ligands for fair top-k comparison.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    c3_rows, c4_rows = [], []
    for b in args.baselines:
        b_root = args.baselines_root / b
        if not b_root.exists():
            logger.warning("baseline %s missing at %s — skip", b, b_root)
            continue
        for t in args.targets:
            r3 = compute_c3(args.tf_root / "vina", b_root / "vina", t, k=args.top_k,
                            max_baseline_pool=args.max_baseline_pool)
            r3["baseline"] = b
            c3_rows.append(r3)
            r4 = compute_c4(args.tf_root / "strain", b_root / "strain", t,
                            max_baseline_pool=args.max_baseline_pool)
            r4["baseline"] = b
            c4_rows.append(r4)

    c3 = pd.DataFrame(c3_rows)
    c4 = pd.DataFrame(c4_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    c3.to_csv(args.out_dir / "c3_vs_generators.csv", index=False)
    c4.to_csv(args.out_dir / "c4_vs_generators.csv", index=False)
    logger.info("wrote %s + %s",
                args.out_dir / "c3_vs_generators.csv",
                args.out_dir / "c4_vs_generators.csv")

    # Summary against acceptance thresholds
    summary = {}
    for b in args.baselines:
        sub3 = c3[(c3["baseline"] == b) & c3["wilcoxon_p"].notna()]
        n_sig = int(((sub3["wilcoxon_p"] < 0.01) &
                     (sub3["tf_minus_baseline"] < 0)).sum())
        sub4 = c4[(c4["baseline"] == b) & c4["cohens_d"].notna()]
        # Cohen's d < -0.3 means TF strain is lower
        c4_pass = int((sub4["cohens_d"] < -0.3).sum())
        summary[b] = {
            "c3_targets_sig_p<0.01_and_tf_better": n_sig,
            "c3_pass_threshold_10_of_15": n_sig >= 10,
            "c4_targets_d<-0.3": c4_pass,
            "c4_mean_d": (None if len(sub4) == 0 else float(sub4["cohens_d"].mean())),
        }
    (args.out_dir / "c3_c4_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("summary: %s", json.dumps(summary, indent=2))

    if len(c3) and len(c4):
        plot_bars(c3, c4, args.out_dir / "c3_c4_bars.png")
        logger.info("figure → %s", args.out_dir / "c3_c4_bars.png")


if __name__ == "__main__":
    main()
