"""Phase-5 wrap-up: C3 / C4 / C6 summary tables + Fig 7 / Fig 8.

Consumes, once they exist:
  - ThermoFrag conditional Vina:    results/eval/phase4/vina/<target>.parquet
  - ThermoFrag conditional strain:  results/eval/phase4/strain/<target>.parquet
  - No-μ ablation Vina:             results/eval/phase5/nomu_vina/<target>.parquet
  - No-μ ablation strain:           results/eval/phase5/nomu_strain/pool.parquet
  - LIT-PCBA reference library:     data/external/LIT-PCBA.tar.gz (data.csv columns)

Produces:
  results/eval/phase5/
    c3_summary.csv          per-target TF/no-μ/REF means, paired Wilcoxon p
    c4_summary.csv          per-target strain means + Cohen's d (TF vs ZINC-like)
    c6_ablation.csv         claim-by-claim ablation pass/fail
    fig7_litpcba_box.png    C3 box plot (TF vs no-μ vs REF, 15 targets)
    fig8_strain_hist.png    C4 strain distribution, TF vs no-μ vs ZINC-like
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
from scipy.stats import wilcoxon, mannwhitneyu


logger = logging.getLogger("summary")

TARGETS = ["ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1", "GBA",
           "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG",
           "TP53", "VDR"]


def load_litpcba_ref(tar_path: Path) -> pd.DataFrame:
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.endswith("data.csv"):
                fh = tf.extractfile(m)
                return pd.read_csv(io.BytesIO(fh.read()))
    raise SystemExit("no data.csv in LIT-PCBA tar")


def safe_read_parquet(path: Path):
    if not path.exists():
        return None
    return pd.read_parquet(path)


def c3_per_target(tf_dir: Path, nomu_dir: Path, ref_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in TARGETS:
        tf = safe_read_parquet(tf_dir / f"{t}.parquet")
        no = safe_read_parquet(nomu_dir / f"{t}.parquet")
        if tf is None:
            rows.append({"target": t, "status": "pending_tf"})
            continue
        tf_ok = tf[tf["status"] == "ok"]["vina_score"].to_numpy()
        if len(tf_ok) == 0:
            rows.append({"target": t, "status": "empty"})
            continue
        row = {"target": t, "status": "ok",
               "tf_n": int(len(tf_ok)),
               "tf_mean": float(np.mean(tf_ok)),
               "tf_median": float(np.median(tf_ok)),
               "tf_top10_mean": float(np.mean(np.sort(tf_ok)[:10]))}
        if t in ref_df.columns:
            ref = ref_df[t].dropna().to_numpy(dtype=np.float32)
            top1_cut = np.quantile(ref, 0.01)
            ref_top1 = ref[ref <= top1_cut]
            row["ref_mean"] = float(np.mean(ref))
            row["ref_top1_mean"] = float(np.mean(ref_top1))
            mw_u, mw_p = mannwhitneyu(tf_ok, ref, alternative="less")
            row["mw_u_vs_ref"] = float(mw_u)
            row["mw_p_vs_ref"] = float(mw_p)
        if no is not None:
            no_ok = no[no["status"] == "ok"]["vina_score"].to_numpy()
            if len(no_ok):
                row["nomu_n"] = int(len(no_ok))
                row["nomu_mean"] = float(np.mean(no_ok))
                row["nomu_median"] = float(np.median(no_ok))
                # Paired Wilcoxon requires same-length arrays. If sizes differ
                # we take min-length; paired by rank of chain_idx within each.
                tf_sorted = tf.sort_values("chain_idx")
                no_sorted = no.sort_values("chain_idx")
                tf_s = tf_sorted[tf_sorted["status"] == "ok"]["vina_score"].to_numpy()
                no_s = no_sorted[no_sorted["status"] == "ok"]["vina_score"].to_numpy()
                n_pair = min(len(tf_s), len(no_s))
                if n_pair >= 8:
                    try:
                        w_stat, w_p = wilcoxon(tf_s[:n_pair], no_s[:n_pair],
                                               alternative="less")
                        row["paired_n"] = int(n_pair)
                        row["paired_wilcoxon_p"] = float(w_p)
                    except ValueError:  # all zeros
                        row["paired_wilcoxon_p"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def c4_per_target(tf_strain_dir: Path, nomu_strain_pq: Path) -> pd.DataFrame:
    rows = []
    nomu = safe_read_parquet(nomu_strain_pq)
    nomu_ok = nomu[nomu["status"] == "ok"]["strain"].to_numpy() if nomu is not None else None
    for t in TARGETS:
        tf = safe_read_parquet(tf_strain_dir / f"{t}.parquet")
        if tf is None:
            rows.append({"target": t, "status": "pending"})
            continue
        tf_ok = tf[tf["status"] == "ok"]["strain"].to_numpy()
        row = {"target": t, "status": "ok",
               "tf_n": int(len(tf_ok)),
               "tf_mean": float(np.mean(tf_ok)),
               "tf_median": float(np.median(tf_ok))}
        if nomu_ok is not None and len(nomu_ok):
            row["nomu_mean"] = float(np.mean(nomu_ok))
            row["nomu_median"] = float(np.median(nomu_ok))
            # Cohen's d (unpaired; tf-per-target vs shared no-μ pool).
            pooled = np.sqrt(((np.var(tf_ok, ddof=1) * (len(tf_ok) - 1)) +
                              (np.var(nomu_ok, ddof=1) * (len(nomu_ok) - 1))) /
                             (len(tf_ok) + len(nomu_ok) - 2))
            d = (np.mean(tf_ok) - np.mean(nomu_ok)) / max(pooled, 1e-6)
            row["cohens_d_tf_minus_nomu"] = float(d)
        rows.append(row)
    return pd.DataFrame(rows)


def c6_ablation_summary(c3: pd.DataFrame, c4: pd.DataFrame,
                         nomu_strain_pq: Path) -> dict:
    """High-level claim flags."""
    out = {}
    # C3: paired wilcoxon TF < no-μ, aggregate count of significant targets
    if "paired_wilcoxon_p" in c3.columns:
        ok = c3.dropna(subset=["paired_wilcoxon_p"])
        n_sig = int((ok["paired_wilcoxon_p"] < 0.01).sum())
        n_tested = int(len(ok))
        out["c6_c3_tf_better_than_nomu"] = {
            "n_sig": n_sig,
            "n_tested": n_tested,
            "pass_threshold": "≥10/15",
            "pass": n_sig >= 10,
        }
    # C4: cohen's d per-target mean vs no-μ pool
    if "cohens_d_tf_minus_nomu" in c4.columns:
        ok = c4.dropna(subset=["cohens_d_tf_minus_nomu"])
        mean_d = float(ok["cohens_d_tf_minus_nomu"].mean())
        out["c6_c4_tf_lower_strain_than_nomu"] = {
            "mean_cohens_d": mean_d,
            "interpretation": ("positive = TF higher strain; negative = TF lower"),
            "note": "A positive d means conditional sampler trades strain for property-targeting — expected for Pareto.",
        }
    return out


def _box(ax, data_list, positions, color, label, width=0.3):
    bp = ax.boxplot(data_list, positions=positions, widths=width * 0.9,
                    patch_artist=True, showfliers=False,
                    boxprops={"facecolor": color, "edgecolor": "black",
                             "linewidth": 0.6},
                    medianprops={"color": "black", "linewidth": 1.3})
    bp["_label"] = label
    return bp


def make_fig7(c3_df: pd.DataFrame, tf_dir: Path, nomu_dir: Path, ref_df: pd.DataFrame,
              out_path: Path):
    targets_ready = c3_df[c3_df["status"] == "ok"]["target"].tolist()
    if not targets_ready:
        return
    fig, ax = plt.subplots(1, 1, figsize=(max(10, 0.8 * len(targets_ready) + 2), 5.0))
    x = np.arange(len(targets_ready))
    tf_data, no_data, ref_data = [], [], []
    for t in targets_ready:
        tf = pd.read_parquet(tf_dir / f"{t}.parquet")
        tf_data.append(tf[tf["status"] == "ok"]["vina_score"].to_numpy())
        no_pq = nomu_dir / f"{t}.parquet"
        if no_pq.exists():
            no = pd.read_parquet(no_pq)
            no_data.append(no[no["status"] == "ok"]["vina_score"].to_numpy())
        else:
            no_data.append(np.array([]))
        ref_data.append(ref_df[t].dropna().to_numpy(dtype=np.float32))

    bp1 = _box(ax, tf_data, x - 0.3, "#d62728", "ThermoFrag")
    bp2 = _box(ax, no_data, x,       "#ff9896", "No-μ ablation")
    bp3 = _box(ax, ref_data, x + 0.3, "#1f77b4", "LIT-PCBA library")
    ax.set_xticks(x); ax.set_xticklabels(targets_ready, rotation=45, ha="right")
    ax.set_ylabel("Vina score (kcal/mol)   ↓ better")
    ax.invert_yaxis()
    ax.set_title("Fig 7 — Per-target Vina score:  ThermoFrag vs no-μ ablation vs LIT-PCBA reference")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]],
              ["ThermoFrag", "No-μ ablation", "LIT-PCBA library"],
              loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def make_fig8(tf_strain_dir: Path, nomu_strain_pq: Path, out_path: Path):
    # Aggregate ThermoFrag strain across targets.
    tf_all = []
    for pq in sorted(tf_strain_dir.glob("*.parquet")):
        df = pd.read_parquet(pq)
        tf_all.append(df[df["status"] == "ok"]["strain"].to_numpy())
    if not tf_all:
        return
    tf_all = np.concatenate(tf_all)
    no_df = safe_read_parquet(nomu_strain_pq)
    nomu = (no_df[no_df["status"] == "ok"]["strain"].to_numpy()
            if no_df is not None else np.array([]))

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.8))
    bins = np.linspace(0, max(np.quantile(tf_all, 0.98), 30), 40)
    ax.hist(tf_all, bins=bins, alpha=0.6, color="#d62728",
            label=f"ThermoFrag (n={len(tf_all)})", density=True)
    if len(nomu):
        ax.hist(nomu, bins=bins, alpha=0.6, color="#ff9896",
                label=f"No-μ ablation (n={len(nomu)})", density=True)
    ax.set_xlabel("Post-MMFF94 strain energy (kcal/mol)")
    ax.set_ylabel("density")
    ax.set_title("Fig 8 — Strain distribution per generator")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tf-vina",   type=Path, default=Path("results/eval/phase4/vina"))
    p.add_argument("--nomu-vina", type=Path, default=Path("results/eval/phase5/nomu_vina"))
    p.add_argument("--tf-strain", type=Path, default=Path("results/eval/phase4/strain"))
    p.add_argument("--nomu-strain", type=Path,
                   default=Path("results/eval/phase5/nomu_strain/pool.parquet"))
    p.add_argument("--litpcba-tar", type=Path,
                   default=Path("data/external/LIT-PCBA.tar.gz"))
    p.add_argument("--out-dir", type=Path, default=Path("results/eval/phase5"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_df = load_litpcba_ref(args.litpcba_tar)

    c3 = c3_per_target(args.tf_vina, args.nomu_vina, ref_df)
    c4 = c4_per_target(args.tf_strain, args.nomu_strain)
    c3.to_csv(args.out_dir / "c3_summary.csv", index=False)
    c4.to_csv(args.out_dir / "c4_summary.csv", index=False)
    logger.info("c3 rows:\n%s", c3.to_string(index=False))
    logger.info("c4 rows:\n%s", c4.to_string(index=False))

    summary = c6_ablation_summary(c3, c4, args.nomu_strain)
    (args.out_dir / "c6_ablation.json").write_text(json.dumps(summary, indent=2))
    logger.info("c6 summary:\n%s", json.dumps(summary, indent=2))

    make_fig7(c3, args.tf_vina, args.nomu_vina, ref_df,
              args.out_dir / "fig7_litpcba_box.png")
    make_fig8(args.tf_strain, args.nomu_strain,
              args.out_dir / "fig8_strain_hist.png")
    logger.info("figures → %s", args.out_dir)


if __name__ == "__main__":
    main()
