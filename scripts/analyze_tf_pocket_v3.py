"""Three-way comparison TF-pocket v1 / v2 / v3 on the C3/C4 arm.

Extends ``analyze_tf_pocket_v2.py`` to include TF-pocket v3 (EGNN-over-Cα
pocket encoder). Reads all three phase5_* directories and prints a
per-baseline sig-wins table + a per-target delta table for C3 top-10
Vina. Writes a JSON summary to
``results/eval/phase5_tf_pocket_v3/v1_v2_v3_summary.json``.

Usage::

    python scripts/analyze_tf_pocket_v3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


V1_C3 = Path("results/eval/phase5_tf_pocket/c3_vs_generators.csv")
V1_C4 = Path("results/eval/phase5_tf_pocket/c4_vs_generators.csv")
V2_C3 = Path("results/eval/phase5_tf_pocket_v2/c3_vs_generators.csv")
V2_C4 = Path("results/eval/phase5_tf_pocket_v2/c4_vs_generators.csv")
V3_C3 = Path("results/eval/phase5_tf_pocket_v3/c3_vs_generators.csv")
V3_C4 = Path("results/eval/phase5_tf_pocket_v3/c4_vs_generators.csv")
OUT = Path("results/eval/phase5_tf_pocket_v3/v1_v2_v3_summary.json")

TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]
BASELINES = ["rxnflow", "bbar", "targetdiff"]


def count_sigwins(df: pd.DataFrame, baseline: str, alpha: float = 0.01) -> int:
    sub = df[df["baseline"] == baseline]
    return int(((sub["wilcoxon_p"] < alpha) & (sub["tf_minus_baseline"] < 0)).sum())


def count_strain_wins(df: pd.DataFrame, baseline: str, thresh: float = -0.3) -> int:
    return int((df[df["baseline"] == baseline]["cohens_d"] < thresh).sum())


def main() -> None:
    for p in (V1_C3, V1_C4, V2_C3, V2_C4, V3_C3, V3_C4):
        if not p.exists():
            raise SystemExit(f"missing {p} — run the pipeline first")

    frames = {
        "v1_c3": pd.read_csv(V1_C3), "v1_c4": pd.read_csv(V1_C4),
        "v2_c3": pd.read_csv(V2_C3), "v2_c4": pd.read_csv(V2_C4),
        "v3_c3": pd.read_csv(V3_C3), "v3_c4": pd.read_csv(V3_C4),
    }

    print("=" * 88)
    print("TF-pocket v1 / v2 / v3 — per-baseline sig-wins @ p<0.01  |  C4 |d|<-0.3")
    print("=" * 88)
    header = (
        f"{'baseline':12s}"
        f" {'v1_C3':>6s} {'v2_C3':>6s} {'v3_C3':>6s}"
        f" {'v1_C4':>6s} {'v2_C4':>6s} {'v3_C4':>6s}"
    )
    print(header)
    print("-" * len(header))
    summary: dict = {}
    for b in BASELINES:
        row = {
            "v1_c3_sigwins": count_sigwins(frames["v1_c3"], b),
            "v2_c3_sigwins": count_sigwins(frames["v2_c3"], b),
            "v3_c3_sigwins": count_sigwins(frames["v3_c3"], b),
            "v1_c4_d_lt_neg0p3": count_strain_wins(frames["v1_c4"], b),
            "v2_c4_d_lt_neg0p3": count_strain_wins(frames["v2_c4"], b),
            "v3_c4_d_lt_neg0p3": count_strain_wins(frames["v3_c4"], b),
        }
        summary[b] = row
        print(
            f"{b:12s}"
            f" {row['v1_c3_sigwins']:>6d} {row['v2_c3_sigwins']:>6d} {row['v3_c3_sigwins']:>6d}"
            f" {row['v1_c4_d_lt_neg0p3']:>6d} {row['v2_c4_d_lt_neg0p3']:>6d} {row['v3_c4_d_lt_neg0p3']:>6d}"
        )

    # Per-target v3 vs v1 deltas on C3 top-10 (most informative).
    print()
    print("=" * 88)
    print("Per-target C3 top-10 Vina mean (lower=better): v1 vs v3 deltas")
    print("=" * 88)
    per_target: dict = {}
    for b in BASELINES:
        print(f"\n-- {b} --")
        v1b = frames["v1_c3"][frames["v1_c3"]["baseline"] == b].set_index("target")
        v3b = frames["v3_c3"][frames["v3_c3"]["baseline"] == b].set_index("target")
        h = f"{'target':12s} {'tf_v1':>8s} {'tf_v3':>8s} {'Δv3-v1':>8s} {'base':>8s} {'v1_p':>8s} {'v3_p':>8s}"
        print(h)
        for t in TARGETS:
            if t not in v1b.index or t not in v3b.index:
                continue
            r1 = v1b.loc[t]
            r3 = v3b.loc[t]
            delta = r3["tf_top10_mean"] - r1["tf_top10_mean"]
            arrow = " ↓" if delta < 0 else " ↑"
            per_target.setdefault(b, {})[t] = {
                "tf_v1": float(r1["tf_top10_mean"]),
                "tf_v3": float(r3["tf_top10_mean"]),
                "delta_tf_v3_minus_v1": float(delta),
                "baseline_top10_mean": float(r1["baseline_top10_mean"]),
                "v1_wilcoxon_p": float(r1["wilcoxon_p"]),
                "v3_wilcoxon_p": float(r3["wilcoxon_p"]),
            }
            print(
                f"{t:12s} {r1['tf_top10_mean']:>8.2f} {r3['tf_top10_mean']:>8.2f} "
                f"{delta:>+7.2f}{arrow} {r1['baseline_top10_mean']:>8.2f} "
                f"{r1['wilcoxon_p']:>8.3g} {r3['wilcoxon_p']:>8.3g}"
            )

    # Per-baseline mean top-10 gap (tf - baseline).
    print()
    print("=" * 88)
    print("Mean top-10 Vina gap (tf - baseline) across 15 targets, lower = TF better")
    print("=" * 88)
    print(f"{'baseline':12s} {'v1_gap':>10s} {'v2_gap':>10s} {'v3_gap':>10s}")
    for b in BASELINES:
        g1 = float(frames["v1_c3"][frames["v1_c3"]["baseline"] == b]["tf_minus_baseline"].mean())
        g2 = float(frames["v2_c3"][frames["v2_c3"]["baseline"] == b]["tf_minus_baseline"].mean())
        g3 = float(frames["v3_c3"][frames["v3_c3"]["baseline"] == b]["tf_minus_baseline"].mean())
        summary[b]["v1_mean_gap"] = g1
        summary[b]["v2_mean_gap"] = g2
        summary[b]["v3_mean_gap"] = g3
        print(f"{b:12s} {g1:>+10.3f} {g2:>+10.3f} {g3:>+10.3f}")

    summary["per_target_c3_v1_vs_v3"] = per_target
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
