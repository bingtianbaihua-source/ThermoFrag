"""Side-by-side comparison of TF-pocket v1 and v2 on the C3/C4 arm.

Reads both phase5_tf_pocket (v1, μ-only) and phase5_tf_pocket_v2
(v2, V^pocket added) results and prints a concise per-target verdict
for C3 and C4 with deltas. Useful for the final memory update.

Usage::

    python scripts/analyze_tf_pocket_v2.py

Prints a markdown-style table to stdout; writes a JSON summary to
``results/eval/phase5_tf_pocket_v2/v1_vs_v2_summary.json``.
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
OUT = Path("results/eval/phase5_tf_pocket_v2/v1_vs_v2_summary.json")

TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]
BASELINES = ["rxnflow", "bbar", "targetdiff"]


def count_sigwins(df: pd.DataFrame, baseline: str, alpha: float = 0.01) -> int:
    sub = df[df["baseline"] == baseline]
    return int(((sub["wilcoxon_p"] < alpha) & (sub["tf_minus_baseline"] < 0)).sum())


def main() -> None:
    if not V2_C3.exists():
        raise SystemExit(f"missing {V2_C3} — run the pipeline first")

    v1_c3 = pd.read_csv(V1_C3)
    v1_c4 = pd.read_csv(V1_C4)
    v2_c3 = pd.read_csv(V2_C3)
    v2_c4 = pd.read_csv(V2_C4)

    print("=" * 78)
    print("TF-pocket v1 (μ-only) vs v2 (μ + V^pocket) — per-baseline sig-wins @ p<0.01")
    print("=" * 78)
    header = f"{'baseline':12s} {'v1_C3_sigwins':>15s} {'v2_C3_sigwins':>15s} {'v1_C4_d<-0.3':>15s} {'v2_C4_d<-0.3':>15s}"
    print(header)
    print("-" * len(header))
    summary = {}
    for b in BASELINES:
        v1_c3_n = count_sigwins(v1_c3, b)
        v2_c3_n = count_sigwins(v2_c3, b)
        v1_c4_n = int((v1_c4[v1_c4["baseline"] == b]["cohens_d"] < -0.3).sum())
        v2_c4_n = int((v2_c4[v2_c4["baseline"] == b]["cohens_d"] < -0.3).sum())
        summary[b] = {
            "v1_c3_sigwins": v1_c3_n,
            "v2_c3_sigwins": v2_c3_n,
            "v1_c4_d_lt_neg0p3": v1_c4_n,
            "v2_c4_d_lt_neg0p3": v2_c4_n,
        }
        print(f"{b:12s} {v1_c3_n:>15d} {v2_c3_n:>15d} {v1_c4_n:>15d} {v2_c4_n:>15d}")

    print()
    print("=" * 78)
    print("Per-target C3 top-10 Vina mean (lower=better): v1 → v2 delta")
    print("=" * 78)
    per_target = {}
    for b in BASELINES:
        print(f"\n-- {b} --")
        v1b = v1_c3[v1_c3["baseline"] == b].set_index("target")
        v2b = v2_c3[v2_c3["baseline"] == b].set_index("target")
        header = f"{'target':12s} {'tf_v1':>8s} {'tf_v2':>8s} {'Δtf':>8s} {'base':>8s} {'v1_p':>8s} {'v2_p':>8s}"
        print(header)
        for t in TARGETS:
            if t not in v1b.index or t not in v2b.index:
                continue
            r1 = v1b.loc[t]
            r2 = v2b.loc[t]
            delta_tf = r2["tf_top10_mean"] - r1["tf_top10_mean"]
            arrow = " ↓" if delta_tf < 0 else " ↑"
            per_target.setdefault(b, {})[t] = {
                "tf_v1": float(r1["tf_top10_mean"]),
                "tf_v2": float(r2["tf_top10_mean"]),
                "delta_tf_v2_minus_v1": float(delta_tf),
                "baseline_top10_mean": float(r1["baseline_top10_mean"]),
                "v1_wilcoxon_p": float(r1["wilcoxon_p"]),
                "v2_wilcoxon_p": float(r2["wilcoxon_p"]),
            }
            print(
                f"{t:12s} {r1['tf_top10_mean']:>8.2f} {r2['tf_top10_mean']:>8.2f} "
                f"{delta_tf:>+7.2f}{arrow} {r1['baseline_top10_mean']:>8.2f} "
                f"{r1['wilcoxon_p']:>8.3g} {r2['wilcoxon_p']:>8.3g}"
            )

    summary["per_target_c3"] = per_target
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote] {OUT}")


if __name__ == "__main__":
    main()
