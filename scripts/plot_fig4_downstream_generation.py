#!/usr/bin/env python
"""Build Fig. 4: downstream Vina evidence for generated molecules.

The figure uses existing Phase-5 summary tables. It does not re-run
docking. A JSON manifest is written beside the figure so every panel has
an explicit data source.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "grey": "#999999",
    "black": "#000000",
}

BASELINE_LABELS = {
    "targetdiff": "TargetDiff",
    "rxnflow": "RxnFlow",
    "bbar": "BBAR",
    "litpcba_ref": "LIT-PCBA ref.",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
        va="top",
        ha="left",
    )


def sig_wins(df: pd.DataFrame) -> dict[str, int]:
    out = {}
    for baseline, group in df.groupby("baseline"):
        out[baseline] = int(((group["tf_minus_baseline"] < 0) & (group["wilcoxon_p"] < 0.01)).sum())
    return out


def draw_panel_a(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.set_title("matched Vina benchmark", pad=8)
    boxes = [
        (0.04, 0.64, 0.24, 0.20, "15 LIT-PCBA\ntargets"),
        (0.38, 0.64, 0.24, 0.20, "pool cap\n100 ligands"),
        (0.72, 0.64, 0.24, 0.20, "AutoDock Vina\ntop-10"),
        (0.22, 0.20, 0.24, 0.20, "rank-paired\nWilcoxon"),
        (0.58, 0.20, 0.24, 0.20, "two-tier\ncomparison"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#f3f4f6", edgecolor="#555555", lw=0.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, linespacing=1.2)
    arrows = [
        ((0.28, 0.74), (0.38, 0.74)),
        ((0.62, 0.74), (0.72, 0.74)),
        ((0.84, 0.64), (0.70, 0.40)),
        ((0.46, 0.30), (0.58, 0.30)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
    ax.text(
        0.5,
        0.03,
        "Structure-only: TargetDiff and reference library; score-aware: RxnFlow and BBAR",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    panel_label(ax, "a")


def draw_summary(ax: plt.Axes, gen_df: pd.DataFrame, lit_df: pd.DataFrame, summary: dict) -> None:
    wins = sig_wins(gen_df)
    lit_wins = int(((lit_df["tf_mean"] < lit_df["ref_mean"]) & (lit_df["mannwhitney_p"] < 0.01)).sum())
    rows = [
        ("TargetDiff", wins["targetdiff"], gen_df.loc[gen_df["baseline"] == "targetdiff", "tf_minus_baseline"].mean()),
        ("LIT-PCBA ref.", lit_wins, float(summary["summary_mean_diff_kcal_per_mol"])),
        ("RxnFlow", wins["rxnflow"], gen_df.loc[gen_df["baseline"] == "rxnflow", "tf_minus_baseline"].mean()),
        ("BBAR", wins["bbar"], gen_df.loc[gen_df["baseline"] == "bbar", "tf_minus_baseline"].mean()),
    ]
    labels = [r[0] for r in rows]
    win_vals = [r[1] for r in rows]
    gap_vals = [r[2] for r in rows]
    colors = [OKABE["blue"], OKABE["green"], OKABE["orange"], OKABE["purple"]]

    y = np.arange(len(rows))
    ax.barh(y, win_vals, color=colors, alpha=0.9)
    ax.axvline(10, color=OKABE["black"], lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 15)
    ax.invert_yaxis()
    ax.set_xlabel("significant ThermoFrag wins (/15)")
    ax.set_title("pre-registered success threshold")
    ax.text(10.15, -0.55, "threshold", fontsize=7, va="center")
    for yi, wins_i, gap in zip(y, win_vals, gap_vals):
        ax.text(wins_i + 0.25, yi, f"{wins_i}/15, mean gap {gap:+.2f}", va="center", fontsize=7)
    panel_label(ax, "b")


def draw_targetdiff(ax: plt.Axes, gen_df: pd.DataFrame) -> None:
    sub = gen_df[gen_df["baseline"] == "targetdiff"].copy()
    sub = sub.sort_values("tf_minus_baseline")
    colors = np.where(sub["tf_minus_baseline"] < 0, OKABE["blue"], OKABE["grey"])
    ax.bar(np.arange(len(sub)), sub["tf_minus_baseline"], color=colors)
    ax.axhline(0, color=OKABE["black"], lw=0.8)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["target"], rotation=60, ha="right")
    ax.set_ylabel("TF - TargetDiff\nVina top-10 mean\n(kcal mol$^{-1}$)")
    ax.set_title("structure-only baseline")
    ax.text(0.02, 0.05, "14/15 significant wins", transform=ax.transAxes, fontsize=7)
    panel_label(ax, "c")


def draw_litpcba(ax: plt.Axes, lit_df: pd.DataFrame) -> None:
    sub = lit_df.copy()
    sub["delta"] = sub["tf_mean"] - sub["ref_mean"]
    sub = sub.sort_values("delta")
    colors = np.where(sub["delta"] < 0, OKABE["green"], OKABE["grey"])
    ax.bar(np.arange(len(sub)), sub["delta"], color=colors)
    ax.axhline(0, color=OKABE["black"], lw=0.8)
    ax.set_xticks(np.arange(len(sub)))
    ax.set_xticklabels(sub["target"], rotation=60, ha="right")
    ax.set_ylabel("TF mean - reference mean\nVina score (kcal mol$^{-1}$)")
    ax.set_title("reference-library enrichment")
    ax.text(0.02, 0.05, "14/15 significant mean shifts", transform=ax.transAxes, fontsize=7)
    panel_label(ax, "d")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generators",
        type=Path,
        default=Path("results/eval/phase5/c3_vs_generators.csv"),
    )
    parser.add_argument(
        "--litpcba",
        type=Path,
        default=Path("results/eval/phase5/c3_vs_litpcba.csv"),
    )
    parser.add_argument(
        "--litpcba-summary",
        type=Path,
        default=Path("results/eval/phase5/c3_vs_litpcba.json"),
    )
    parser.add_argument(
        "--c3c4-summary",
        type=Path,
        default=Path("results/eval/phase5/c3_c4_summary.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/figures/paper/fig4_downstream_generation.png"))
    args = parser.parse_args()

    configure_style()
    gen_df = pd.read_csv(args.generators)
    lit_df = pd.read_csv(args.litpcba)
    lit_summary = json.loads(args.litpcba_summary.read_text())
    c3c4_summary = json.loads(args.c3c4_summary.read_text())

    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(2, 2, wspace=0.42, hspace=0.58)
    draw_panel_a(fig.add_subplot(gs[0, 0]))
    draw_summary(fig.add_subplot(gs[0, 1]), gen_df, lit_df, lit_summary)
    draw_targetdiff(fig.add_subplot(gs[1, 0]), gen_df)
    draw_litpcba(fig.add_subplot(gs[1, 1]), lit_df)
    fig.suptitle("Downstream docking evidence without docking-score supervision", y=0.995, fontsize=10)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    manifest = {
        "figure": "Fig. 4",
        "script": str(Path(__file__).as_posix()),
        "outputs": [str(args.out), str(args.out.with_suffix(".pdf"))],
        "panels": {
            "a": {
                "description": "Benchmark design schematic: 15 LIT-PCBA targets, pool cap 100, Vina top-10, one-sided rank-paired Wilcoxon.",
                "data_sources": [
                    str(args.generators),
                    str(args.litpcba),
                    "paper/04_methods.md",
                ],
            },
            "b": {
                "description": "Significant ThermoFrag wins by comparison arm, with mean Vina gaps.",
                "data_sources": [
                    str(args.generators),
                    str(args.litpcba),
                    str(args.litpcba_summary),
                    str(args.c3c4_summary),
                ],
            },
            "c": {
                "description": "Per-target Vina top-10 mean gap against TargetDiff.",
                "data_sources": [str(args.generators)],
            },
            "d": {
                "description": "Per-target mean Vina score shift against the LIT-PCBA reference library.",
                "data_sources": [str(args.litpcba), str(args.litpcba_summary)],
            },
        },
        "summary_values": {
            "targetdiff_sig_wins": int(
                (
                    (gen_df.loc[gen_df["baseline"] == "targetdiff", "tf_minus_baseline"] < 0)
                    & (gen_df.loc[gen_df["baseline"] == "targetdiff", "wilcoxon_p"] < 0.01)
                ).sum()
            ),
            "targetdiff_mean_gap": float(
                gen_df.loc[gen_df["baseline"] == "targetdiff", "tf_minus_baseline"].mean()
            ),
            "litpcba_sig_wins": int(
                ((lit_df["tf_mean"] < lit_df["ref_mean"]) & (lit_df["mannwhitney_p"] < 0.01)).sum()
            ),
            "litpcba_mean_gap": float(lit_summary["summary_mean_diff_kcal_per_mol"]),
            "rxnflow_sig_wins": int(c3c4_summary["rxnflow"]["c3_targets_sig_p<0.01_and_tf_better"]),
            "rxnflow_mean_gap": float(
                gen_df.loc[gen_df["baseline"] == "rxnflow", "tf_minus_baseline"].mean()
            ),
            "bbar_sig_wins": int(c3c4_summary["bbar"]["c3_targets_sig_p<0.01_and_tf_better"]),
            "bbar_mean_gap": float(gen_df.loc[gen_df["baseline"] == "bbar", "tf_minus_baseline"].mean()),
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {args.out}")
    print(f"[write] {args.out.with_suffix('.pdf')}")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
