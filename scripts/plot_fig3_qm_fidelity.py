#!/usr/bin/env python
"""Build Fig. 3: QM energy fidelity from archived Phase-1 metrics.

The original raw SPICE validation shards are not shipped in the current
working tree, so this figure is generated from the archived numerical
outputs of the Phase-1 QM evaluation. The manifest records that each panel
uses summary metrics rather than raw per-molecule predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "grey": "#999999",
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
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/eval/phase1_fig2_v2/fig2_metrics.json"))
    parser.add_argument(
        "--outlier-report",
        type=Path,
        default=Path("results/eval/phase1_outlier_diag/outlier_report.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/figures/paper/fig3_qm_fidelity.png"))
    args = parser.parse_args()

    configure_style()
    metrics = json.loads(args.metrics.read_text())
    outliers = json.loads(args.outlier_report.read_text())

    strata = metrics["strata"]
    force = metrics["per_element_force_mae_kcal_per_A"]
    large_agg = metrics["large_aggregate"]
    small_out = outliers["small"]["outlier_contribution_to_MAE"]

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, wspace=0.38, hspace=0.58)

    # a: aggregate and drug-like energy metrics.
    ax = fig.add_subplot(gs[0, 0])
    labels = ["all\nMAE/mol", "all\nSpearman", ">=30 atoms\nMAE/atom", ">=40 atoms\nMAE/atom"]
    vals = [
        large_agg["mae_kcal"],
        large_agg["spearman"],
        next(s["large_per_atom_mae"] for s in strata if s["thresh"] == 30),
        next(s["large_per_atom_mae"] for s in strata if s["thresh"] == 40),
    ]
    colors = [OKABE["grey"], OKABE["blue"], OKABE["green"], OKABE["green"]]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylabel("metric value")
    ax.set_title("energy metrics")
    for i, v in enumerate(vals):
        ax.text(i, v * (1.08 if v > 1 else 1.25), f"{v:.3g}", ha="center", va="bottom")
    panel_label(ax, "a")

    # b: force MAE by element.
    ax = fig.add_subplot(gs[0, 1])
    elems = list(force.keys())
    vals_force = [force[e] for e in elems]
    bar_colors = [OKABE["vermillion"] if e == "P" else OKABE["sky"] for e in elems]
    ax.bar(elems, vals_force, color=bar_colors)
    ax.axhline(5.0, color="black", ls="--", lw=0.8)
    ax.text(0.02, 0.92, "5 kcal mol$^{-1}$ A$^{-1}$", transform=ax.transAxes)
    ax.set_ylabel("force MAE\n(kcal mol$^{-1}$ A$^{-1}$)")
    ax.set_title("force fidelity by element")
    panel_label(ax, "b")

    # c: per-atom MAE by molecular size.
    ax = fig.add_subplot(gs[1, 0])
    x = np.array([s["thresh"] for s in strata])
    xlabels = ["all" if t == 0 else str(t) for t in x]
    small = np.array([s["small_per_atom_mae"] for s in strata])
    large = np.array([s["large_per_atom_mae"] for s in strata])
    idx = np.arange(len(x))
    w = 0.34
    ax.bar(idx - w / 2, small, width=w, color=OKABE["grey"], label="small head")
    ax.bar(idx + w / 2, large, width=w, color=OKABE["blue"], label="large head")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xticks(idx)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("minimum atom count")
    ax.set_ylabel("per-atom MAE\n(kcal mol$^{-1}$ atom$^{-1}$)")
    ax.set_title("drug-like size stratification")
    ax.legend(frameon=False)
    panel_label(ax, "c")

    # d: outlier concentration from archived report.
    ax = fig.add_subplot(gs[1, 1])
    pct = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
    contrib = np.array(
        [
            small_out["top_0.1pct_contrib_to_MAE_total"],
            small_out["top_0.5pct_contrib_to_MAE_total"],
            small_out["top_1.0pct_contrib_to_MAE_total"],
            small_out["top_5.0pct_contrib_to_MAE_total"],
            small_out["top_10.0pct_contrib_to_MAE_total"],
        ]
    )
    ax.plot(pct, contrib * 100, marker="o", color=OKABE["orange"])
    ax.set_xscale("log")
    ax.set_xlabel("top molecules by residual (%)")
    ax.set_ylabel("MAE contribution (%)")
    ax.set_title("aggregate error is tail-driven")
    ax.grid(True, alpha=0.25)
    ax.text(0.04, 0.84, "top 1% -> 23.2%", transform=ax.transAxes)
    panel_label(ax, "d")

    fig.suptitle("Quantum-energy fidelity in the drug-like regime", y=0.995, fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    manifest = {
        "figure": "Fig. 3",
        "script": str(Path(__file__).as_posix()),
        "outputs": [str(args.out), str(args.out.with_suffix(".pdf"))],
        "note": "Generated from archived Phase-1 summary metrics; raw SPICE prediction shards are not present in this working tree.",
        "panels": {
            "a": {
                "description": "Aggregate and drug-like energy metrics for the large recalibrated QM head",
                "data_sources": [str(args.metrics)],
            },
            "b": {
                "description": "Per-element force MAE for the large recalibrated QM head",
                "data_sources": [str(args.metrics)],
            },
            "c": {
                "description": "Per-atom energy MAE stratified by molecule size",
                "data_sources": [str(args.metrics)],
            },
            "d": {
                "description": "Outlier contribution to aggregate MAE",
                "data_sources": [str(args.outlier_report)],
            },
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {args.out}")
    print(f"[write] {args.out.with_suffix('.pdf')}")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
