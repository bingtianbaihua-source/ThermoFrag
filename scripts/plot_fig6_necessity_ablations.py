#!/usr/bin/env python
"""Build Fig. 6: three sample-time necessity ablations.

The figure is generated from the Phase-5 unified ablation summary. Each panel
shows the specific failure mode caused by removing one Hamiltonian term.
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


OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#999999",
    "black": "#000000",
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
    ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontweight="bold", fontsize=11, va="top")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("results/eval/phase5/c6_unified.json"))
    parser.add_argument("--out", type=Path, default=Path("results/figures/paper/fig6_necessity_ablations.png"))
    args = parser.parse_args()

    configure_style()
    summary = json.loads(args.summary.read_text())
    ablations = summary["ablations"]
    base = summary["tf_base"]

    fig = plt.figure(figsize=(7.2, 3.15))
    gs = fig.add_gridspec(1, 3, wspace=0.70)

    # a: no-QM randomization breaks energy ranking and absolute scale.
    no_qm = ablations["no_qm"]
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2)
    spearman_vals = [no_qm["trained_spearman"], no_qm["random_init_spearman"]]
    ax.bar(x, spearman_vals, color=[OKABE["blue"], OKABE["grey"]], width=0.6)
    ax.axhline(0, color=OKABE["black"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["trained", "random\nQM head"])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Spearman vs DFT")
    ax.set_title(r"$E^{QM}$ carries energy ranking", fontsize=8, pad=8)
    for i, v in enumerate(spearman_vals):
        ax.text(i, v + (0.07 if v >= 0 else -0.12), f"{v:.3f}", ha="center", va="center")
    ax2 = ax.twinx()
    mae_vals = [no_qm["trained_mae_kcal_per_mol"], no_qm["random_init_mae_kcal_per_mol"]]
    ax2.plot(x, mae_vals, marker="o", color=OKABE["vermillion"], lw=1.4)
    ax2.set_yscale("log")
    ax2.set_ylabel("MAE", color=OKABE["vermillion"], labelpad=1)
    ax2.tick_params(axis="y", colors=OKABE["vermillion"])
    panel_label(ax, "a")

    # b: no-coupling increases raw acceptance but collapses valid decoding.
    no_cpl = ablations["no_coupling"]
    ax = fig.add_subplot(gs[0, 1])
    labels = ["decoded\nyield", "MH accept.\nrate"]
    trained = [no_cpl["tf_base_yield"], no_cpl["accept_rate_tf_base"]]
    ablated = [no_cpl["nocoup_yield"], no_cpl["accept_rate_nocoup"]]
    idx = np.arange(len(labels))
    w = 0.34
    ax.bar(idx - w / 2, trained, width=w, color=OKABE["green"], label="trained")
    ax.bar(idx + w / 2, ablated, width=w, color=OKABE["grey"], label=r"$V^{couple}=0$")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("fraction")
    ax.set_title(r"$V^{couple}$ preserves decodable graphs", fontsize=8, pad=8)
    ax.legend(frameon=False, loc="upper right")
    ax.text(0.04, 0.70, "yield ratio = 0.23", transform=ax.transAxes, fontsize=7)
    panel_label(ax, "b")

    # c: no-mu removes target-dependent Vina advantage.
    no_mu = ablations["no_mu"]
    ax = fig.add_subplot(gs[0, 2])
    ax.bar([0], [no_mu["c3_sigwins_tf_vs_nomu"]], color=OKABE["purple"], width=0.46)
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(0, no_mu["c3_n_tested"])
    ax.set_xticks([0])
    ax.set_xticklabels([r"TF vs $\mu=0$"])
    ax.set_ylabel("targets with\nTF Vina advantage")
    ax.set_title(r"$\mu(y)$ carries property targeting", fontsize=8, pad=8)
    ax.text(0, no_mu["c3_sigwins_tf_vs_nomu"] + 0.45, "13 / 15", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.50,
        0.13,
        "paired top-10\nVina comparison",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7,
    )
    panel_label(ax, "c")

    fig.suptitle("Each Hamiltonian term fails through a distinct ablation mode", y=1.02, fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    manifest = {
        "figure": "Fig. 6",
        "script": str(Path(__file__).as_posix()),
        "outputs": [str(args.out), str(args.out.with_suffix(".pdf"))],
        "panels": {
            "a": {
                "description": "Random-initialized QM head breaks DFT energy ranking and scale",
                "data_sources": [str(args.summary), "results/eval/phase5/c6_noqm.json"],
            },
            "b": {
                "description": "Zeroing coupling potential lowers decoded yield despite higher MH acceptance",
                "data_sources": [str(args.summary), "results/eval/phase5/nocoupling_samples"],
            },
            "c": {
                "description": "Zeroing chemical-potential field removes Vina advantage on most LIT-PCBA targets",
                "data_sources": [str(args.summary), "results/eval/phase5/c6_ablation.json", "results/eval/phase5/nomu_vina"],
            },
        },
        "summary_values": {
            "no_qm_trained_spearman": float(no_qm["trained_spearman"]),
            "no_qm_random_spearman": float(no_qm["random_init_spearman"]),
            "no_qm_trained_mae": float(no_qm["trained_mae_kcal_per_mol"]),
            "no_qm_random_mae": float(no_qm["random_init_mae_kcal_per_mol"]),
            "no_coupling_yield_trained": float(no_cpl["tf_base_yield"]),
            "no_coupling_yield_ablated": float(no_cpl["nocoup_yield"]),
            "no_coupling_accept_trained": float(no_cpl["accept_rate_tf_base"]),
            "no_coupling_accept_ablated": float(no_cpl["accept_rate_nocoup"]),
            "no_mu_sigwins": int(no_mu["c3_sigwins_tf_vs_nomu"]),
            "no_mu_n_tested": int(no_mu["c3_n_tested"]),
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {args.out}")
    print(f"[write] {args.out.with_suffix('.pdf')}")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
