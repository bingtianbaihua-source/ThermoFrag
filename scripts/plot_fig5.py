"""Fig 5 — chemical-potential interpretability (C2).

Three panels from existing Phase-3 outputs + lightweight derivations:

  (a) The μ(y) response diagonal plotted against Bickerton QED weights on
      six aligned properties. Scatter + r-squared.
  (b) Bar plot: diagonal |μ_j(y_j=+1σ) − μ_j(y_j=−1σ)| vs the two reference
      weight vectors (Bickerton, WC proxy) for the 8 training properties.
  (c) μ response matrix heatmap (recycled from mu_response_heatmap.png if
      present, else skipped).

Input: ``results/eval/phase3/c2_report.json`` (produced by scripts/eval_chempot.py).
Output: ``results/eval/phase5/fig5_chempot.png``.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger("fig5")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path,
                   default=Path("results/eval/phase3/c2_report.json"))
    p.add_argument("--out", type=Path,
                   default=Path("results/eval/phase5/fig5_chempot.png"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    d = json.loads(args.report.read_text())
    props = d["phi_properties"]
    response = np.asarray(d["response_matrix"])  # [K, K]

    bick = d["bickerton"]
    wcp  = d["wc_proxy"]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    # --- Panel (a): Bickerton scatter ---
    ax = axes[0]
    mu_b = np.asarray(bick["mu_diag"])
    wt_b = np.asarray(bick["bickerton_weight_mean"])
    ax.scatter(wt_b, mu_b, s=72, c="#d62728", edgecolor="black", linewidth=0.6)
    for i, name in enumerate(bick["aligned_properties"]):
        ax.annotate(name, (wt_b[i], mu_b[i]),
                    xytext=(5, 4), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Bickerton QED weight")
    ax.set_ylabel("learned $|\\partial \\mu_j / \\partial y_j|$")
    ax.set_title(f"Fig 5a — Bickerton alignment   ρ={bick['spearman']:.3f}")

    # --- Panel (b): bar plot, all 8 props, Bickerton vs WC proxy ---
    ax = axes[1]
    x = np.arange(len(props))
    mu_diag = np.diag(response)
    mu_abs = np.abs(mu_diag)
    mu_abs_n = mu_abs / (mu_abs.max() + 1e-9)
    # Normalise reference weights to max=1 for display.
    bw = np.zeros(len(props)); ww = np.zeros(len(props))
    for name, w in zip(bick["aligned_properties"], bick["bickerton_weight_mean"]):
        bw[props.index(name)] = w
    for name, w in zip(wcp["other_properties"], wcp["empirical_logp_pearson_abs"]):
        ww[props.index(name)] = w
    bw = bw / (bw.max() + 1e-9)
    ww = ww / (ww.max() + 1e-9)
    w = 0.28
    ax.bar(x - w, mu_abs_n, width=w, label="ThermoFrag μ diag", color="#d62728")
    ax.bar(x,     bw,       width=w, label="Bickerton QED",      color="#1f77b4")
    ax.bar(x + w, ww,       width=w, label="WC proxy (|logP-pearson|)", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(props, rotation=45, ha="right")
    ax.set_ylabel("normalized weight")
    ax.set_title("Fig 5b — per-property weight alignment")
    ax.legend(loc="upper right", fontsize=8)

    # --- Panel (c): heatmap of response matrix ---
    ax = axes[2]
    im = ax.imshow(response, cmap="RdBu_r",
                   vmin=-max(abs(response.min()), abs(response.max())),
                   vmax=+max(abs(response.min()), abs(response.max())))
    ax.set_xticks(np.arange(len(props))); ax.set_xticklabels(props, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(props))); ax.set_yticklabels(props)
    ax.set_xlabel(r"$y_i$ (condition axis)")
    ax.set_ylabel(r"$\mu_j$ (potential axis)")
    ax.set_title(r"Fig 5c — $\partial \mu_j / \partial y_i$ heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    logger.info("fig → %s", args.out)


if __name__ == "__main__":
    main()
