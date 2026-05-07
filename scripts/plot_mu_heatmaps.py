"""Off-diagonal-only chemical-potential heatmap with rule annotations.

Diagonal is masked (light gray). The color scale is set by the
off-diagonal range. Strong off-diagonal entries (|M|>0.05) are framed
with a category-coloured border and labelled with the rule they
recover. Veber's RotB independence is annotated on the empty RotB axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROPS_CANON = ["logP", "qed", "sa", "tpsa", "mw", "hba", "hbd", "rotb"]
PROP_LABELS = {
    "logP": "logP", "qed": "QED", "sa": "SA", "tpsa": "TPSA",
    "mw": "MW", "hba": "HBA", "hbd": "HBD", "rotb": "RotB",
}
STRONG_THRESH = 0.05

# pushed (condition axis) -> responded (potential axis) -> (category, short label, full rule name)
RULE_MAP = {
    # Cat A: geometric / TPSA identity (Veber-style geometry plus size couplings)
    ("hba", "tpsa"):  ("A", "geom", "geometric TPSA identity"),
    ("tpsa", "hba"):  ("A", "geom", "geometric TPSA identity"),
    ("hbd", "tpsa"):  ("A", "geom", "geometric TPSA identity"),
    ("tpsa", "hbd"):  ("A", "geom", "geometric TPSA identity"),
    ("tpsa", "mw"):   ("A", "geom", "size-polarity coupling"),
    ("logP", "mw"):   ("A", "geom", "size-lipophilicity coupling"),
    ("mw", "logP"):   ("A", "geom", "size-lipophilicity coupling"),
    # Cat B: Bickerton QED per-axis penalties
    ("mw",   "qed"):  ("B", "QED",  "Bickerton QED penalty"),
    ("tpsa", "qed"):  ("B", "QED",  "Bickerton QED penalty"),
    ("hba",  "qed"):  ("B", "QED",  "Bickerton QED penalty"),
    ("rotb", "qed"):  ("B", "QED",  "Bickerton QED penalty"),
    ("logP", "qed"):  ("B", "QED",  "Bickerton QED penalty"),
    # Cat C: Lipinski polarity-lipophilicity anti-correlation
    ("logP", "tpsa"): ("C", "Lip",  "Lipinski polarity-lipophilicity"),
    ("tpsa", "logP"): ("C", "Lip",  "Lipinski polarity-lipophilicity"),
    # Cat D: Ertl complexity-accessibility coupling
    ("logP", "sa"):   ("D", "Ertl", "Ertl complexity-cost"),
    ("sa", "logP"):   ("D", "Ertl", "Ertl complexity-cost"),
    # Cat E: candidate HBA-HBD allocation trade-off
    ("hba", "hbd"):   ("E", "new",  "candidate HBA-HBD rule"),
    ("hbd", "hba"):   ("E", "new",  "candidate HBA-HBD rule"),
}

CAT_COLORS = {
    "A": "#56B4E9",  # sky    — geometric / TPSA identity
    "B": "#E69F00",  # orange — Bickerton QED
    "C": "#0072B2",  # blue   — Lipinski
    "D": "#009E73",  # green  — Ertl
    "E": "#D55E00",  # vermillion — candidate
}
CAT_LEGEND = [
    ("A", "geom", "geometric TPSA identity"),
    ("B", "QED",  "Bickerton QED penalties"),
    ("C", "Lip",  "Lipinski polarity-lipophilicity"),
    ("D", "Ertl", "Ertl complexity-cost"),
    ("E", "new",  "candidate HBA-HBD rule"),
]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_matrix(report: Path) -> np.ndarray:
    d = json.loads(report.read_text())
    if d["phi_properties"] != PROPS_CANON:
        raise RuntimeError(f"unexpected phi_properties: {d['phi_properties']}")
    return np.asarray(d["response_matrix"], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path,
                        default=Path("results/eval/phase3/c2_report.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/figures/paper/fig_mu_heatmap_offdiag.png"))
    args = parser.parse_args()

    configure_style()
    M = load_matrix(args.report)
    n = M.shape[0]
    eye = np.eye(n, dtype=bool)

    M_off = np.where(eye, np.nan, M)
    vlim = float(np.nanmax(np.abs(M_off)))

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#dcdcdc")

    # Layout: upper-left = legend, right = heatmap, lower-left = blank.
    fig = plt.figure(figsize=(11.0, 7.4))
    ax_legend = fig.add_axes([0.04, 0.55, 0.36, 0.38])
    ax_legend.axis("off")
    ax = fig.add_axes([0.46, 0.10, 0.45, 0.82])

    im = ax.imshow(M_off, cmap=cmap, vmin=-vlim, vmax=vlim)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([PROP_LABELS[p] for p in PROPS_CANON])
    ax.set_yticklabels([PROP_LABELS[p] for p in PROPS_CANON])
    ax.set_xlabel(r"condition axis $y_i$")
    ax.set_ylabel(r"response axis $\mu_j$")
    ax.set_title(
        rf"Off-diagonal chemical-potential matrix    color range $\pm${vlim:.3f}",
        pad=8,
    )

    # diagonal: keep its value visible in italic gray so the reader sees what is masked
    for i in range(n):
        ax.text(i, i, f"{M[i,i]:.2f}", ha="center", va="center",
                fontsize=7.5, color="#666666", style="italic")

    # off-diagonal annotations + strong-cell category borders + rule short labels
    n_strong = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            is_strong = abs(v) > STRONG_THRESH
            # value in the cell
            text_color = "white" if abs(v) >= 0.55 * vlim else "black"
            if is_strong:
                ax.text(j, i - 0.18, f"{v:+.2f}",
                        ha="center", va="center", fontsize=7.6,
                        fontweight="bold", color=text_color)
            else:
                ax.text(j, i, f"{v:+.2f}",
                        ha="center", va="center", fontsize=7.0, color=text_color)

            if not is_strong:
                continue
            n_strong += 1
            pushed = PROPS_CANON[i]
            responded = PROPS_CANON[j]
            cat, short, _ = RULE_MAP[(pushed, responded)]
            color = CAT_COLORS[cat]
            ax.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                linewidth=2.2, edgecolor=color, facecolor="none",
            ))
            ax.text(j, i + 0.22, short,
                    ha="center", va="center", fontsize=7.4,
                    fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("off-diagonal response")

    # legend in upper-left axes (no border, anchored to top of its panel)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=CAT_COLORS[cat],
                  linewidth=2.0, label=f"{short} — {full}")
        for (cat, short, full) in CAT_LEGEND
    ]
    leg = ax_legend.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=9,
        title=f"strong off-diagonal entries   |M|>{STRONG_THRESH}   (n={n_strong})",
        title_fontsize=9,
        handlelength=1.4,
        borderaxespad=0.0,
    )
    leg._legend_box.align = "left"

    # Veber callout placed under the legend (still in upper-left band).
    ax_legend.text(
        0.0, 0.10,
        "Veber: RotB axis is statistically independent of\n"
        "the compositional axes (only one weak coupling\n"
        "on the entire row and column).",
        transform=ax_legend.transAxes,
        ha="left", va="top", fontsize=8.5, color="#444444",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[write] {args.out}")
    print(f"[write] {args.out.with_suffix('.pdf')}")
    print(f"[info] off-diagonal vlim = {vlim:.4f}")
    print(f"[info] {n_strong} strong off-diagonal entries (|M|>{STRONG_THRESH})")


if __name__ == "__main__":
    main()
