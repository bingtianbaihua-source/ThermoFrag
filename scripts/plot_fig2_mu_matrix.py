#!/usr/bin/env python
"""Build Fig. 2: chemical-potential matrix evidence.

Inputs are the Phase-3 deterministic chemical-potential report and the
Phase-7 mu-matrix validation outputs. The script writes a six-panel
composite figure plus a JSON manifest listing the data source for every
panel.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROPS_CANON = ["logP", "qed", "sa", "tpsa", "mw", "hba", "hbd", "rotb"]
PROP_LABELS = {
    "logP": "logP",
    "qed": "QED",
    "sa": "SA",
    "tpsa": "TPSA",
    "mw": "MW",
    "hba": "HBA",
    "hbd": "HBD",
    "rotb": "RotB",
}

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


def rule_category_rows() -> list[dict[str, str]]:
    """Manual category map from docs/validation/MU_MATRIX_FINDINGS.md.

    The rows intentionally cover only the strict strong off-diagonal
    entries used in the Phase-7 sign/stability tests.
    """
    rows = []

    def add(pushed: str, responded: str, cat: str, label: str) -> None:
        rows.append(
            {
                "pushed": pushed,
                "responded": responded,
                "category": cat,
                "label": label,
            }
        )

    # A: geometric / near-definition couplings.
    add("hba", "tpsa", "A", "polar geometry")
    add("tpsa", "hba", "A", "polar geometry")
    add("hbd", "tpsa", "A", "polar geometry")
    add("tpsa", "hbd", "A", "polar geometry")
    add("tpsa", "mw", "A", "size-polarity")
    add("logP", "mw", "A", "size-lipophilicity")
    add("mw", "logP", "A", "size-lipophilicity")

    # B: Bickerton/QED penalty axes.
    add("mw", "qed", "B", "QED penalty")
    add("tpsa", "qed", "B", "QED penalty")
    add("hba", "qed", "B", "QED penalty")
    add("rotb", "qed", "B", "QED penalty")
    add("logP", "qed", "B", "QED penalty")

    # C: Lipinski-style polarity/lipophilicity trade-off.
    add("logP", "tpsa", "C", "polarity-lipophilicity")
    add("tpsa", "logP", "C", "polarity-lipophilicity")

    # D: Ertl-style synthetic complexity/accessibility coupling.
    add("logP", "sa", "D", "complexity-accessibility")
    add("sa", "logP", "D", "complexity-accessibility")

    # E: candidate HBA-HBD allocation trade-off.
    add("hba", "hbd", "E", "HBA-HBD allocation")
    add("hbd", "hba", "E", "HBA-HBD allocation")
    return rows


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


def load_response(report_path: Path) -> tuple[list[str], np.ndarray]:
    report = json.loads(report_path.read_text())
    props = report["phi_properties"]
    matrix = np.asarray(report["response_matrix"], dtype=float)
    if props != PROPS_CANON:
        raise RuntimeError(f"unexpected phi_properties: {props}")
    return props, matrix


def pivot_corr(corr_df: pd.DataFrame, value: str) -> np.ndarray:
    mat = np.zeros((len(PROPS_CANON), len(PROPS_CANON)), dtype=float)
    for row in corr_df.itertuples(index=False):
        mat[int(row.i), int(row.j)] = float(getattr(row, value))
    return mat


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
        va="top",
        ha="left",
    )


def draw_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    *,
    cmap: str = "RdBu_r",
    vlim: float | None = None,
    annotate_strong: bool = False,
) -> None:
    if vlim is None:
        vlim = float(np.nanmax(np.abs(matrix)))
    im = ax.imshow(matrix, cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(PROPS_CANON)))
    ax.set_xticklabels([PROP_LABELS[p] for p in PROPS_CANON], rotation=45, ha="right")
    ax.set_yticks(range(len(PROPS_CANON)))
    ax.set_yticklabels([PROP_LABELS[p] for p in PROPS_CANON])
    ax.set_title(title, pad=6)
    if annotate_strong:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j and abs(matrix[i, j]) > 0.05:
                    ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=9)
    return im


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("results/eval/phase3/c2_report.json"))
    parser.add_argument(
        "--corr",
        type=Path,
        default=Path("results/eval/phase7/mu_crossval/chembl_corr_matrix.parquet"),
    )
    parser.add_argument(
        "--chembl-partial",
        type=Path,
        default=Path("results/eval/phase7/mu_crossval/chembl_partial_corr.parquet"),
    )
    parser.add_argument(
        "--litpcba-partial",
        type=Path,
        default=Path(
            "results/eval/phase7/mu_crossval/independent_population/litpcba_actives_corr.parquet"
        ),
    )
    parser.add_argument(
        "--seed-summary",
        type=Path,
        default=Path("results/eval/phase7/mu_crossval/seed_stability/M_mean_std.parquet"),
    )
    parser.add_argument(
        "--feature-summary",
        type=Path,
        default=Path("results/eval/phase7/mu_crossval/feature_perturbation/summary.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("results/figures/paper/fig2_mu_matrix.png"))
    parser.add_argument(
        "--category-map",
        type=Path,
        default=Path("results/eval/phase7/mu_crossval/rule_category_map.csv"),
    )
    args = parser.parse_args()

    configure_style()

    props, M = load_response(args.report)
    corr_df = pd.read_parquet(args.corr)
    P = pivot_corr(corr_df, "pearson")
    chembl_partial = pd.read_parquet(args.chembl_partial)
    litpcba_partial = pd.read_parquet(args.litpcba_partial)
    seed = pd.read_parquet(args.seed_summary)
    feature = pd.read_parquet(args.feature_summary)

    cat_df = pd.DataFrame(rule_category_rows())
    args.category_map.parent.mkdir(parents=True, exist_ok=True)
    cat_df.to_csv(args.category_map, index=False)

    strong = (np.abs(M) > 0.05) & (~np.eye(M.shape[0], dtype=bool))
    sign_match = (np.sign(M) == np.sign(P)) & strong

    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], wspace=0.55, hspace=0.46)

    # a: response matrix.
    ax = fig.add_subplot(gs[0, 0])
    im = draw_heatmap(ax, M, "chemical-potential response", annotate_strong=True)
    ax.set_xlabel("condition axis")
    ax.set_ylabel("response axis")
    panel_label(ax, "a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("response")

    # b: sign agreement.
    ax = fig.add_subplot(gs[0, 1])
    sign_panel = np.full_like(M, np.nan, dtype=float)
    sign_panel[strong] = np.where(sign_match[strong], 1.0, -1.0)
    im2 = ax.imshow(sign_panel, cmap="BrBG", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PROPS_CANON)))
    ax.set_xticklabels([PROP_LABELS[p] for p in PROPS_CANON], rotation=45, ha="right")
    ax.set_yticks(range(len(PROPS_CANON)))
    ax.set_yticklabels([PROP_LABELS[p] for p in PROPS_CANON])
    ax.set_title(f"sign agreement ({int(sign_match.sum())}/{int(strong.sum())})", pad=6)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if strong[i, j]:
                ax.text(j, i, "✓" if sign_match[i, j] else "x", ha="center", va="center", fontsize=8)
    panel_label(ax, "b")

    # c: category map/counts.
    ax = fig.add_subplot(gs[0, 2])
    cat_colors = {
        "A": OKABE_ITO["sky"],
        "B": OKABE_ITO["orange"],
        "C": OKABE_ITO["blue"],
        "D": OKABE_ITO["green"],
        "E": OKABE_ITO["vermillion"],
    }
    counts = cat_df["category"].value_counts().sort_index()
    bars = ax.bar(counts.index, counts.values, color=[cat_colors[c] for c in counts.index])
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.15, str(int(val)), ha="center")
    ax.set_ylim(0, max(counts.values) + 2)
    ax.set_ylabel("strong entries")
    ax.set_xlabel("rule category")
    ax.set_title("rule-category coverage", pad=6)
    ax.text(
        0.02,
        0.98,
        "A geometry\nB QED penalties\nC polarity-logP\nD SA coupling\nE HBA-HBD",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.8,
    )
    panel_label(ax, "c")

    # d/e: partial correlations.
    def partial_panel(ax: plt.Axes, df: pd.DataFrame, title: str, label: str) -> None:
        order = ["marginal", "controls=mw", "controls=tpsa", "controls=tpsa,mw"]
        sub = df[(df["pair"] == "hba_hbd") & (df["controls"].isin(order))].copy()
        sub["controls"] = pd.Categorical(sub["controls"], categories=order, ordered=True)
        sub = sub.sort_values("controls")
        x = np.arange(len(sub))
        colors = [OKABE_ITO["vermillion"] if c == "controls=tpsa,mw" else "#9E9E9E" for c in sub["controls"]]
        ax.bar(x, sub["r"], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["marg.", "MW", "TPSA", "TPSA+MW"], rotation=30, ha="right")
        ax.set_ylabel("partial r")
        ax.set_ylim(-0.62, 0.12)
        ax.set_title(title, pad=6)
        main = sub[sub["controls"] == "controls=tpsa,mw"].iloc[0]
        ax.text(0.03, 0.08, f"n={int(main.n):,}\nr={float(main.r):+.3f}", transform=ax.transAxes)
        panel_label(ax, label)

    ax = fig.add_subplot(gs[1, 0])
    partial_panel(ax, chembl_partial, "ChEMBL HBA-HBD", "d")

    ax = fig.add_subplot(gs[1, 1])
    partial_panel(ax, litpcba_partial, "LIT-PCBA actives HBA-HBD", "e")

    # f: seed stability plus feature-swap summary.
    ax = fig.add_subplot(gs[1, 2])
    strong_seed = seed[seed["strong_offdiag"]].copy()
    strong_seed["name"] = strong_seed["pushed"].map(PROP_LABELS) + "->" + strong_seed["responded"].map(PROP_LABELS)
    strong_seed["is_hba_hbd"] = strong_seed.apply(
        lambda r: {r["pushed"], r["responded"]} == {"hba", "hbd"}, axis=1
    )
    strong_seed = strong_seed.sort_values(["is_hba_hbd", "M_seed_mean"], ascending=[False, True])
    y = np.arange(len(strong_seed))
    x = strong_seed["M_seed_mean"].to_numpy()
    lo = strong_seed["ci_lo_2_5"].to_numpy()
    hi = strong_seed["ci_hi_97_5"].to_numpy()
    xerr = np.vstack([x - lo, hi - x])
    colors = np.where(strong_seed["is_hba_hbd"], OKABE_ITO["vermillion"], OKABE_ITO["blue"])
    ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor="#555555", elinewidth=0.8, capsize=1.8, zorder=1)
    ax.scatter(x, y, c=colors, s=18, zorder=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(strong_seed["name"], fontsize=5.8)
    ax.set_xlabel("seed mean response")
    ax.set_title("10-seed stability", pad=6)
    n_ci = int(strong_seed["ci_excludes_zero"].sum())
    n_total = int(len(strong_seed))
    ax.text(
        0.02,
        0.98,
        f"CI excludes 0: {n_ci}/{n_total}\nHBA-HBD highlighted",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    panel_label(ax, "f")

    fig.suptitle("Chemical-potential matrix evidence", y=0.985, fontsize=10)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")

    manifest = {
        "figure": "Fig. 2",
        "script": str(Path(__file__).as_posix()),
        "outputs": [str(args.out), str(args.out.with_suffix(".pdf"))],
        "panels": {
            "a": {
                "description": "8x8 deterministic chemical-potential response matrix",
                "data_sources": [str(args.report)],
            },
            "b": {
                "description": "strong off-diagonal sign agreement against ChEMBL Pearson correlations",
                "data_sources": [str(args.report), str(args.corr)],
            },
            "c": {
                "description": "manual rule-category map for strong off-diagonal entries",
                "data_sources": [
                    "docs/validation/MU_MATRIX_FINDINGS.md",
                    "results/eval/phase7/mu_crossval/findings_report.md",
                    str(args.category_map),
                ],
            },
            "d": {
                "description": "ChEMBL HBA-HBD partial correlations",
                "data_sources": [str(args.chembl_partial)],
            },
            "e": {
                "description": "LIT-PCBA active-set HBA-HBD partial correlations",
                "data_sources": [str(args.litpcba_partial)],
            },
            "f": {
                "description": "10-seed mu-head stability and feature-definition perturbation summary",
                "data_sources": [str(args.seed_summary), str(args.feature_summary)],
            },
        },
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[write] {args.out}")
    print(f"[write] {args.out.with_suffix('.pdf')}")
    print(f"[write] {args.category_map}")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
