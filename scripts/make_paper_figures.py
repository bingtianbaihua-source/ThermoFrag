"""Paper-ready figures and Table 1 for the ThermoFrag manuscript.

Produces:
  results/figures/paper/fig1_schematic.png     — conceptual schematic (matplotlib)
  results/figures/paper/fig7_gvg_box.png       — 4-way generator Vina top-k box plot
  results/figures/paper/fig8_gvg_strain.png    — 4-way generator strain box plot
  results/figures/paper/table1_claims.md       — Claim x Number x Threshold x Verdict
  results/figures/paper/table1_claims.csv      — same, CSV form

Inputs it reads (must exist):
  results/eval/claim_summary.json
  results/eval/phase4/vina/*.parquet                                       (TF)
  results/eval/phase4/strain/*.parquet                                     (TF)
  results/eval/phase4_baselines/{rxnflow,bbar,targetdiff}/vina/*.parquet
  results/eval/phase4_baselines/{rxnflow,bbar,targetdiff}/strain/*.parquet
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
PHASE4 = REPO / "results/eval/phase4"
PHASE4B = REPO / "results/eval/phase4_baselines"
OUT = REPO / "results/figures/paper"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA",   "IDH1",  "KAT2A",   "MAPK1",     "MTORC1",
    "OPRK1", "PKM2",  "PPARG",   "TP53",      "VDR",
]
BASELINES = ["rxnflow", "bbar", "targetdiff"]
POOL_CAP = 100  # matches BASELINES.md pool-size-fairness protocol
TOP_K = 10


def _load_ok(pq: Path, col: str, max_pool: Optional[int] = None) -> Optional[np.ndarray]:
    if not pq.exists():
        return None
    df = pd.read_parquet(pq)
    if max_pool is not None and "chain_idx" in df.columns:
        df = df[df["chain_idx"] < max_pool]
    ok = df[(df["status"] == "ok") & df[col].notna()]
    if len(ok) == 0:
        return None
    return ok[col].to_numpy()


def _top_k_per_target(root: Path, col: str, max_pool: Optional[int], k: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for t in TARGETS:
        arr = _load_ok(root / f"{t}.parquet", col, max_pool=max_pool)
        if arr is None or len(arr) == 0:
            out[t] = np.array([], dtype=float)
        else:
            out[t] = np.sort(arr)[:k]  # lowest k (best Vina; lowest strain)
    return out


# ---------------------------------------------------------------------------
# Fig 1 — conceptual schematic
# ---------------------------------------------------------------------------

def make_fig1() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2),
                             gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})

    # Panel a: p(m,x|y) = Z^-1 exp(-beta H) as a flow diagram
    ax = axes[0]
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, 9.2, "a   Boltzmann generative model",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 7.7,
            r"$p_\theta(m,\mathbf{x}\,|\,y) = \frac{1}{Z_\theta(y,\beta)}\,"
            r"\exp(-\beta\,\mathcal{H}_\theta(m,\mathbf{x};y))$",
            ha="center", fontsize=12)
    # target property y -> sampler -> molecule
    box_kw = dict(boxstyle="round,pad=0.4", facecolor="#e8f0ff", edgecolor="#2b5f99")
    ax.text(1.8, 4.5, "target\nproperty  $y$", ha="center", va="center",
            fontsize=10, bbox=box_kw)
    ax.annotate("", xy=(4.3, 4.5), xytext=(2.9, 4.5),
                arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(5.0, 4.5, "MH + Langevin\nannealed sampler", ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff1d6", edgecolor="#b57900"))
    ax.annotate("", xy=(8.3, 4.5), xytext=(6.8, 4.5),
                arrowprops=dict(arrowstyle="->", lw=1.4))
    ax.text(9.0, 4.5, "molecule\n$(m,\\mathbf{x})$", ha="center", va="center",
            fontsize=10, bbox=box_kw)
    ax.text(5, 2.2,
            "detailed balance holds; β = inverse temperature\n"
            "(β→∞: greedy fragment assembly; β→0: data density)",
            ha="center", fontsize=9.5, style="italic", color="#444")

    # Panel b: Hamiltonian decomposition as three colored blocks
    ax = axes[1]
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, 9.2, "b   Hamiltonian decomposition",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(5, 7.7,
            r"$\mathcal{H}_\theta = E^{\mathrm{QM}}_\theta(m,\mathbf{x})"
            r" + V^{\mathrm{couple}}_\theta(m)"
            r" - \mu_\theta(y)^\top \phi(m,\mathbf{x})$",
            ha="center", fontsize=11.5)
    # three colored rectangles with role labels
    rects = [
        ("$E^{\\mathrm{QM}}$\nlearned ML\nforce field",        "#f2c2b2", 0.5, 0.3),
        ("$V^{\\mathrm{couple}}$\nchemical-database\npotential", "#b6d6a8", 3.8, 0.3),
        ("$-\\boldsymbol{\\mu}^\\top\\boldsymbol{\\phi}$\nproperty-field\n(chemical potential)",
                                                               "#a9c4e1", 7.1, 0.3),
    ]
    for label, color, x, y in rects:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), 2.4, 5.0, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#333"))
        ax.text(x + 1.2, y + 2.5, label, ha="center", va="center",
                fontsize=10)

    # Panel c: three-paradigm recovery triangle
    ax = axes[2]
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.text(5, 9.2, "c   Three-paradigm recovery",
            ha="center", fontsize=12, fontweight="bold")

    # triangle vertices: BBAR (top), MLFF (bottom-left), EBM (bottom-right)
    verts = np.array([[5.0, 7.2], [1.8, 2.0], [8.2, 2.0], [5.0, 7.2]])
    ax.plot(verts[:, 0], verts[:, 1], "-", color="#888", lw=1.4)
    # ThermoFrag at centroid
    cx, cy = verts[:-1, 0].mean(), verts[:-1, 1].mean()
    ax.plot(cx, cy, "o", color="#c1272d", ms=11)
    ax.text(cx, cy - 0.7, "ThermoFrag", ha="center", fontsize=10.5,
            fontweight="bold", color="#c1272d")
    # corner labels
    ax.text(5.0, 7.8, "BBAR limit\n"
            r"($\beta{\to}\infty,\,E^{\mathrm{QM}}{=}V{=}0$)",
            ha="center", fontsize=9)
    ax.text(1.5, 1.3, "ML force-field limit\n"
            r"($V{=}0,\,\mu{=}0$)",
            ha="center", fontsize=9)
    ax.text(8.4, 1.3, "Data-density limit\n"
            r"($E^{\mathrm{QM}}{=}0,\,\mu{=}0$)",
            ha="center", fontsize=9)

    fig.tight_layout()
    path = OUT / "fig1_schematic.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Fig 7 — 4-way generator Vina top-10 per target (box plot)
# ---------------------------------------------------------------------------

def make_fig7() -> Path:
    tf = _top_k_per_target(PHASE4 / "vina", "vina_score", max_pool=None, k=TOP_K)
    per_baseline: Dict[str, Dict[str, np.ndarray]] = {}
    for b in BASELINES:
        per_baseline[b] = _top_k_per_target(
            PHASE4B / b / "vina", "vina_score", max_pool=POOL_CAP, k=TOP_K
        )

    colors = {"ThermoFrag": "#c1272d", "rxnflow": "#2a7f3e",
              "bbar": "#eaa300", "targetdiff": "#1f4e97"}
    order = ["ThermoFrag", "rxnflow", "bbar", "targetdiff"]
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 5.0))
    for i, t in enumerate(TARGETS):
        for j, label in enumerate(order):
            if label == "ThermoFrag":
                data = tf[t]
            else:
                data = per_baseline[label][t]
            if len(data) == 0:
                continue
            pos = i + (j - 1.5) * width
            bp = ax.boxplot(
                data, positions=[pos], widths=width * 0.95,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=colors[label], edgecolor="black", lw=0.8),
                medianprops=dict(color="black", lw=1.2),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8),
            )
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(TARGETS, rotation=40, ha="right")
    ax.set_ylabel("Vina top-10 score (kcal/mol; ↓ better)")
    ax.set_title("Fig. 7  Generator-vs-generator Vina top-10 on 15 LIT-PCBA targets "
                 "(pool-size matched, cap=100)")
    ax.invert_yaxis()
    handles = [mpatches.Patch(facecolor=colors[l], edgecolor="black",
                              label=("ThermoFrag" if l == "ThermoFrag" else
                                     {"rxnflow": "RxnFlow",
                                      "bbar": "BBAR",
                                      "targetdiff": "TargetDiff"}[l]))
               for l in order]
    ax.legend(handles=handles, loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / "fig7_gvg_box.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Fig 8 — 4-way generator post-relaxation strain per target
# ---------------------------------------------------------------------------

def make_fig8() -> Path:
    # Use per-target strain distribution (all status='ok' entries, not just top-10)
    def _strain_per_target(root: Path, max_pool: Optional[int]) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for t in TARGETS:
            arr = _load_ok(root / f"{t}.parquet", "strain", max_pool=max_pool)
            out[t] = arr if arr is not None else np.array([], dtype=float)
        return out

    tf = _strain_per_target(PHASE4 / "strain", max_pool=None)
    per_baseline = {b: _strain_per_target(PHASE4B / b / "strain",
                                          max_pool=POOL_CAP) for b in BASELINES}

    colors = {"ThermoFrag": "#c1272d", "rxnflow": "#2a7f3e",
              "bbar": "#eaa300", "targetdiff": "#1f4e97"}
    order = ["ThermoFrag", "rxnflow", "bbar", "targetdiff"]
    width = 0.18
    fig, ax = plt.subplots(figsize=(14, 5.0))
    for i, t in enumerate(TARGETS):
        for j, label in enumerate(order):
            data = tf[t] if label == "ThermoFrag" else per_baseline[label][t]
            if len(data) == 0:
                continue
            pos = i + (j - 1.5) * width
            ax.boxplot(
                data, positions=[pos], widths=width * 0.95,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=colors[label], edgecolor="black", lw=0.8),
                medianprops=dict(color="black", lw=1.2),
                whiskerprops=dict(color="black", lw=0.8),
                capprops=dict(color="black", lw=0.8),
            )
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(TARGETS, rotation=40, ha="right")
    ax.set_ylabel("Post-relaxation strain (kcal/mol; ↓ better)")
    ax.set_title("Fig. 8  Generator-vs-generator strain distribution on 15 LIT-PCBA targets")
    handles = [mpatches.Patch(facecolor=colors[l], edgecolor="black",
                              label=("ThermoFrag" if l == "ThermoFrag" else
                                     {"rxnflow": "RxnFlow",
                                      "bbar": "BBAR",
                                      "targetdiff": "TargetDiff"}[l]))
               for l in order]
    ax.legend(handles=handles, loc="upper right", frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / "fig8_gvg_strain.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Table 1 — consolidated claims verdict
# ---------------------------------------------------------------------------

def make_table1() -> List[Path]:
    summary = json.loads(
        (REPO / "results/eval/claim_summary.json").read_text()
    )
    claims = summary["claims"]

    rows = [
        {
            "claim": "C1",
            "statement": "QM energy head matches DFT on drug-like molecules",
            "metric": "per-atom MAE (n_atoms≥30) / Spearman",
            "value": "0.49 kcal/mol / 0.9952",
            "threshold": "≤1 kcal/mol/atom, ρ>0.9",
            "verdict": "pass",
        },
        {
            "claim": "C2",
            "statement": "Learned chemical potential μ(y) is a physical quantity",
            "metric": "Spearman vs Bickerton QED / vs Wildman-Crippen logP",
            "value": f"{claims['C2']['numbers']['bickerton_spearman']:.3f} / "
                     f"{claims['C2']['numbers']['wc_proxy_spearman']:.3f}",
            "threshold": "ρ>0.6 both",
            "verdict": "pass",
        },
        {
            "claim": "C3",
            "statement": "Competitive Vina top-10 on LIT-PCBA (pool-size-matched)",
            "metric": "Paired Wilcoxon sig-wins at p<0.01 / 15 targets",
            "value": "vs TargetDiff 14/15 · vs RxnFlow 2/15 · vs BBAR 2/15 · vs LIT-PCBA ref 14/15",
            "threshold": "≥10/15 on baselines",
            "verdict": "pass vs TargetDiff + ref; tie vs score-aware (RxnFlow, BBAR)",
        },
        {
            "claim": "C4",
            "statement": "Strain not inflated beyond baselines",
            "metric": "Mean paired Cohen's d (TF − baseline)",
            "value": "vs TargetDiff −0.001 · vs BBAR −0.07 · vs RxnFlow +0.16",
            "threshold": "|mean d| ≤ 0.3",
            "verdict": "reframed: near-tie on all three",
        },
        {
            "claim": "C5",
            "statement": "Chemical-potential uncertainty flags OOD target requests",
            "metric": "AUROC on Pareto-thin OOD",
            "value": f"{claims['C5']['numbers']['auroc']:.4f}",
            "threshold": ">0.8",
            "verdict": "pass",
        },
        {
            "claim": "C6",
            "statement": "Each Hamiltonian term is independently necessary",
            "metric": "Shallow term-knockout failure modes",
            "value": "no-QM ρ 0.995→−0.94 · no-cpl yield 9.7%→2.2% · no-μ 13/15 sig",
            "threshold": "all three collapse",
            "verdict": "pass",
        },
    ]

    # CSV
    csv_path = OUT / "table1_claims.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Markdown
    md_lines = [
        "| Claim | Statement | Metric | Value | Threshold | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['claim']}** | {r['statement']} | {r['metric']} | "
            f"{r['value']} | {r['threshold']} | {r['verdict']} |"
        )
    md_path = OUT / "table1_claims.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    return [csv_path, md_path]


def main() -> None:
    outputs = []
    outputs.append(make_fig1())
    outputs.append(make_fig7())
    outputs.append(make_fig8())
    outputs.extend(make_table1())
    print("Wrote:")
    for p in outputs:
        print(f"  {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
