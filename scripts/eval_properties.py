"""Phase-2 exit-criterion evaluator.

Compares the distribution of generated fragment-assembly graphs against the
ZINC reference distribution on:

  1. Fragment-ID marginal (the quantity V directly shapes).
  2. Per-fragment property marginals (logP, QED, MW, TPSA): for each
     generated/ZINC graph, we sum logP/MW/TPSA and average QED across its
     fragment's core SMILES. This is a coarse proxy for molecular properties
     but is well-defined without a graph-to-SMILES re-assembly step (which is
     a Phase-3 deliverable), and it tracks the shape of the distribution
     enough to flag large deviations.

Produces:
  - results/eval/phase2/property_kl.json
  - results/eval/phase2/hist_{frag,logp,mw,qed,tpsa}.png

Phase-2 exit criterion from docs/MILESTONES.md:
  ``KL(generated || ZINC) on first three property marginals < 0.05``

Usage::

    python scripts/eval_properties.py \
        --samples results/eval/phase2/samples.pkl \
        --ref data/processed/zinc_unconditional.lmdb \
        --library data/processed/fragment_library.parquet \
        --out results/eval/phase2/
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from thermofrag.data.zinc_fragments import ZINCFragmentDataset


# -----------------------------------------------------------------------------
# Fragment property table
# -----------------------------------------------------------------------------


def _compute_frag_properties(library_path: Path) -> pd.DataFrame:
    """Compute logP/QED/MW/TPSA for each fragment's core SMILES (no anchor).

    Returns a dataframe indexed by frag_id, with NaN for the UNK row.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Crippen, Descriptors, QED

    RDLogger.DisableLog("rdApp.*")
    lib = pd.read_parquet(library_path).sort_values("frag_id").reset_index(drop=True)

    def _props(smi: str) -> tuple[float, float, float, float]:
        if smi == "__UNK__":
            return (np.nan, np.nan, np.nan, np.nan)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return (np.nan, np.nan, np.nan, np.nan)
        return (
            float(Crippen.MolLogP(mol)),
            float(QED.qed(mol)),
            float(Descriptors.MolWt(mol)),
            float(Descriptors.TPSA(mol)),
        )

    rows = [_props(s) for s in lib["fragment_smi"]]
    prop = pd.DataFrame(rows, columns=["logp", "qed", "mw", "tpsa"])
    return pd.concat([lib, prop], axis=1).set_index("frag_id")


def _aggregate_props(frag_ids: Iterable[int], prop_df: pd.DataFrame) -> tuple[float, float, float, float]:
    vals = prop_df.reindex(list(frag_ids))
    # Sum MW, logP, TPSA (extensive-ish). Mean QED (intensive score). Ignore NaN from UNK.
    logp = float(vals["logp"].sum(skipna=True))
    qed = float(vals["qed"].mean(skipna=True)) if not vals["qed"].isna().all() else 0.0
    mw = float(vals["mw"].sum(skipna=True))
    tpsa = float(vals["tpsa"].sum(skipna=True))
    return logp, qed, mw, tpsa


# -----------------------------------------------------------------------------
# Distributions and KL
# -----------------------------------------------------------------------------


def _kl_discrete(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _binned_kl(gen: np.ndarray, ref: np.ndarray, bins: int = 40) -> dict:
    lo = float(min(gen.min(), ref.min()))
    hi = float(max(gen.max(), ref.max()))
    edges = np.linspace(lo, hi, bins + 1)
    hg, _ = np.histogram(gen, bins=edges, density=False)
    hr, _ = np.histogram(ref, bins=edges, density=False)
    kl_gr = _kl_discrete(hg.astype(np.float64), hr.astype(np.float64))
    kl_rg = _kl_discrete(hr.astype(np.float64), hg.astype(np.float64))
    return {
        "range": [lo, hi],
        "edges": edges.tolist(),
        "hist_gen": hg.tolist(),
        "hist_ref": hr.tolist(),
        "kl_gen_to_ref": kl_gr,
        "kl_ref_to_gen": kl_rg,
    }


def _fragment_marginal_kl(
    gen_samples: list[dict], ref_ds: ZINCFragmentDataset, n_frag: int, n_ref: int
) -> tuple[float, np.ndarray, np.ndarray]:
    gc: Counter = Counter()
    for s in gen_samples:
        gc.update(s["frag_id"])
    rc: Counter = Counter()
    n_ref = min(n_ref, len(ref_ds))
    for i in range(n_ref):
        rc.update(ref_ds[i].frag_id.tolist())

    gp = np.array([gc.get(i, 0) for i in range(n_frag)], dtype=np.float64)
    rp = np.array([rc.get(i, 0) for i in range(n_frag)], dtype=np.float64)
    return _kl_discrete(gp, rp), gp, rp


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _plot_overlay(gen: np.ndarray, ref: np.ndarray, name: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    lo = float(min(gen.min(), ref.min()))
    hi = float(max(gen.max(), ref.max()))
    bins = np.linspace(lo, hi, 41)
    ax.hist(ref, bins=bins, alpha=0.45, density=True, label="ZINC (ref)", color="C0")
    ax.hist(gen, bins=bins, alpha=0.45, density=True, label="generated", color="C3")
    ax.set_xlabel(name)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_frag_marginal(gp: np.ndarray, rp: np.ndarray, out_path: Path, top: int = 40) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Rank fragments by reference frequency.
    order = np.argsort(-rp)[:top]
    rp_top = rp[order] / max(rp.sum(), 1)
    gp_top = gp[order] / max(gp.sum(), 1)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.bar(x - 0.2, rp_top, width=0.4, label="ZINC (ref)", color="C0", alpha=0.8)
    ax.bar(x + 0.2, gp_top, width=0.4, label="generated", color="C3", alpha=0.8)
    ax.set_xlabel(f"fragment rank (top-{top} by ref frequency)")
    ax.set_ylabel("prob")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True, help=".pkl from sample_unconditional.py")
    p.add_argument("--ref", type=Path, required=True, help="ZINC LMDB")
    p.add_argument("--library", type=Path, required=True, help="fragment_library.parquet")
    p.add_argument("--out", type=Path, required=True, help="output dir")
    p.add_argument("--ref-n", type=int, default=5000, help="max reference mols to aggregate")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    with open(args.samples, "rb") as f:
        sampled = pickle.load(f)
    samples = sampled["samples"]
    print(f"[eval] loaded {len(samples)} generated samples  accept_rate={sampled.get('accept_rate')}")

    print(f"[eval] computing per-fragment properties from {args.library}")
    prop_df = _compute_frag_properties(args.library)
    print(f"[eval]   library size={len(prop_df)}  valid props={int(prop_df['logp'].notna().sum())}")

    ref_ds = ZINCFragmentDataset(args.ref, split="train")
    n_frag = ref_ds.n_fragments

    # --- Fragment marginal ---------------------------------------------------
    print(f"[eval] computing fragment marginal on {min(args.ref_n, len(ref_ds))} ref mols")
    kl_frag, gp, rp = _fragment_marginal_kl(samples, ref_ds, n_frag=n_frag, n_ref=args.ref_n)
    _plot_frag_marginal(gp, rp, args.out / "hist_frag.png")
    print(f"[eval]   KL(gen||ref) fragment marginal = {kl_frag:.4f}")

    # --- Per-graph property marginals ---------------------------------------
    print(f"[eval] aggregating per-graph properties")
    gen_logp, gen_qed, gen_mw, gen_tpsa = [], [], [], []
    for s in samples:
        l, q, m, t = _aggregate_props(s["frag_id"], prop_df)
        gen_logp.append(l)
        gen_qed.append(q)
        gen_mw.append(m)
        gen_tpsa.append(t)

    ref_logp, ref_qed, ref_mw, ref_tpsa = [], [], [], []
    for i in range(min(args.ref_n, len(ref_ds))):
        fids = ref_ds[i].frag_id.tolist()
        l, q, m, t = _aggregate_props(fids, prop_df)
        ref_logp.append(l)
        ref_qed.append(q)
        ref_mw.append(m)
        ref_tpsa.append(t)

    gen_logp = np.asarray(gen_logp)
    gen_qed = np.asarray(gen_qed)
    gen_mw = np.asarray(gen_mw)
    gen_tpsa = np.asarray(gen_tpsa)
    ref_logp = np.asarray(ref_logp)
    ref_qed = np.asarray(ref_qed)
    ref_mw = np.asarray(ref_mw)
    ref_tpsa = np.asarray(ref_tpsa)

    kl_logp = _binned_kl(gen_logp, ref_logp)
    kl_qed = _binned_kl(gen_qed, ref_qed)
    kl_mw = _binned_kl(gen_mw, ref_mw)
    kl_tpsa = _binned_kl(gen_tpsa, ref_tpsa)

    _plot_overlay(gen_logp, ref_logp, "logP (frag-sum)", args.out / "hist_logp.png")
    _plot_overlay(gen_qed, ref_qed, "QED (frag-mean)", args.out / "hist_qed.png")
    _plot_overlay(gen_mw, ref_mw, "MW (frag-sum)", args.out / "hist_mw.png")
    _plot_overlay(gen_tpsa, ref_tpsa, "TPSA (frag-sum)", args.out / "hist_tpsa.png")

    summary = {
        "n_samples": len(samples),
        "n_ref_used": min(args.ref_n, len(ref_ds)),
        "accept_rate": sampled.get("accept_rate"),
        "mh_steps": sampled.get("mh_steps"),
        "kl_fragment_marginal": float(kl_frag),
        "kl_logp_gen_to_ref": kl_logp["kl_gen_to_ref"],
        "kl_qed_gen_to_ref": kl_qed["kl_gen_to_ref"],
        "kl_mw_gen_to_ref": kl_mw["kl_gen_to_ref"],
        "kl_tpsa_gen_to_ref": kl_tpsa["kl_gen_to_ref"],
        "exit_criterion_primary_three": {
            "logp": kl_logp["kl_gen_to_ref"],
            "qed": kl_qed["kl_gen_to_ref"],
            "mw": kl_mw["kl_gen_to_ref"],
            "all_below_0.05": all(
                v < 0.05
                for v in [
                    kl_logp["kl_gen_to_ref"],
                    kl_qed["kl_gen_to_ref"],
                    kl_mw["kl_gen_to_ref"],
                ]
            ),
        },
        "note": (
            "Properties are aggregated over fragment core SMILES (sum of logP/MW/TPSA; "
            "mean QED). This is a proxy for molecular-level properties; full assembly "
            "requires Phase-3 proposal kernels."
        ),
    }

    (args.out / "property_kl.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"[eval] wrote {args.out / 'property_kl.json'}")


if __name__ == "__main__":
    main()
