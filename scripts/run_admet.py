"""Phase-7 task 6 — ADMET property prediction on top-N pool.

Pulls the top-N (default 30) generated molecules per (generator, target)
ranked by Vina score, runs them through ADMET-AI, then applies five
rule-based RDKit filters (Lipinski, Veber, PAINS, Ghose, Egan).

Outputs::

    results/eval/phase7/admet/<gen>/predictions.parquet
        generator, target, chain_idx, smiles, vina_score, <40 ADMET cols>,
        lipinski_pass, veber_pass, pains_pass, ghose_pass, egan_pass
    results/eval/phase7/admet/summary.parquet
        per-(generator,target) pass-rates and ADMET medians
    results/eval/phase7/AGGREGATE/06_admet_summary.json

Run inside the ``tf-admet`` conda env (admet-ai==1.4.0). Defaults pick
the GPU automatically; we expose ``--device`` to override.

Pre-registered thresholds (``docs/validation/06_admet.md``):
  * TF Lipinski rate ≥ 80 % (averaged over 15 targets)
  * TF Veber rate ≥ 80 %
  * TF PAINS rate ≥ 90 %
  * TF rates ≥ TargetDiff rates (parity, descriptive)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("admet")

ALL_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]

GENERATORS = ["thermofrag", "targetdiff", "rxnflow", "bbar"]

GENERATOR_DIR = {
    "thermofrag": Path("results/eval/phase4"),
    "targetdiff": Path("results/eval/phase4_baselines/targetdiff"),
    "rxnflow":    Path("results/eval/phase4_baselines/rxnflow"),
    "bbar":       Path("results/eval/phase4_baselines/bbar"),
}


def build_input(top_k: int, repo_root: Path,
                pool_cap: int = 100) -> pd.DataFrame:
    """Top-K rows per (gen, target), filtered by chain_idx < pool_cap."""
    rows = []
    for gen in GENERATORS:
        base = repo_root / GENERATOR_DIR[gen]
        for target in ALL_TARGETS:
            vina_pq = base / "vina" / f"{target}.parquet"
            if not vina_pq.exists():
                logger.warning("missing vina parquet: %s", vina_pq)
                continue
            df = pd.read_parquet(vina_pq)
            df = df[df["status"] == "ok"].copy()
            df = df[df["chain_idx"] < pool_cap]
            df = df.sort_values("vina_score").head(top_k)
            df["generator"] = gen
            df["target"] = target
            rows.append(df[["generator", "target", "chain_idx",
                            "smiles", "vina_score"]])
    if not rows:
        raise RuntimeError("no input rows assembled — check vina paths")
    return pd.concat(rows, ignore_index=True)


def canonicalize(smiles_list):
    from rdkit import Chem
    out = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        out.append(Chem.MolToSmiles(m) if m is not None else s)
    return out


def compute_rule_filters(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    pains_cat = FilterCatalog(FilterCatalogParams(
        FilterCatalogParams.FilterCatalogs.PAINS))

    cols = {k: [] for k in ["lipinski_pass", "veber_pass", "pains_pass",
                            "ghose_pass", "egan_pass"]}
    for s in smiles_list:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            for k in cols:
                cols[k].append(False)
            continue
        mw = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        hba = rdMolDescriptors.CalcNumHBA(m)
        hbd = rdMolDescriptors.CalcNumHBD(m)
        rotb = rdMolDescriptors.CalcNumRotatableBonds(m)
        tpsa = Descriptors.TPSA(m)
        cols["lipinski_pass"].append(mw <= 500 and logp <= 5
                                     and hba <= 10 and hbd <= 5)
        cols["veber_pass"].append(rotb <= 10 and tpsa <= 140)
        cols["pains_pass"].append(not pains_cat.HasMatch(m))
        cols["ghose_pass"].append(160 <= mw <= 480 and -0.4 <= logp <= 5.6)
        cols["egan_pass"].append(logp <= 5.88 and tpsa <= 131.6)
    return pd.DataFrame(cols)


def run_admet_ai(smiles_list, batch_size: int = 64):
    from admet_ai import ADMETModel
    logger.info("loading ADMETModel (~10 s init, ~600 MB weights)...")
    model = ADMETModel()
    logger.info("predicting %d SMILES (batch_size=%d)...",
                len(smiles_list), batch_size)
    preds = model.predict(smiles=smiles_list)
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    preds = preds.reset_index(drop=True)
    return preds


def aggregate(out: pd.DataFrame) -> pd.DataFrame:
    """Per-(gen, target) pass-rates and ADMET medians."""
    filter_cols = ["lipinski_pass", "veber_pass", "pains_pass",
                   "ghose_pass", "egan_pass"]
    median_cols = [c for c in
                   ["hERG", "BBB_Martins", "CYP3A4_Veith", "AMES",
                    "DILI", "Carcinogens_Lagunin",
                    "Lipophilicity_AstraZeneca", "Solubility_AqSolDB",
                    "HIA_Hou"]
                   if c in out.columns]
    g = out.groupby(["generator", "target"], dropna=False)
    rates = g[filter_cols].mean()
    medians = g[median_cols].median() if median_cols else pd.DataFrame()
    counts = g.size().to_frame("n")
    summary = pd.concat([counts, rates, medians], axis=1).reset_index()
    return summary


def write_aggregate_summary(summary_df: pd.DataFrame,
                            agg_path: Path,
                            thresholds: dict[str, float]) -> dict:
    per_gen_macro = (summary_df
                     .groupby("generator")[["lipinski_pass", "veber_pass",
                                            "pains_pass", "ghose_pass",
                                            "egan_pass"]]
                     .mean())
    obs = {g: per_gen_macro.loc[g].to_dict() if g in per_gen_macro.index else {}
           for g in GENERATORS}

    th = {}
    tf = obs.get("thermofrag", {})
    td = obs.get("targetdiff", {})
    th["tf_lipinski"] = {
        "target": f">= {thresholds['lipinski']:.2f}",
        "observed": float(tf.get("lipinski_pass", 0.0)),
        "pass": tf.get("lipinski_pass", 0.0) >= thresholds["lipinski"],
    }
    th["tf_veber"] = {
        "target": f">= {thresholds['veber']:.2f}",
        "observed": float(tf.get("veber_pass", 0.0)),
        "pass": tf.get("veber_pass", 0.0) >= thresholds["veber"],
    }
    th["tf_pains"] = {
        "target": f">= {thresholds['pains']:.2f}",
        "observed": float(tf.get("pains_pass", 0.0)),
        "pass": tf.get("pains_pass", 0.0) >= thresholds["pains"],
    }
    th["tf_vs_targetdiff_lipinski"] = {
        "target": ">= TargetDiff (parity)",
        "observed": (float(tf.get("lipinski_pass", 0.0))
                     - float(td.get("lipinski_pass", 0.0))),
        "pass": (tf.get("lipinski_pass", 0.0)
                 >= td.get("lipinski_pass", 0.0)),
    }
    th["tf_vs_targetdiff_veber"] = {
        "target": ">= TargetDiff (parity)",
        "observed": (float(tf.get("veber_pass", 0.0))
                     - float(td.get("veber_pass", 0.0))),
        "pass": (tf.get("veber_pass", 0.0)
                 >= td.get("veber_pass", 0.0)),
    }
    th["tf_vs_targetdiff_pains"] = {
        "target": ">= TargetDiff (parity)",
        "observed": (float(tf.get("pains_pass", 0.0))
                     - float(td.get("pains_pass", 0.0))),
        "pass": (tf.get("pains_pass", 0.0)
                 >= td.get("pains_pass", 0.0)),
    }

    summary = {
        "task_id": "06_admet",
        "completed_utc": datetime.now(timezone.utc)
                         .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thresholds": th,
        "per_generator_macro_mean": {g: obs[g] for g in GENERATORS},
        "thresholds_target": thresholds,
    }
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--pool_cap", type=int, default=100)
    p.add_argument("--out_root", type=Path,
                   default=Path("results/eval/phase7/admet"))
    p.add_argument("--repo_root", type=Path,
                   default=Path("/home/zhao/code/ThermoFrag"))
    p.add_argument("--lipinski_target", type=float, default=0.80)
    p.add_argument("--veber_target",    type=float, default=0.80)
    p.add_argument("--pains_target",    type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    repo_root = args.repo_root.resolve()
    out_root = (args.out_root if args.out_root.is_absolute()
                else repo_root / args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    inp = build_input(top_k=args.top_k, repo_root=repo_root,
                      pool_cap=args.pool_cap)
    logger.info("input rows: %d (per gen counts: %s)",
                len(inp), inp.groupby("generator").size().to_dict())
    inp["smiles"] = canonicalize(inp["smiles"].tolist())

    if args.dry_run:
        logger.info("[dry] would predict %d SMILES", len(inp))
        return

    t0 = time.time()
    preds = run_admet_ai(inp["smiles"].tolist())
    logger.info("admet-ai predict: %d rows × %d cols in %.1f s",
                len(preds), len(preds.columns), time.time() - t0)

    if len(preds) != len(inp):
        raise RuntimeError(
            f"admet-ai returned {len(preds)} rows for {len(inp)} input")

    filt = compute_rule_filters(inp["smiles"].tolist())
    out = pd.concat([inp.reset_index(drop=True),
                     preds.reset_index(drop=True),
                     filt.reset_index(drop=True)], axis=1)

    # Per-generator parquet
    for gen in GENERATORS:
        sub = out[out["generator"] == gen]
        if len(sub) == 0:
            continue
        gen_dir = out_root / gen
        gen_dir.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(gen_dir / "predictions.parquet")
        logger.info("  wrote %s (%d rows)", gen_dir / "predictions.parquet",
                    len(sub))

    summary_df = aggregate(out)
    summary_df.to_parquet(out_root / "summary.parquet")
    logger.info("wrote %s", out_root / "summary.parquet")

    agg_path = (repo_root / "results/eval/phase7/AGGREGATE"
                / "06_admet_summary.json")
    thresholds = {
        "lipinski": args.lipinski_target,
        "veber": args.veber_target,
        "pains": args.pains_target,
    }
    summary = write_aggregate_summary(summary_df, agg_path, thresholds)
    logger.info("aggregate → %s", agg_path)
    logger.info("TF macro means: lipinski=%.3f veber=%.3f pains=%.3f",
                summary["per_generator_macro_mean"]["thermofrag"]
                       .get("lipinski_pass", float("nan")),
                summary["per_generator_macro_mean"]["thermofrag"]
                       .get("veber_pass", float("nan")),
                summary["per_generator_macro_mean"]["thermofrag"]
                       .get("pains_pass", float("nan")))


if __name__ == "__main__":
    main()
