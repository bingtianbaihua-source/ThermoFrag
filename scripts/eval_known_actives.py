"""Phase-7 task 3 — Tanimoto-to-known-actives recovery.

For each (generator, target):
  * Load decoded SMILES + Vina scores (status='ok' rows).
  * Compute the Tanimoto similarity from each generated molecule to its
    nearest LIT-PCBA active (radius-2 Morgan, 2048 bits).
  * Compare TF top-10 distance distribution against a random ChEMBL
    background pulled from ``data/processed/chembl_conditional.lmdb``.

Pre-registered thresholds (``docs/validation/03_known_actives.md``):
  * Novelty: TF top-10 ``max_sim`` < 0.7 on every target.
  * Coverage: TF top-10 closer to actives than random ChEMBL on
    >= 10/15 targets at one-sided Mann-Whitney p < 0.05.

Outputs::

    results/eval/phase7/actives/<target>.parquet
        target, generator, chain_idx, smiles, vina_score,
        tanimoto_to_nearest_active, nearest_active_smi,
        is_top10, is_top30
    results/eval/phase7/actives/<target>_chembl_random.parquet
        target, smiles, tanimoto_to_nearest_active
    results/eval/phase7/AGGREGATE/03_actives_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("known_actives")

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

# Disallowed atomic numbers for active molecules (organometals, salts).
_METAL_ATOMS = {3, 4, 11, 12, 13, 19, 20, 24, 25, 26, 27, 28, 29, 30,
                33, 34, 38, 47, 48, 50, 51, 56, 78, 79, 80, 81, 82, 83}


def _druglike_active(mol) -> bool:
    """Filter LIT-PCBA actives to drug-like organic (MW<700, no metals)."""
    from rdkit.Chem import Descriptors
    if mol is None:
        return False
    if any(a.GetAtomicNum() in _METAL_ATOMS for a in mol.GetAtoms()):
        return False
    if Descriptors.MolWt(mol) > 700:
        return False
    return True


_MORGAN_GEN = {}


def _morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    key = (radius, n_bits)
    if key not in _MORGAN_GEN:
        _MORGAN_GEN[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    fp = _MORGAN_GEN[key].GetFingerprint(mol)
    return mol, fp


def _build_actives(target: str, actives_dir: Path,
                   radius: int, n_bits: int) -> tuple[list, list]:
    """Return (smiles_list, fp_list) for drug-like actives of one target."""
    smi_path = actives_dir / f"{target}.smi"
    if not smi_path.exists():
        raise FileNotFoundError(f"missing actives file: {smi_path}")
    smis: list[str] = []
    fps: list = []
    n_total = 0
    n_drop = 0
    for line in smi_path.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        n_total += 1
        smi = parts[0]
        mol, fp = _morgan_fp(smi, radius=radius, n_bits=n_bits)
        if mol is None or not _druglike_active(mol) or fp is None:
            n_drop += 1
            continue
        smis.append(smi)
        fps.append(fp)
    logger.info("  %s actives: kept %d / %d (dropped %d)",
                target, len(smis), n_total, n_drop)
    return smis, fps


def _max_sim_to_actives(smi: str, actives_smis, actives_fps, radius, n_bits):
    """Returns (max_tanimoto, nearest_active_smiles) or (None, None)."""
    from rdkit import DataStructs
    _, fp = _morgan_fp(smi, radius=radius, n_bits=n_bits)
    if fp is None or not actives_fps:
        return None, None
    sims = DataStructs.BulkTanimotoSimilarity(fp, actives_fps)
    j = int(np.argmax(sims))
    return float(sims[j]), actives_smis[j]


def _load_random_chembl(lmdb_path: Path, n: int, seed: int) -> list[str]:
    import lmdb
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False,
                    max_readers=1, subdir=False)
    with env.begin() as txn:
        stat = txn.stat()
        n_entries = stat["entries"]
        rng = random.Random(seed)
        ks = sorted(rng.sample(range(n_entries),
                               k=min(n, n_entries)))
        smis = []
        cur = txn.cursor()
        # Scan once, picking the chosen indices.
        ks_set = set(ks)
        for i, (_, v) in enumerate(cur):
            if i in ks_set:
                try:
                    obj = pickle.loads(v)
                    if isinstance(obj, dict) and "smiles" in obj:
                        smis.append(obj["smiles"])
                except Exception:
                    pass
            if i >= max(ks_set):
                break
    return smis


def _score_pool(smiles_list: list[str], actives_smis, actives_fps,
                radius: int, n_bits: int) -> list[tuple[Optional[float], Optional[str]]]:
    return [_max_sim_to_actives(s, actives_smis, actives_fps, radius, n_bits)
            for s in smiles_list]


def _read_decoded(target: str, gen: str, repo_root: Path) -> pd.DataFrame:
    base = repo_root / GENERATOR_DIR[gen]
    decoded = base / "decoded" / f"{target}.parquet"
    vina = base / "vina" / f"{target}.parquet"
    dec_df = pd.read_parquet(decoded)
    vina_df = pd.read_parquet(vina)[["chain_idx", "vina_score", "status"]]
    df = dec_df.merge(vina_df, on="chain_idx", how="left")
    df = df[df["status"].fillna("missing") == "ok"].reset_index(drop=True)
    return df


def evaluate_target(target: str, actives_dir: Path, repo_root: Path,
                    out_dir: Path, random_smis: list[str],
                    novelty_threshold: float, top_k_label: int,
                    radius: int, n_bits: int) -> dict:
    out_pq = out_dir / f"{target}.parquet"
    rand_pq = out_dir / f"{target}_chembl_random.parquet"

    actives_smis, actives_fps = _build_actives(
        target, actives_dir, radius=radius, n_bits=n_bits)

    rows = []
    per_gen: dict[str, list[float]] = {}

    for gen in GENERATORS:
        try:
            df = _read_decoded(target, gen, repo_root)
        except FileNotFoundError as exc:
            logger.warning("  %s/%s: %s", gen, target, exc)
            continue
        if len(df) == 0:
            logger.warning("  %s/%s: no ok rows", gen, target)
            continue

        # Sort by vina_score asc; mark top-K membership.
        df = df.sort_values("vina_score", ascending=True).reset_index(drop=True)
        df["rank"] = np.arange(len(df))
        df["is_top10"] = df["rank"] < 10
        df["is_top30"] = df["rank"] < 30

        scored = _score_pool(df["smiles"].tolist(),
                             actives_smis, actives_fps, radius, n_bits)
        sims = [s[0] for s in scored]
        nearest = [s[1] for s in scored]

        gen_rows = pd.DataFrame({
            "target": target,
            "generator": gen,
            "chain_idx": df["chain_idx"],
            "smiles": df["smiles"],
            "vina_score": df["vina_score"],
            "tanimoto_to_nearest_active": sims,
            "nearest_active_smi": nearest,
            "is_top10": df["is_top10"],
            "is_top30": df["is_top30"],
        })
        rows.append(gen_rows)

        top10_sims = [s for s, t in zip(sims, df["is_top10"]) if t and s is not None]
        per_gen[gen] = top10_sims

    full = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    full.to_parquet(out_pq)

    # Random ChEMBL background — score once per target.
    rand_scored = _score_pool(random_smis, actives_smis, actives_fps,
                              radius, n_bits)
    rand_sims = [s[0] for s in rand_scored if s[0] is not None]
    rand_df = pd.DataFrame({
        "target": target,
        "smiles": random_smis,
        "tanimoto_to_nearest_active": [s[0] for s in rand_scored],
    })
    rand_df.to_parquet(rand_pq)

    # Stats: per-generator vs random ChEMBL, one-sided Mann-Whitney.
    from scipy.stats import mannwhitneyu
    stats: dict[str, dict] = {}
    for gen, sims in per_gen.items():
        if not sims or not rand_sims:
            stats[gen] = {"n_top10": len(sims),
                          "max_sim": (max(sims) if sims else None),
                          "median_sim": (float(np.median(sims)) if sims else None),
                          "p_value_vs_chembl": None,
                          "novelty_pass": None,
                          "coverage_pass": None}
            continue
        try:
            u, p = mannwhitneyu(sims, rand_sims, alternative="greater")
        except ValueError:
            u, p = float("nan"), float("nan")
        max_sim = float(max(sims))
        novelty = max_sim < novelty_threshold
        coverage = (p < 0.05)
        stats[gen] = {
            "n_top10": len(sims),
            "max_sim": max_sim,
            "median_sim": float(np.median(sims)),
            "median_sim_chembl": float(np.median(rand_sims)),
            "p_value_vs_chembl": float(p),
            "novelty_pass": bool(novelty),
            "coverage_pass": bool(coverage),
        }

    return {
        "target": target,
        "n_actives_kept": len(actives_smis),
        "n_random_chembl": len(rand_sims),
        "per_gen": stats,
    }


def write_summary(results: list[dict], summary_path: Path,
                  novelty_threshold: float):
    """Aggregate the per-target stats into the JSON contract."""
    per_target = {}
    n_targets = len(results)
    novelty_counts = {g: 0 for g in GENERATORS}
    coverage_counts = {g: 0 for g in GENERATORS}
    for r in results:
        per_target[r["target"]] = r
        for g, s in r["per_gen"].items():
            if s.get("novelty_pass"):
                novelty_counts[g] += 1
            if s.get("coverage_pass"):
                coverage_counts[g] += 1

    thresholds = {
        f"novelty_{g}": {
            "target": f"max_sim < {novelty_threshold} on all {n_targets}",
            "observed": f"{novelty_counts[g]}/{n_targets}",
            "pass": novelty_counts[g] == n_targets,
        } for g in GENERATORS
    }
    thresholds.update({
        f"coverage_{g}": {
            "target": "≥ 10/15 targets with p<0.05 vs random ChEMBL",
            "observed": f"{coverage_counts[g]}/{n_targets}",
            "pass": coverage_counts[g] >= 10,
        } for g in GENERATORS
    })

    summary = {
        "task_id": "03_known_actives",
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "thresholds": thresholds,
        "per_target": per_target,
        "novelty_threshold": novelty_threshold,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("summary → %s", summary_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actives_dir", type=Path,
                   default=Path("data/external/litpcba_actives"))
    p.add_argument("--chembl_lmdb", type=Path,
                   default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--random_size", type=int, default=5000)
    p.add_argument("--novelty_threshold", type=float, default=0.7)
    p.add_argument("--out_root", type=Path,
                   default=Path("results/eval/phase7/actives"))
    p.add_argument("--target", default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n_bits", type=int, default=2048)
    p.add_argument("--repo_root", type=Path,
                   default=Path("/home/zhao/code/ThermoFrag"))
    p.add_argument("--rebuild_random", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    targets = ALL_TARGETS if args.target == "all" else [args.target]

    repo_root = args.repo_root.resolve()
    actives_dir = (args.actives_dir if args.actives_dir.is_absolute()
                   else repo_root / args.actives_dir)
    chembl_lmdb = (args.chembl_lmdb if args.chembl_lmdb.is_absolute()
                   else repo_root / args.chembl_lmdb)
    out_dir = (args.out_root if args.out_root.is_absolute()
               else repo_root / args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache the random ChEMBL pool — same draw across targets so the
    # null distribution is identical (only the actives reference changes).
    rand_cache = out_dir / "random_chembl_pool.smi"
    if rand_cache.exists() and not args.rebuild_random:
        random_smis = [l.strip() for l in rand_cache.read_text().splitlines()
                       if l.strip()]
        logger.info("loaded %d cached random ChEMBL SMILES", len(random_smis))
    else:
        random_smis = _load_random_chembl(chembl_lmdb, args.random_size, args.seed)
        rand_cache.write_text("\n".join(random_smis) + "\n")
        logger.info("sampled %d random ChEMBL SMILES → %s",
                    len(random_smis), rand_cache)

    t0 = time.time()
    results = []
    for target in targets:
        logger.info("== %s ==", target)
        try:
            r = evaluate_target(
                target=target, actives_dir=actives_dir, repo_root=repo_root,
                out_dir=out_dir, random_smis=random_smis,
                novelty_threshold=args.novelty_threshold,
                top_k_label=10, radius=args.radius, n_bits=args.n_bits)
        except Exception as exc:
            logger.exception("  %s failed: %s", target, exc)
            continue
        results.append(r)
        for g, s in r["per_gen"].items():
            logger.info(
                "  %s top10 max=%s med=%s p=%s nov=%s cov=%s",
                g,
                f"{s['max_sim']:.3f}" if s.get("max_sim") is not None else "NA",
                f"{s['median_sim']:.3f}" if s.get("median_sim") is not None else "NA",
                f"{s['p_value_vs_chembl']:.3g}" if s.get("p_value_vs_chembl") is not None else "NA",
                s.get("novelty_pass"), s.get("coverage_pass"))

    summary_path = repo_root / "results/eval/phase7/AGGREGATE/03_known_actives_summary.json"
    write_summary(results, summary_path, args.novelty_threshold)
    logger.info("done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
