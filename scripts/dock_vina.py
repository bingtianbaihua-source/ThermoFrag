"""Vina docking driver for the Phase-4 decoded SMILES (claim C3).

For each LIT-PCBA target in ``data/external/receptors/<target>/``:

  1. Load decoded SMILES from ``results/eval/phase4/decoded/<target>.parquet``.
  2. Embed + MMFF94 minimize with RDKit, write a transient SDF.
  3. Convert to PDBQT via meeko ``mk_prepare_ligand.py`` (gasteiger charges).
  4. Run Vina against the prebuilt ``receptor.pdbqt`` + ``box.json``.
  5. Parse the top mode's affinity (kcal/mol).

Output::

    results/eval/phase4/vina/<target>.parquet
        target, chain_idx, smiles, vina_score, status
    results/eval/phase4/vina/summary.csv

Parallelism: a process pool across ligands. Vina itself is invoked with
``--cpu 1`` so workers do not oversubscribe.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dock")

_SCORE_RE = re.compile(r"^\s*1\s+(-?\d+\.\d+)", re.MULTILINE)


@dataclass
class DockResult:
    chain_idx: int
    smiles: str
    vina_score: Optional[float]
    status: str


def _embed_and_write_sdf(smiles: str, sdf_path: Path, seed: int = 42) -> bool:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass
    w = Chem.SDWriter(str(sdf_path))
    w.write(mol)
    w.close()
    return True


def _prep_ligand_pdbqt(sdf_path: Path, pdbqt_path: Path) -> bool:
    cmd = ["mk_prepare_ligand.py", "-i", str(sdf_path), "-o", str(pdbqt_path),
           "--charge_model", "gasteiger"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return res.returncode == 0 and pdbqt_path.exists() and pdbqt_path.stat().st_size > 100


def _run_vina(lig_pdbqt: Path, rec_pdbqt: Path, box: dict,
              exhaustiveness: int = 8, num_modes: int = 5,
              cpu: int = 1) -> Optional[float]:
    out_pdbqt = lig_pdbqt.with_suffix(".dock.pdbqt")
    cmd = [
        "vina",
        "--receptor", str(rec_pdbqt),
        "--ligand", str(lig_pdbqt),
        "--center_x", str(box["center"][0]),
        "--center_y", str(box["center"][1]),
        "--center_z", str(box["center"][2]),
        "--size_x", str(box["size"][0]),
        "--size_y", str(box["size"][1]),
        "--size_z", str(box["size"][2]),
        "--cpu", str(cpu),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--out", str(out_pdbqt),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        return None
    m = _SCORE_RE.search(res.stdout)
    if not m:
        return None
    return float(m.group(1))


def _dock_one(args):
    chain_idx, smiles, rec_pdbqt_str, box, exhaustiveness = args
    rec_pdbqt = Path(rec_pdbqt_str)
    with tempfile.TemporaryDirectory(prefix="vina_") as tmp:
        tmp = Path(tmp)
        sdf = tmp / "lig.sdf"
        pdbqt = tmp / "lig.pdbqt"
        if not _embed_and_write_sdf(smiles, sdf):
            return DockResult(chain_idx, smiles, None, "embed_failed")
        if not _prep_ligand_pdbqt(sdf, pdbqt):
            return DockResult(chain_idx, smiles, None, "prep_failed")
        try:
            score = _run_vina(pdbqt, rec_pdbqt, box, exhaustiveness=exhaustiveness)
        except subprocess.TimeoutExpired:
            return DockResult(chain_idx, smiles, None, "vina_timeout")
        if score is None:
            return DockResult(chain_idx, smiles, None, "vina_failed")
    return DockResult(chain_idx, smiles, float(score), "ok")


def dock_target(target: str, decoded_pq: Path, rec_dir: Path,
                out_pq: Path, workers: int, exhaustiveness: int,
                limit: Optional[int]) -> pd.DataFrame:
    df = pd.read_parquet(decoded_pq)
    valid = df[df["smiles"].notna()].reset_index(drop=True)
    if limit and len(valid) > limit:
        valid = valid.head(limit)
    if len(valid) == 0:
        logger.warning("no valid SMILES for %s", target)
        return pd.DataFrame()

    box_path = rec_dir / "box.json"
    box = json.loads(box_path.read_text())
    rec_pdbqt = rec_dir / "receptor.pdbqt"

    jobs = [
        (int(row["chain_idx"]), str(row["smiles"]), str(rec_pdbqt), box, exhaustiveness)
        for _, row in valid.iterrows()
    ]
    logger.info("%s: %d ligands, %d workers", target, len(jobs), workers)

    results: list[DockResult] = [None] * len(jobs)
    if workers <= 1:
        for i, j in enumerate(jobs):
            results[i] = _dock_one(j)
            if (i + 1) % 25 == 0:
                logger.info("  %s: %d / %d", target, i + 1, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_dock_one, j): idx for idx, j in enumerate(jobs)}
            done = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    sm = jobs[idx][1]
                    results[idx] = DockResult(jobs[idx][0], sm, None, f"error:{exc}")
                done += 1
                if done % 25 == 0 or done == len(jobs):
                    logger.info("  %s: %d / %d", target, done, len(jobs))

    out_df = pd.DataFrame({
        "target":     [target] * len(results),
        "chain_idx":  [r.chain_idx for r in results],
        "smiles":     [r.smiles for r in results],
        "vina_score": [r.vina_score for r in results],
        "status":     [r.status for r in results],
    })
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_pq)
    return out_df


def summarize(df: pd.DataFrame) -> dict:
    ok = df[df["status"] == "ok"]
    n_total = len(df)
    n_ok = len(ok)
    if n_ok == 0:
        return {"n_total": n_total, "n_ok": 0, "mean": None, "median": None,
                "top10_mean": None, "p90": None}
    s = ok["vina_score"].to_numpy()
    # top10 = best (most negative) 10 docking scores
    top10 = np.sort(s)[:10]
    return {
        "n_total": n_total,
        "n_ok": n_ok,
        "mean":       float(np.mean(s)),
        "median":     float(np.median(s)),
        "top10_mean": float(np.mean(top10)),
        "p90":        float(np.percentile(s, 90)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoded-dir", type=Path,
                   default=Path("results/eval/phase4/decoded"))
    p.add_argument("--receptors", type=Path,
                   default=Path("data/external/receptors"))
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/eval/phase4/vina"))
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--targets", nargs="*",
                   help="Subset of target names (default: all in decoded-dir)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap ligands per target for smoke tests.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pqs = sorted(args.decoded_dir.glob("*.parquet"))
    if args.targets:
        keep = set(args.targets)
        pqs = [q for q in pqs if q.stem in keep]
    if not pqs:
        raise SystemExit(f"no parquets in {args.decoded_dir}")

    summary_rows = []
    for pq in pqs:
        target = pq.stem
        rec_dir = args.receptors / target
        if not (rec_dir / "receptor.pdbqt").exists():
            logger.error("missing receptor for %s — skip", target)
            continue
        out_pq = args.out_dir / f"{target}.parquet"
        df = dock_target(target, pq, rec_dir, out_pq,
                         workers=args.workers,
                         exhaustiveness=args.exhaustiveness,
                         limit=args.limit)
        s = summarize(df)
        s["target"] = target
        summary_rows.append(s)
        logger.info(
            "%s: n_ok=%d/%d mean=%s top10=%s",
            target, s["n_ok"], s["n_total"],
            f"{s['mean']:.2f}" if s["mean"] is not None else "NA",
            f"{s['top10_mean']:.2f}" if s["top10_mean"] is not None else "NA",
        )

    s_df = pd.DataFrame(summary_rows).set_index("target").sort_index()
    s_df.to_csv(args.out_dir / "summary.csv")
    logger.info("summary → %s", args.out_dir / "summary.csv")
    print(s_df)


if __name__ == "__main__":
    main()
