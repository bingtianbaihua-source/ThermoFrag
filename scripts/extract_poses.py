"""Phase-7 task 0 — extract docked Vina poses for downstream rescoring.

The original Phase-4 docking driver (``scripts/dock_vina.py``) records
``vina_score`` only; it does not persist the pose. Tasks 1 (MM-GBSA),
2 (multi-scoring), 4 (MD), and 5 (ProLIF) all need the ligand 3D
coordinates inside the binding box.

For each ``(generator, target)`` pair this script:

  1. Reads the matching Vina parquet, filters ``status == 'ok'``,
     sorts ascending by ``vina_score``, takes the top ``--top_k`` rows.
  2. Re-builds the 3D ligand from SMILES (RDKit ETKDG + MMFF94, seed=42).
  3. Prepares the PDBQT via Meeko ``mk_prepare_ligand.py``.
  4. Re-docks with Vina against the same receptor + box, persisting the
     output PDBQT.
  5. Extracts MODEL 1 from the multi-mode docked PDBQT and converts it
     to PDB via Meeko ``mk_export.py``.

Output layout::

    <out_root>/<gen>/<target>/<chain_idx>.pdb
    <out_root>/<gen>/<target>/<chain_idx>.sdf
    <out_root>/<gen>/<target>/manifest.parquet
        chain_idx, smiles, vina_score, vina_pose_score, status
    <out_root>/<gen>/<target>/manifest.json
        git SHA, args, env, runtime, input SHAs

Idempotent: if the per-ligand PDB exists, the row is skipped on re-run.

Conventions: see ``docs/validation/00_shared_infrastructure.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("extract_poses")

_SCORE_RE = re.compile(r"^\s*1\s+(-?\d+(?:\.\d+)?)", re.MULTILINE)

GENERATOR_VINA_DIR = {
    "thermofrag": Path("results/eval/phase4/vina"),
    "targetdiff": Path("results/eval/phase4_baselines/targetdiff/vina"),
    "rxnflow":    Path("results/eval/phase4_baselines/rxnflow/vina"),
    "bbar":       Path("results/eval/phase4_baselines/bbar/vina"),
}

ALL_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]

TF_EVAL_BIN = "/home/zhao/miniconda3/envs/tf-eval/bin"


@dataclass
class PoseResult:
    chain_idx: int
    smiles: str
    vina_score: float
    vina_pose_score: Optional[float]
    status: str  # ok | skipped | embed_failed | prep_failed | vina_timeout |
                 # vina_failed | split_failed | export_failed


def _env_with_path() -> dict:
    """Subprocess env with tf-eval/bin prepended to PATH (env-bin gotcha)."""
    env = os.environ.copy()
    env["PATH"] = f"{TF_EVAL_BIN}:{env.get('PATH', '')}"
    return env


def _embed_and_write_sdf(smiles: str, sdf_path: Path, seed: int) -> bool:
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


def _prep_ligand_pdbqt(sdf_path: Path, pdbqt_path: Path, env: dict) -> bool:
    cmd = ["mk_prepare_ligand.py", "-i", str(sdf_path), "-o", str(pdbqt_path),
           "--charge_model", "gasteiger"]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=120, env=env)
    return res.returncode == 0 and pdbqt_path.exists() and pdbqt_path.stat().st_size > 100


def _run_vina(lig_pdbqt: Path, rec_pdbqt: Path, box: dict, out_pdbqt: Path,
              exhaustiveness: int, num_modes: int, cpu: int,
              seed: int, env: dict) -> Optional[float]:
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
        "--seed", str(seed),
        "--out", str(out_pdbqt),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=600, env=env)
    if res.returncode != 0:
        return None
    m = _SCORE_RE.search(res.stdout)
    if not m:
        return None
    return float(m.group(1))


def _split_first_model(multi_pdbqt: Path, single_pdbqt: Path) -> bool:
    """Extract the first MODEL block from a Vina multi-pose PDBQT."""
    try:
        with multi_pdbqt.open() as fh:
            text = fh.read()
    except OSError:
        return False
    # First MODEL ... ENDMDL block.
    m = re.search(r"^MODEL\s+1\s*$.*?^ENDMDL\s*$", text,
                  flags=re.DOTALL | re.MULTILINE)
    if not m:
        # Some Vina builds emit a single-pose file without MODEL records.
        if "ATOM" in text or "HETATM" in text:
            single_pdbqt.write_text(text)
            return True
        return False
    block = m.group(0)
    single_pdbqt.write_text(block + "\n")
    return True


def _export_pdb(single_pdbqt: Path, out_pdb: Path, out_sdf: Path,
                env: dict) -> bool:
    """Convert a single-MODEL Vina PDBQT to PDB + SDF using Meeko + RDKit.

    Runs in-process; ``env`` is unused but kept in the signature for symmetry
    with the other subprocess helpers. ``mk_export.py`` requires a receptor
    JSON sidecar (from ``mk_prepare_receptor``) for ``-p``, which we don't
    produce, so we go through Meeko's Python API instead.
    """
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
        from rdkit import Chem
    except ImportError:
        return False
    try:
        pmol = PDBQTMolecule.from_file(str(single_pdbqt),
                                       is_dlg=False, skip_typing=True)
        rdmols = RDKitMolCreate.from_pdbqt_mol(pmol)
    except Exception:
        return False
    if not rdmols:
        return False
    mol = rdmols[0]
    if mol is None or mol.GetNumConformers() == 0:
        return False
    try:
        Chem.MolToPDBFile(mol, str(out_pdb))
        w = Chem.SDWriter(str(out_sdf))
        w.write(mol)
        w.close()
    except Exception:
        return False
    return out_pdb.exists() and out_pdb.stat().st_size > 100


def _extract_one(args):
    (chain_idx, smiles, vina_score_orig, rec_pdbqt_str, box,
     out_pdb_str, out_sdf_str, exhaustiveness, num_modes, seed) = args
    out_pdb = Path(out_pdb_str)
    out_sdf = Path(out_sdf_str)
    env = _env_with_path()

    if out_pdb.exists() and out_pdb.stat().st_size > 100:
        return PoseResult(chain_idx, smiles, vina_score_orig, None, "skipped")

    rec_pdbqt = Path(rec_pdbqt_str)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="extract_pose_") as tmp:
        tmp = Path(tmp)
        sdf = tmp / "lig.sdf"
        pdbqt = tmp / "lig.pdbqt"
        multi = tmp / "dock_all.pdbqt"
        first = tmp / "dock_top.pdbqt"

        if not _embed_and_write_sdf(smiles, sdf, seed=seed):
            return PoseResult(chain_idx, smiles, vina_score_orig, None,
                              "embed_failed")
        if not _prep_ligand_pdbqt(sdf, pdbqt, env):
            return PoseResult(chain_idx, smiles, vina_score_orig, None,
                              "prep_failed")
        try:
            score = _run_vina(pdbqt, rec_pdbqt, box, multi,
                              exhaustiveness=exhaustiveness,
                              num_modes=num_modes, cpu=1, seed=seed, env=env)
        except subprocess.TimeoutExpired:
            return PoseResult(chain_idx, smiles, vina_score_orig, None,
                              "vina_timeout")
        if score is None:
            return PoseResult(chain_idx, smiles, vina_score_orig, None,
                              "vina_failed")

        if not _split_first_model(multi, first):
            return PoseResult(chain_idx, smiles, vina_score_orig, score,
                              "split_failed")
        if not _export_pdb(first, out_pdb, out_sdf, env):
            return PoseResult(chain_idx, smiles, vina_score_orig, score,
                              "export_failed")

    return PoseResult(chain_idx, smiles, vina_score_orig, float(score), "ok")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha(repo: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def extract_target(generator: str, target: str, vina_pq: Path,
                   rec_dir: Path, out_dir: Path, top_k: int, workers: int,
                   exhaustiveness: int, num_modes: int, seed: int,
                   dry_run: bool, repo_root: Path) -> Optional[pd.DataFrame]:
    if not vina_pq.exists():
        logger.error("missing vina parquet: %s", vina_pq)
        return None
    box_path = rec_dir / "box.json"
    rec_pdbqt = rec_dir / "receptor.pdbqt"
    if not (box_path.exists() and rec_pdbqt.exists()):
        logger.error("missing receptor files: %s", rec_dir)
        return None

    df = pd.read_parquet(vina_pq)
    ok = df[df["status"] == "ok"].copy()
    ok = ok.sort_values("vina_score", ascending=True)
    ok = ok.head(top_k).reset_index(drop=True)

    if len(ok) == 0:
        logger.warning("%s/%s: no ok rows in vina parquet", generator, target)
        return None

    box = json.loads(box_path.read_text())

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prior manifest so an idempotent re-run preserves the ok status
    # and vina_pose_score from the original extraction. Without this the
    # second run rewrites every cached row as 'skipped'.
    manifest_pq = out_dir / "manifest.parquet"
    prior: dict[int, dict] = {}
    if manifest_pq.exists():
        try:
            pdf = pd.read_parquet(manifest_pq)
            for _, r in pdf.iterrows():
                prior[int(r["chain_idx"])] = r.to_dict()
        except Exception:
            prior = {}

    jobs = []
    skipped_rows: list[PoseResult] = []
    for _, row in ok.iterrows():
        ci = int(row["chain_idx"])
        out_pdb = out_dir / f"{ci}.pdb"
        out_sdf = out_dir / f"{ci}.sdf"
        if out_pdb.exists() and out_pdb.stat().st_size > 100:
            # PDB on disk is the invariant. Replay the prior row's
            # ``vina_pose_score`` if we have it; otherwise leave it None.
            p = prior.get(ci, {})
            pose_score = p.get("vina_pose_score")
            try:
                pose_score = (float(pose_score)
                              if pose_score is not None and pd.notna(pose_score)
                              else None)
            except (TypeError, ValueError):
                pose_score = None
            skipped_rows.append(PoseResult(
                chain_idx=ci,
                smiles=str(row["smiles"]),
                vina_score=float(row["vina_score"]),
                vina_pose_score=pose_score,
                status="ok",
            ))
            continue
        jobs.append((
            ci, str(row["smiles"]), float(row["vina_score"]),
            str(rec_pdbqt), box, str(out_pdb), str(out_sdf),
            exhaustiveness, num_modes, seed,
        ))

    if dry_run:
        logger.info("[dry] %s/%s: would extract %d poses (top_k=%d, "
                    "%d cached) → %s",
                    generator, target, len(jobs), top_k,
                    len(skipped_rows), out_dir)
        return None

    logger.info("%s/%s: %d ligands to run (%d cached), %d workers",
                generator, target, len(jobs), len(skipped_rows), workers)

    results: list[PoseResult] = [None] * len(jobs)
    if workers <= 1:
        for i, j in enumerate(jobs):
            results[i] = _extract_one(j)
            if (i + 1) % 10 == 0:
                logger.info("  %s/%s: %d / %d", generator, target,
                            i + 1, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_extract_one, j): idx for idx, j in enumerate(jobs)}
            done = 0
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    j = jobs[idx]
                    results[idx] = PoseResult(j[0], j[1], j[2], None,
                                              f"error:{type(exc).__name__}")
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    logger.info("  %s/%s: %d / %d", generator, target,
                                done, len(jobs))

    rows = [asdict(r) for r in results + skipped_rows]
    out_df = pd.DataFrame(rows)
    out_df.insert(0, "target", target)
    out_df.insert(0, "generator", generator)
    out_df = out_df.sort_values("vina_score").reset_index(drop=True)
    out_df.to_parquet(manifest_pq)

    n_ok = int((out_df["status"] == "ok").sum())
    n_cache = len(skipped_rows)
    n_fail = len(out_df) - n_ok
    logger.info("%s/%s: ok=%d (of which %d cached) failed=%d → %s",
                generator, target, n_ok, n_cache, n_fail, manifest_pq)
    return out_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", required=True,
                   choices=["thermofrag", "targetdiff", "rxnflow", "bbar", "all"])
    p.add_argument("--target", default="all",
                   help="One of the 15 LIT-PCBA targets, or 'all'")
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--out_root", type=Path,
                   default=Path("results/eval/phase7/poses"))
    p.add_argument("--receptors", type=Path,
                   default=Path("data/external/receptors"))
    p.add_argument("--n_workers", type=int,
                   default=max(1, (os.cpu_count() or 4) - 1))
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num_modes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--repo_root", type=Path,
                   default=Path("/home/zhao/code/ThermoFrag"))
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    generators = (["thermofrag", "targetdiff", "rxnflow", "bbar"]
                  if args.generator == "all" else [args.generator])
    targets = ALL_TARGETS if args.target == "all" else [args.target]

    if args.target != "all" and args.target not in ALL_TARGETS:
        raise SystemExit(f"unknown target {args.target}; "
                         f"expected one of {ALL_TARGETS}")

    repo_root = args.repo_root.resolve()
    git_sha = _git_sha(repo_root)
    t0 = time.time()

    for gen in generators:
        vina_dir = repo_root / GENERATOR_VINA_DIR[gen]
        for target in targets:
            vina_pq = vina_dir / f"{target}.parquet"
            rec_dir = (args.receptors if args.receptors.is_absolute()
                       else repo_root / args.receptors) / target
            out_dir = (args.out_root if args.out_root.is_absolute()
                       else repo_root / args.out_root) / gen / target
            extract_target(
                generator=gen, target=target, vina_pq=vina_pq,
                rec_dir=rec_dir, out_dir=out_dir,
                top_k=args.top_k, workers=args.n_workers,
                exhaustiveness=args.exhaustiveness, num_modes=args.num_modes,
                seed=args.seed, dry_run=args.dry_run, repo_root=repo_root,
            )

    if args.dry_run:
        logger.info("[dry] done")
        return

    # Top-level run manifest (one per CLI invocation).
    out_root_abs = (args.out_root if args.out_root.is_absolute()
                    else repo_root / args.out_root)
    out_root_abs.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "git_sha": git_sha,
        "args": vars(args),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.executable,
        "wall_seconds": time.time() - t0,
    }
    # JSON-friendly Path serialization.
    run_manifest["args"] = {k: (str(v) if isinstance(v, Path) else v)
                            for k, v in run_manifest["args"].items()}
    ts = time.strftime("%Y%m%dT%H%M%S")
    (out_root_abs / f"run_manifest_{ts}.json").write_text(
        json.dumps(run_manifest, indent=2))
    logger.info("done in %.1f s", run_manifest["wall_seconds"])


if __name__ == "__main__":
    main()
