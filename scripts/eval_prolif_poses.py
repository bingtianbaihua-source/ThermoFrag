#!/usr/bin/env python
"""ProLIF interaction-fingerprint evaluation of generator poses vs cognate
co-crystal ligands (Phase-7 Task 5).

Spec: ``docs/validation/05_pose_validation.md``.

Per (generator, target, chain_idx in the top-K Vina pool):

  * Compute the cognate fingerprint once per target.
  * Compute the pose fingerprint and Tanimoto-compare to the cognate's
    bit vector (union of {residue_id, interaction_type} columns).
  * Record number of shared / unique / missed contacts and the list of
    shared residues.

Outputs land under ``results/eval/phase7/prolif/``. The aggregator entry
goes to ``AGGREGATE/05_prolif_summary.json``. See task-5 doc for the
acceptance thresholds.

Run env: ``py311`` (prolif 2.0.3 + MDAnalysis 2.9 + rdkit 2025).

The receptor PDBs in ``data/external/receptors/<target>/receptor_clean.pdb``
have no hydrogens; we cache a protonated copy to ``receptor_for_prolif.pdb``
on first use via ``reduce`` (from the ``tf-eval`` env).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REDUCE_BIN = Path("/home/zhao/miniconda3/envs/tf-eval/bin/reduce")
OBABEL_BIN = Path("/home/zhao/miniconda3/envs/py311/bin/obabel")
TARGETS_15 = ["ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1", "GBA",
              "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG",
              "TP53", "VDR"]
GENERATORS_4 = ["thermofrag", "targetdiff", "rxnflow", "bbar"]


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _force_uniform_pdb_info(mol, resname: str, resnumber: int, chain: str) -> None:
    """Patch every atom's AtomPDBResidueInfo to share the same residue.

    AddHs leaves freshly-added hydrogens without PDB info, which makes
    ProLIF's residue grouping see mixed (chain=str, chain=None) tuples
    and crash on sort. We rewrite every atom uniformly so the ligand is
    one residue.
    """
    from rdkit import Chem
    for i, atom in enumerate(mol.GetAtoms()):
        info = Chem.AtomPDBResidueInfo()
        info.SetResidueName(resname.ljust(3)[:3])
        info.SetResidueNumber(int(resnumber))
        info.SetChainId(chain)
        info.SetName(f"{atom.GetSymbol():>2}{i:>2d}"[:4])
        info.SetIsHeteroAtom(True)
        atom.SetMonomerInfo(info)


def protonate_receptor(receptor_pdb: Path, out_path: Path) -> Path:
    """Add hydrogens to a receptor PDB using ``obabel -h``.

    Idempotent: skips if ``out_path`` already exists. We prefer obabel
    over ``reduce`` because reduce occasionally writes Hs that confuse
    MDAnalysis's RDKit converter (e.g., on ADRB2 LYS sidechains, the
    ``_standardize_patterns`` step raises an explicit-valence error).
    """
    if out_path.exists():
        return out_path
    if not OBABEL_BIN.exists():
        raise RuntimeError(f"obabel not found at {OBABEL_BIN}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [str(OBABEL_BIN), str(receptor_pdb), "-O", str(out_path), "-h"],
        capture_output=True, text=True
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"obabel produced empty output for {receptor_pdb}\n"
            f"stderr: {proc.stderr[:500]}"
        )
    return out_path


def cognate_resname_for(target: str, repo_root: Path) -> str:
    box = json.loads((repo_root / "data" / "external" / "receptors" / target / "box.json").read_text())
    return box["cognate_resname"]


def _trim_protein(receptor_h: Path, cognate_pdb: Path,
                  out_path: Path, cutoff: float = 8.0) -> Path:
    """Write a binding-site-only PDB containing complete residues with any
    atom within ``cutoff`` Å of any cognate atom. The full-receptor PDB
    is too slow for ProLIF's MDA→RDKit converter on big complexes
    (e.g. ALDH1 tetramer, 30k atoms). The trimmed version only contains
    residues that can possibly contact any ligand sitting in the binding
    box, so the fingerprint is identical for the contacts we care about.

    Idempotent.
    """
    if out_path.exists():
        return out_path
    import MDAnalysis as mda
    import numpy as np
    u = mda.Universe(str(receptor_h))
    cog = mda.Universe(str(cognate_pdb))
    prot_atoms = u.select_atoms("protein")
    if len(prot_atoms) == 0:
        out_path.write_text(receptor_h.read_text())  # nothing to trim
        return out_path
    cog_atoms = cog.atoms
    # MDA's `around` needs both groups in the same Universe; ours come
    # from different files. Compute distances manually with numpy and
    # select residues by index.
    pp = prot_atoms.positions  # [N_prot, 3]
    cp = cog_atoms.positions   # [N_cog, 3]
    # Per-prot-atom min distance to any cog atom (use chunking for large N)
    chunk = 4096
    near_mask = np.zeros(len(pp), dtype=bool)
    cutoff_sq = cutoff * cutoff
    for s in range(0, len(pp), chunk):
        e = min(s + chunk, len(pp))
        diff = pp[s:e, None, :] - cp[None, :, :]
        d2 = (diff * diff).sum(axis=-1)
        near_mask[s:e] = d2.min(axis=1) <= cutoff_sq
    # Promote to whole-residue: if any atom of a residue is near, keep all
    # atoms of that residue.
    near_resids = set()
    for atom_idx, atom in enumerate(prot_atoms):
        if near_mask[atom_idx]:
            near_resids.add((atom.segid, atom.resid))
    keep_atom_idx = [i for i, atom in enumerate(prot_atoms)
                     if (atom.segid, atom.resid) in near_resids]
    if not keep_atom_idx:
        # Fallback: keep everything (let caller deal with size)
        nearby = prot_atoms
    else:
        nearby = prot_atoms[keep_atom_idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nearby.write(str(out_path))
    return out_path


def compute_cognate_fingerprint(target: str, repo_root: Path, out_root: Path) -> tuple[pd.DataFrame, Path]:
    """Compute and cache the cognate fingerprint for one target."""
    import MDAnalysis as mda
    import prolif as plf
    from rdkit import Chem

    cog_parquet = out_root / f"{target}_cognate.parquet"
    if cog_parquet.exists():
        return pd.read_parquet(cog_parquet), cog_parquet

    rec_dir = repo_root / "data" / "external" / "receptors" / target
    receptor_clean = rec_dir / "receptor_clean.pdb"
    cognate_pdb = rec_dir / "cognate_ligand.pdb"
    receptor_h = rec_dir / "receptor_for_prolif.pdb"
    if not receptor_h.exists():
        print(f"[protonate] {target}: writing {receptor_h.name}", flush=True)
        protonate_receptor(receptor_clean, receptor_h)
    receptor_trim = rec_dir / "receptor_for_prolif_trim.pdb"
    if not receptor_trim.exists():
        print(f"[trim] {target}: writing {receptor_trim.name}", flush=True)
        _trim_protein(receptor_h, cognate_pdb, receptor_trim, cutoff=8.0)

    u = mda.Universe(str(receptor_trim))
    prot = plf.Molecule.from_mda(u.select_atoms("protein"))

    cog_mol = Chem.MolFromPDBFile(str(cognate_pdb), removeHs=False, sanitize=False)
    if cog_mol is None:
        raise RuntimeError(f"RDKit failed to read cognate {cognate_pdb}")
    try:
        Chem.SanitizeMol(cog_mol)
    except Exception:
        # Some cognate PDBs have aromatic-perception edge cases; try with
        # partial sanitization.
        Chem.SanitizeMol(cog_mol,
                         sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                     Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    cog_mol = Chem.AddHs(cog_mol, addCoords=True)
    _force_uniform_pdb_info(cog_mol, resname=cognate_resname_for(target, repo_root),
                            resnumber=1, chain="X")
    cog = plf.Molecule.from_rdkit(cog_mol, resname=cognate_resname_for(target, repo_root),
                                  resnumber=1, chain="X")

    fp = plf.Fingerprint()
    fp.run_from_iterable([cog], prot)
    df = fp.to_dataframe()
    if df.empty or len(df) == 0:
        raise RuntimeError(f"empty cognate fingerprint for {target}")

    # Save flat representation: contact_id (residue, type), residue_id,
    # residue_name, interaction_type
    rows = []
    for col in df.columns:
        # ProLIF v2 columns are MultiIndex of (ligand_residue, protein_residue, interaction)
        if isinstance(col, tuple) and len(col) == 3:
            lig_res, prot_res, interaction = col
        elif isinstance(col, tuple) and len(col) == 2:
            prot_res, interaction = col
            lig_res = "LIG"
        else:
            prot_res, interaction = col, "?"; lig_res = "LIG"
        present = bool(df.iloc[0][col])
        if not present:
            continue
        residue_str = str(prot_res)
        # Best-effort resid extraction: ProLIF residue id is "RESN.RESID.CHAIN"
        if "." in residue_str:
            parts = residue_str.split(".")
            resname = parts[0]
            try:
                resid = int(parts[1])
            except (ValueError, IndexError):
                resid = -1
        else:
            resname, resid = residue_str, -1
        rows.append({
            "ligand_residue": str(lig_res),
            "residue_id": residue_str,
            "residue_name": resname,
            "residue_num": resid,
            "interaction_type": interaction,
        })
    flat = pd.DataFrame(rows)
    flat.to_parquet(cog_parquet, index=False)
    print(f"[cognate] {target}: {len(flat)} contacts -> {cog_parquet}")
    return flat, cog_parquet


def _build_protein(receptor_h: Path):
    import MDAnalysis as mda
    import prolif as plf
    u = mda.Universe(str(receptor_h))
    return plf.Molecule.from_mda(u.select_atoms("protein"))


def _ifp_to_set(df: pd.DataFrame) -> set[tuple]:
    """Flatten one fingerprint dataframe row into a set of contact tuples
    (protein_residue, interaction). Ligand residue is dropped because we are
    comparing across different ligands."""
    out = set()
    if len(df) == 0:
        return out
    for col in df.columns:
        if isinstance(col, tuple) and len(col) == 3:
            _, prot_res, interaction = col
        elif isinstance(col, tuple) and len(col) == 2:
            prot_res, interaction = col
        else:
            continue
        if bool(df.iloc[0][col]):
            out.add((str(prot_res), str(interaction)))
    return out


def evaluate_pose(pose_sdf: Path, prot, cog_set: set[tuple]) -> dict:
    """Compute Tanimoto vs cognate set + share / unique / missed counts."""
    import prolif as plf
    from rdkit import Chem

    suppl = Chem.SDMolSupplier(str(pose_sdf), removeHs=False, sanitize=True)
    pose_mol = suppl[0] if len(suppl) else None
    if pose_mol is None:
        return {"status": "rdkit_failed"}
    # SDF poses don't carry PDB residue info; assign uniform info so
    # ProLIF treats the ligand as a single residue (matches cognate setup).
    _force_uniform_pdb_info(pose_mol, resname="LIG", resnumber=1, chain="X")
    pose = plf.Molecule.from_rdkit(pose_mol, resname="LIG", resnumber=1, chain="X")
    fp = plf.Fingerprint()
    fp.run_from_iterable([pose], prot)
    pose_df = fp.to_dataframe()
    pose_set = _ifp_to_set(pose_df)

    inter = cog_set & pose_set
    only_pose = pose_set - cog_set
    only_cog = cog_set - pose_set
    union = cog_set | pose_set
    tanimoto = len(inter) / len(union) if union else 0.0

    shared_residues = sorted({r for (r, _) in inter})
    return {
        "status": "ok",
        "prolif_tanimoto_vs_cognate": float(tanimoto),
        "prolif_n_shared_contacts": int(len(inter)),
        "prolif_n_unique_to_pose": int(len(only_pose)),
        "prolif_n_missed_from_cognate": int(len(only_cog)),
        "shared_residues": shared_residues,
    }


def evaluate_generator_target(generator: str, target: str, top_k: int,
                              repo_root: Path, out_root: Path,
                              poses_root: Path) -> Path:
    """Process top-K poses for one (generator, target) and write parquet."""
    out_dir = out_root / generator
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"{target}.parquet"

    # Load cognate flat table to get the contact set for comparisons
    cog_flat, _ = compute_cognate_fingerprint(target, repo_root, out_root)
    cog_set = {(row.residue_id, row.interaction_type)
               for row in cog_flat.itertuples(index=False)}
    if not cog_set:
        print(f"[skip] {target}: cognate fingerprint is empty")
        return out_parquet

    # Locate poses
    pose_dir = poses_root / generator / target
    manifest_path = pose_dir / "manifest.parquet"
    if not manifest_path.exists():
        print(f"[skip] {generator}/{target}: no manifest at {manifest_path}")
        return out_parquet
    manifest = pd.read_parquet(manifest_path)
    manifest_ok = manifest[manifest["status"] == "ok"].sort_values("vina_score").head(top_k)

    rec_dir = repo_root / "data" / "external" / "receptors" / target
    receptor_h = rec_dir / "receptor_for_prolif.pdb"
    receptor_trim = rec_dir / "receptor_for_prolif_trim.pdb"
    if not receptor_h.exists():
        protonate_receptor(rec_dir / "receptor_clean.pdb", receptor_h)
    if not receptor_trim.exists():
        _trim_protein(receptor_h, rec_dir / "cognate_ligand.pdb",
                      receptor_trim, cutoff=8.0)
    prot = _build_protein(receptor_trim)

    rows = []
    for _, m in manifest_ok.iterrows():
        chain_idx = int(m["chain_idx"])
        pose_sdf = pose_dir / f"{chain_idx}.sdf"
        if not pose_sdf.exists():
            rows.append({"chain_idx": chain_idx, "smiles": m["smiles"],
                         "vina_score": float(m["vina_score"]),
                         "status": "no_pose"})
            continue
        try:
            res = evaluate_pose(pose_sdf, prot, cog_set)
        except Exception as e:
            res = {"status": f"eval_failed: {type(e).__name__}: {str(e)[:120]}"}
        rows.append({"chain_idx": chain_idx, "smiles": m["smiles"],
                     "vina_score": float(m["vina_score"]), **res})

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"[done] {generator}/{target}: {n_ok}/{len(df)} ok -> {out_parquet}")
    return out_parquet


def aggregate(args, out_root: Path) -> dict:
    """Build the AGGREGATE summary JSON over all written parquets."""
    rows_all = []
    for gen in GENERATORS_4:
        for tgt in TARGETS_15:
            p = out_root / gen / f"{tgt}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["generator"] = gen
            df["target"] = tgt
            rows_all.append(df)
    if not rows_all:
        return {"task_id": "05_prolif", "n_files": 0,
                "notes": "no per-(gen,target) parquets found"}
    full = pd.concat(rows_all, ignore_index=True)

    per_target_per_gen = {}
    for tgt in sorted(full["target"].unique()):
        per_gen = {}
        for gen in sorted(full["generator"].unique()):
            sub = full[(full["target"] == tgt) & (full["generator"] == gen) &
                       (full["status"] == "ok")]
            if len(sub) == 0:
                per_gen[gen] = {"n_top_k": 0, "mean_tanimoto": None,
                                "max_tanimoto": None, "any_canonical_hit": False}
                continue
            mean_t = float(sub["prolif_tanimoto_vs_cognate"].mean())
            max_t = float(sub["prolif_tanimoto_vs_cognate"].max())
            # "Canonical hit" = at least one shared contact with cognate
            any_canonical = bool((sub["prolif_n_shared_contacts"] > 0).any())
            per_gen[gen] = {"n_top_k": int(len(sub)),
                            "mean_tanimoto": mean_t,
                            "max_tanimoto": max_t,
                            "any_canonical_hit": any_canonical}
        per_target_per_gen[tgt] = per_gen

    # Threshold A: TF top-K mean Tanimoto >= 0.4 on >= 10/15 targets
    tf_pass = sum(1 for tgt, d in per_target_per_gen.items()
                  if d.get("thermofrag", {}).get("mean_tanimoto") is not None
                  and d["thermofrag"]["mean_tanimoto"] >= 0.4)
    # Threshold B: TF mean >= TargetDiff mean on >= 8/15 targets
    tf_vs_td = 0
    for tgt, d in per_target_per_gen.items():
        tf = d.get("thermofrag", {}).get("mean_tanimoto")
        td = d.get("targetdiff", {}).get("mean_tanimoto")
        if tf is None or td is None:
            continue
        if tf >= td:
            tf_vs_td += 1
    # Threshold C: TF has at least one canonical hit on every target
    tf_canonical_hits = sum(1 for tgt, d in per_target_per_gen.items()
                            if d.get("thermofrag", {}).get("any_canonical_hit"))

    summary = {
        "task_id": "05_prolif",
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "thresholds": {
            "tf_mean_tanimoto_geq_0.4": {
                "target": ">= 10/15", "observed": f"{tf_pass}/15",
                "pass": bool(tf_pass >= 10),
            },
            "tf_geq_targetdiff_mean": {
                "target": ">= 8/15", "observed": f"{tf_vs_td}/15",
                "pass": bool(tf_vs_td >= 8),
            },
            "tf_any_canonical_hit_per_target": {
                "target": "= 15/15", "observed": f"{tf_canonical_hits}/15",
                "pass": bool(tf_canonical_hits == 15),
            },
        },
        "per_target": per_target_per_gen,
        "n_rows": int(len(full)),
        "top_k": args.top_k,
    }

    # Print summary
    print()
    print("=" * 70)
    print("Task 5 (ProLIF) summary")
    print("=" * 70)
    print(f"  TF mean Tanimoto >= 0.4 on              {tf_pass}/15 targets   "
          f"(target >= 10/15) {'PASS' if tf_pass >= 10 else 'FAIL'}")
    print(f"  TF mean >= TargetDiff mean on           {tf_vs_td}/15 targets   "
          f"(target >= 8/15)  {'PASS' if tf_vs_td >= 8 else 'FAIL'}")
    print(f"  TF >=1 canonical hit per target on      {tf_canonical_hits}/15 targets  "
          f"(target = 15/15)  {'PASS' if tf_canonical_hits == 15 else 'FAIL'}")
    print()

    agg_dir = out_root.parent / "AGGREGATE"
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "05_prolif_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[write] {agg_dir / '05_prolif_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", default="all",
                   choices=GENERATORS_4 + ["all"])
    p.add_argument("--target", default="all",
                   choices=TARGETS_15 + ["all"])
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--out_root", type=Path,
                   default=Path("results/eval/phase7/prolif"))
    p.add_argument("--poses_root", type=Path,
                   default=Path("results/eval/phase7/poses"))
    p.add_argument("--repo_root", type=Path, default=Path("/home/zhao/code/ThermoFrag"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild_cognates", action="store_true")
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip per-pose evaluation; only rebuild AGGREGATE JSON.")
    p.add_argument("--subprocess_per_target", action="store_true",
                   help="Spawn a fresh Python subprocess per (generator,target). "
                        "Isolates native crashes (e.g. RDKit segfault on the "
                        "NAP cognate used by IDH1).")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.rebuild_cognates:
        for tgt in TARGETS_15:
            cog = out_root / f"{tgt}_cognate.parquet"
            if cog.exists():
                cog.unlink()

    if args.dry_run:
        print(f"[dry] generator={args.generator} target={args.target} "
              f"top_k={args.top_k} out_root={out_root}")
        return

    if args.aggregate_only:
        aggregate(args, out_root)
        return

    gens = GENERATORS_4 if args.generator == "all" else [args.generator]
    tgts = TARGETS_15 if args.target == "all" else [args.target]

    t0 = time.time()
    if args.subprocess_per_target:
        # Run each (gen, target) in a fresh Python subprocess. Isolates
        # native-crash failures (e.g. RDKit segfault on the NAP cognate
        # used by IDH1) so they don't kill the whole sweep.
        for gen in gens:
            for tgt in tgts:
                cmd = [
                    sys.executable, "-u", __file__,
                    "--generator", gen, "--target", tgt,
                    "--top_k", str(args.top_k),
                    "--out_root", str(out_root),
                    "--poses_root", str(args.poses_root),
                    "--repo_root", str(args.repo_root),
                    "--seed", str(args.seed),
                ]
                proc = subprocess.run(cmd, capture_output=False)
                if proc.returncode != 0:
                    print(f"[FAIL] {gen}/{tgt}: subprocess exited "
                          f"with code {proc.returncode}", flush=True)
    else:
        for gen in gens:
            for tgt in tgts:
                evaluate_generator_target(gen, tgt, args.top_k, args.repo_root,
                                          out_root, args.poses_root)
    wall = time.time() - t0

    summary = aggregate(args, out_root)

    manifest = {
        "git_sha": _git_sha(args.repo_root),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.executable,
        "wall_seconds": wall,
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    out_root.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
