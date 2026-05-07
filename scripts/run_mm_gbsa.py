"""Phase-7 task 1 — MM-GBSA single-frame rescoring of docked Vina poses.

For each (generator, target) pair this script consumes the top-K
docked poses written by ``scripts/extract_poses.py`` (task 0) and
rescores them with single-frame MM-GBSA (Amber GB^OBC2, igb=5,
saltcon=0.150 M).

Pipeline per ligand
-------------------

  1. ``antechamber -c bcc -at gaff2`` on the docked SDF (explicit Hs +
     bond orders preserved). On AM1-BCC failure, fall back to ``-c gas``
     (Gasteiger) and record ``charge_method='gas'``.
  2. ``parmchk2 -s gaff2`` produces the ligand frcmod.
  3. ``tleap`` builds three prmtops: complex, receptor-only, ligand-only.
  4. ``cpptraj`` converts the complex inpcrd to a 1-frame DCD.
  5. ``MMPBSA.py -O -i mmpbsa.in -cp ... -rp ... -lp ... -y complex.dcd``.
  6. Parse ``RESULTS.dat`` for ``DELTA TOTAL``, ``VDWAALS``, ``EEL``,
     ``EGB``, ``ESURF``.

Receptor prep is done **once per target** (pdb4amber + tleap) before
worker fan-out. The per-target receptor cache lives at
``/tmp/thermofrag_validation/01_mm_gbsa/<target>/receptor/``.

Outputs
-------

::

    results/eval/phase7/mm_gbsa/<gen>/<target>/<chain_idx>.json
    results/eval/phase7/mm_gbsa/<gen>/<target>.parquet
    results/eval/phase7/mm_gbsa/<gen>/<target>/manifest.json
    results/eval/phase7/AGGREGATE/01_mm_gbsa_summary.json   (this script)

Status column values
--------------------

* ``ok`` — full pipeline succeeded; energies recorded.
* ``ante_failed`` — AM1-BCC and Gasteiger both failed.
* ``parmchk_failed`` — ``parmchk2`` did not produce a frcmod.
* ``tleap_failed`` — ``tleap`` could not build the complex prmtop.
* ``mmpbsa_failed`` — MMPBSA.py crashed or RESULTS.dat malformed.
* ``timeout`` — any subprocess exceeded its budget.

Conventions: see ``docs/validation/00_shared_infrastructure.md`` and
``docs/validation/01_mm_gbsa.md``.
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
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("run_mm_gbsa")

ALL_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]

ALL_GENERATORS = ["thermofrag", "targetdiff", "rxnflow", "bbar"]

TF_EVAL_ROOT = "/home/zhao/miniconda3/envs/tf-eval"
TF_EVAL_BIN = f"{TF_EVAL_ROOT}/bin"
TF_EVAL_PY = f"{TF_EVAL_BIN}/python"

POSES_ROOT = Path("results/eval/phase7/poses")
RECEPTOR_ROOT = Path("data/external/receptors")

# Pocket-proximity radius (Å): chains with no atom within this distance
# from the docking box center are dropped before tleap. This makes
# multimeric assemblies (ALDH1 = 8 chains × ~500 res = 61k atoms)
# tractable for sander minimization without changing the local
# binding-pocket energetics.
POCKET_CHAIN_RADIUS = 15.0

WORKDIR_ROOT = Path("/tmp/thermofrag_validation/01_mm_gbsa")

# Subprocess budgets (seconds).
ANTECHAMBER_BCC_TIMEOUT = 300
ANTECHAMBER_GAS_TIMEOUT = 60
PARMCHK_TIMEOUT = 60
TLEAP_TIMEOUT = 120
CPPTRAJ_TIMEOUT = 60
SANDER_TIMEOUT = 1800
MMPBSA_TIMEOUT = 600

# Pre-registered thresholds (mirror docs/validation/01_mm_gbsa.md).
THRESHOLDS = {
    "spearman_vs_vina": 0.5,
    "tf_vs_targetdiff_sigwins_min": 10,  # of 15
    "tf_vs_targetdiff_p_alpha": 0.05,
    "max_failure_rate": 0.30,
}


# --------------------------------------------------------------------- env --

def _amber_env() -> dict:
    env = os.environ.copy()
    env["AMBERHOME"] = TF_EVAL_ROOT
    env["PATH"] = f"{TF_EVAL_BIN}:{env.get('PATH', '')}"
    # OpenMP — keep workers from oversubscribing.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.setdefault("LANG", "C")
    return env


# ------------------------------------------------------------- per-target --

@dataclass
class ReceptorPrep:
    target: str
    workdir: Path
    receptor_amber_pdb: Path  # cleaned, hydrogenated
    receptor_prmtop: Path
    receptor_inpcrd: Path
    status: str = "ok"
    note: str = ""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _trim_to_pocket_chains(pdb_in: Path, pdb_out: Path,
                            box_center, radius: float) -> tuple[set, int]:
    """Keep only chains with at least one atom within `radius` of box center.

    Returns (kept_chains, n_atoms_after).
    """
    import numpy as np
    cx, cy, cz = box_center
    chain_atoms = {}  # chain -> list of (x,y,z) for proximity check
    chain_lines = {}  # chain -> list of raw lines (ATOM/TER/HETATM)
    raw_lines = pdb_in.read_text().splitlines()
    for line in raw_lines:
        if line.startswith(("ATOM  ", "HETATM", "TER")):
            chain = line[21] if len(line) > 21 else " "
            chain_lines.setdefault(chain, []).append(line)
            if line.startswith("ATOM  "):
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except ValueError:
                    continue
                chain_atoms.setdefault(chain, []).append((x, y, z))
    # Find chains with any atom within radius.
    kept = set()
    for chain, coords in chain_atoms.items():
        arr = np.asarray(coords)
        d = np.linalg.norm(arr - np.array([cx, cy, cz]), axis=1)
        if (d <= radius).any():
            kept.add(chain)
    if not kept:
        # Fallback: keep everything to avoid an empty file.
        kept = set(chain_lines.keys())
    out = []
    n_atoms = 0
    for line in raw_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            chain = line[21] if len(line) > 21 else " "
            if chain in kept:
                out.append(line)
                n_atoms += 1
        elif line.startswith("TER"):
            chain = line[21] if len(line) > 21 else " "
            if chain in kept:
                out.append(line)
        elif line.startswith(("END", "MODEL", "ENDMDL")):
            out.append(line)
    pdb_out.write_text("\n".join(out) + "\n")
    return kept, n_atoms


def prepare_receptor(target: str) -> ReceptorPrep:
    """Run pdb4amber + tleap once per target; cache on disk.

    Multimeric receptors are pre-trimmed to the chain(s) that contain
    the docking pocket — see `_trim_to_pocket_chains`.
    """
    workdir = WORKDIR_ROOT / target / "receptor"
    workdir.mkdir(parents=True, exist_ok=True)

    rec_in_full = (RECEPTOR_ROOT / target / "receptor_clean.pdb").resolve()
    if not rec_in_full.exists():
        return ReceptorPrep(target, workdir, Path(), Path(), Path(),
                            status="missing_input",
                            note=f"{rec_in_full} not found")

    box_path = (RECEPTOR_ROOT / target / "box.json").resolve()
    if not box_path.exists():
        return ReceptorPrep(target, workdir, Path(), Path(), Path(),
                            status="missing_box",
                            note=f"{box_path} not found")
    box = json.loads(box_path.read_text())
    box_center = tuple(box["center"])

    # Trim multimers to pocket-bearing chains. Cache the trimmed PDB.
    rec_in = workdir / "receptor_clean_pocket.pdb"
    if not rec_in.exists():
        kept, n_atoms = _trim_to_pocket_chains(rec_in_full, rec_in,
                                                box_center, POCKET_CHAIN_RADIUS)
        logger.info("%s: kept chains %s -> %d atoms (was %d)",
                    target, sorted(kept), n_atoms,
                    sum(1 for ln in rec_in_full.read_text().splitlines()
                        if ln.startswith("ATOM  ")))

    rec_amber = workdir / "receptor_amber.pdb"
    rec_prmtop = workdir / "receptor.prmtop"
    rec_inpcrd = workdir / "receptor.inpcrd"

    if rec_prmtop.exists() and rec_inpcrd.exists() and rec_amber.exists():
        return ReceptorPrep(target, workdir, rec_amber, rec_prmtop, rec_inpcrd)

    env = _amber_env()

    # pdb4amber: add Hs (reduce), drop waters, drop CONECT.
    # NOTE: --add-missing-atoms places atoms naively and produces severe
    # steric clashes (1e15 VDW) that destroy MMPBSA delta precision.
    # Let tleap fill missing heavy atoms from ff14SB residue templates.
    cmd = [
        "pdb4amber",
        "-i", str(rec_in),
        "-o", str(rec_amber),
        "--reduce", "--dry", "--no-conect",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=600, env=env, cwd=str(workdir))
    if res.returncode != 0 or not rec_amber.exists():
        return ReceptorPrep(target, workdir, Path(), Path(), Path(),
                            status="pdb4amber_failed",
                            note=res.stderr[-2000:])

    # tleap: build receptor-only prmtop.
    leap_in = workdir / "leap_receptor.in"
    leap_in.write_text(
        "source leaprc.protein.ff14SB\n"
        f"mol = loadpdb {rec_amber.name}\n"
        f"saveamberparm mol {rec_prmtop.name} {rec_inpcrd.name}\n"
        "quit\n"
    )
    res = subprocess.run(["tleap", "-f", leap_in.name],
                         capture_output=True, text=True,
                         timeout=TLEAP_TIMEOUT, env=env,
                         cwd=str(workdir))
    if res.returncode != 0 or not rec_prmtop.exists():
        return ReceptorPrep(target, workdir, rec_amber, Path(), Path(),
                            status="tleap_receptor_failed",
                            note=res.stdout[-2000:])

    return ReceptorPrep(target, workdir, rec_amber, rec_prmtop, rec_inpcrd)


# ------------------------------------------------------------ per-ligand ---

@dataclass
class LigandResult:
    chain_idx: int
    smiles: str
    vina_score: float
    vina_pose_score: Optional[float]
    formal_charge: int = 0
    charge_method: str = ""
    mm_gbsa_total: Optional[float] = None
    mm_gbsa_vdw: Optional[float] = None
    mm_gbsa_eel: Optional[float] = None
    mm_gbsa_egb: Optional[float] = None
    mm_gbsa_esurf: Optional[float] = None
    mm_gbsa_gas: Optional[float] = None
    mm_gbsa_solv: Optional[float] = None
    status: str = "ok"
    note: str = ""
    runtime_s: float = 0.0


def _formal_charge_from_sdf(sdf_path: Path) -> int:
    from rdkit import Chem
    m = Chem.SDMolSupplier(str(sdf_path), removeHs=False)[0]
    if m is None:
        # Fallback: parse SMILES from sibling manifest later.
        return 0
    return sum(a.GetFormalCharge() for a in m.GetAtoms())


_DELTA_RE = re.compile(
    r"Delta\s+\(Complex\s*-\s*Receptor\s*-\s*Ligand\)"
    r"(?P<body>.*?)(?=^-+\s*$|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)


def _parse_mmpbsa_results(results_dat: Path) -> Optional[dict]:
    """Parse the GB block from MMPBSA.py output.

    The relevant block looks roughly like::

        Differences (Complex - Receptor - Ligand):
        Energy Component          Average      ...
        ----------------------------------------------------
        VDWAALS                   -42.1234     ...
        EEL                       -10.5678     ...
        ...
        EGB                       28.9876      ...
        ESURF                     -5.4321      ...
        ----------------------------------------------------
        DELTA G gas               -52.6912     ...
        DELTA G solv              23.5555      ...
        DELTA TOTAL               -29.1357     ...
    """
    try:
        text = results_dat.read_text()
    except OSError:
        return None

    # Locate the GENERALIZED BORN block; in single-frame mode there is one.
    gb_block = re.search(r"GENERALIZED BORN.*?(?=POISSON BOLTZMANN|\Z)",
                         text, flags=re.DOTALL)
    if gb_block is None:
        # Some versions just print one block without a banner.
        gb_block = re.search(r".*", text, flags=re.DOTALL)
    block = gb_block.group(0) if gb_block else text

    delta = re.search(
        r"(?:Delta|Differences)\s*\(Complex\s*-\s*Receptor\s*-\s*Ligand\)\s*:?(.*?)\Z",
        block, flags=re.DOTALL | re.IGNORECASE,
    )
    if delta is None:
        return None
    body = delta.group(1)

    def _grab(label: str) -> Optional[float]:
        # Match a line that starts with the label and has a numeric Average column.
        m = re.search(
            rf"^\s*{re.escape(label)}\s+(-?\d+\.\d+)",
            body, flags=re.MULTILINE,
        )
        if m is None:
            return None
        return float(m.group(1))

    out = {
        "vdw":   _grab("VDWAALS"),
        "eel":   _grab("EEL"),
        "egb":   _grab("EGB"),
        "esurf": _grab("ESURF"),
        "gas":   _grab("DELTA G gas") or _grab("DELTA G  gas")
                  or _grab("G gas"),
        "solv":  _grab("DELTA G solv") or _grab("DELTA G  solv")
                  or _grab("G solv"),
        "total": _grab("DELTA TOTAL") or _grab("TOTAL"),
    }
    if out["total"] is None:
        return None
    return out


def run_one_ligand(
    *,
    generator: str,
    target: str,
    chain_idx: int,
    smiles: str,
    vina_score: float,
    vina_pose_score: Optional[float],
    sdf_path: Path,
    receptor_amber_pdb: Path,
    out_dir: Path,
) -> LigandResult:
    """Single-ligand pipeline. Returns a populated LigandResult."""
    out_json = out_dir / f"{chain_idx}.json"
    if out_json.exists():
        # Idempotent: trust on-disk record unless it was a tool failure.
        try:
            cached = json.loads(out_json.read_text())
        except Exception:
            cached = None
        if cached and cached.get("status") not in (None, "tool_failed",
                                                    "timeout"):
            return LigandResult(**{k: v for k, v in cached.items()
                                   if k in LigandResult.__dataclass_fields__})

    t0 = time.time()
    res = LigandResult(chain_idx=chain_idx, smiles=smiles,
                       vina_score=vina_score,
                       vina_pose_score=vina_pose_score)
    work = WORKDIR_ROOT / target / generator / str(chain_idx)
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)

    env = _amber_env()

    try:
        # Formal charge from SDF.
        try:
            res.formal_charge = _formal_charge_from_sdf(sdf_path)
        except Exception:
            res.formal_charge = 0

        # Step 1: antechamber AM1-BCC.
        lig_mol2 = work / "lig.gaff2.mol2"
        sdf_local = work / "lig.sdf"
        shutil.copy(sdf_path, sdf_local)
        ante_cmd = [
            "antechamber",
            "-i", sdf_local.name, "-fi", "sdf",
            "-o", lig_mol2.name, "-fo", "mol2",
            "-c", "bcc", "-at", "gaff2",
            "-nc", str(res.formal_charge),
            "-rn", "LIG", "-pf", "y",
        ]
        try:
            ante = subprocess.run(ante_cmd, capture_output=True, text=True,
                                  timeout=ANTECHAMBER_BCC_TIMEOUT,
                                  env=env, cwd=str(work))
        except subprocess.TimeoutExpired:
            res.status = "ante_failed"
            res.note = "antechamber bcc timeout"
            ante = None
        if ante is None or ante.returncode != 0 or not lig_mol2.exists():
            # Fallback: Gasteiger.
            for cleanup in work.glob("ANTECHAMBER*"):
                cleanup.unlink(missing_ok=True)
            for cleanup in work.glob("ATOMTYPE*"):
                cleanup.unlink(missing_ok=True)
            ante_cmd_g = [
                "antechamber",
                "-i", sdf_local.name, "-fi", "sdf",
                "-o", lig_mol2.name, "-fo", "mol2",
                "-c", "gas", "-at", "gaff2",
                "-nc", str(res.formal_charge),
                "-rn", "LIG", "-pf", "y",
            ]
            try:
                ante = subprocess.run(ante_cmd_g, capture_output=True,
                                      text=True,
                                      timeout=ANTECHAMBER_GAS_TIMEOUT,
                                      env=env, cwd=str(work))
            except subprocess.TimeoutExpired:
                ante = None
            if ante is None or ante.returncode != 0 or not lig_mol2.exists():
                res.status = "ante_failed"
                res.note = (ante.stderr[-1500:] if ante else "timeout")
                return res
            res.charge_method = "gas"
        else:
            res.charge_method = "bcc"

        # Step 2: parmchk2.
        frcmod = work / "lig.frcmod"
        pc_cmd = ["parmchk2", "-i", lig_mol2.name, "-f", "mol2",
                  "-o", frcmod.name, "-s", "gaff2"]
        try:
            pc = subprocess.run(pc_cmd, capture_output=True, text=True,
                                timeout=PARMCHK_TIMEOUT,
                                env=env, cwd=str(work))
        except subprocess.TimeoutExpired:
            res.status = "parmchk_failed"
            res.note = "parmchk2 timeout"
            return res
        if pc.returncode != 0 or not frcmod.exists():
            res.status = "parmchk_failed"
            res.note = pc.stderr[-1500:]
            return res

        # Step 3: tleap — build complex + ligand prmtops (receptor was
        # already built once per target; we rebuild a local copy here so
        # tleap's atom indexing matches the complex it produces).
        leap_in = work / "leap_complex.in"
        leap_in.write_text(
            "source leaprc.protein.ff14SB\n"
            "source leaprc.gaff2\n"
            f"LIG = loadmol2 {lig_mol2.name}\n"
            f"loadamberparams {frcmod.name}\n"
            f"PROT = loadpdb {receptor_amber_pdb}\n"
            "COMP = combine { PROT LIG }\n"
            "saveamberparm COMP complex.prmtop complex.inpcrd\n"
            "saveamberparm PROT receptor.prmtop receptor.inpcrd\n"
            "saveamberparm LIG ligand.prmtop ligand.inpcrd\n"
            "quit\n"
        )
        try:
            tl = subprocess.run(["tleap", "-f", leap_in.name],
                                capture_output=True, text=True,
                                timeout=TLEAP_TIMEOUT, env=env,
                                cwd=str(work))
        except subprocess.TimeoutExpired:
            res.status = "tleap_failed"
            res.note = "tleap timeout"
            return res
        if (tl.returncode != 0 or
                not (work / "complex.prmtop").exists() or
                not (work / "ligand.prmtop").exists() or
                not (work / "receptor.prmtop").exists()):
            res.status = "tleap_failed"
            res.note = tl.stdout[-1500:]
            return res

        # Step 4a: minimize the complex with GB. Without this step,
        # tleap-placed missing heavy atoms create local clashes that push
        # absolute VDW into the 1e15 range, destroying the float
        # precision of (Complex - Receptor) - Ligand. We restrain heavy
        # atoms of the protein so the binding pose is preserved while
        # local clashes relax.
        min_in = work / "min.in"
        # Heavy-restrained min: relieve clashes, preserve docked pose.
        # 200 cycles is enough — clash energy relaxes by step 100 (1e15 → 1e3 VDW).
        # cut=12.0 keeps wall-time linear-in-N (ALDH1=61k atoms would
        # otherwise take >4 h with infinite cutoff). This is fine for
        # *clash relief*; the MMPBSA single-point that follows uses
        # default infinite cutoff.
        min_in.write_text(
            "Restrained min — relieve clashes, preserve docked pose\n"
            "&cntrl\n"
            "  imin=1, maxcyc=200, ncyc=100,\n"
            "  ntmin=1, drms=0.1,\n"
            "  ntb=0, igb=5, cut=12.0, gbsa=0,\n"
            "  ntpr=50, ntwx=0,\n"
            "  ntr=1, restraint_wt=10.0,\n"
            "  restraintmask='!:LIG & !@H=',\n"
            "/\n"
        )
        try:
            sd = subprocess.run([
                "sander", "-O",
                "-i", min_in.name,
                "-p", "complex.prmtop",
                "-c", "complex.inpcrd",
                "-ref", "complex.inpcrd",
                "-o", "min.out",
                "-r", "complex.min.rst7",
                "-x", "min.dcd",
            ], capture_output=True, text=True,
               timeout=SANDER_TIMEOUT, env=env, cwd=str(work))
        except subprocess.TimeoutExpired:
            res.status = "tool_failed"
            res.note = "sander min timeout"
            return res
        if sd.returncode != 0 or not (work / "complex.min.rst7").exists():
            res.status = "tool_failed"
            res.note = (sd.stdout + "\n" + sd.stderr)[-1500:]
            return res

        # Step 4b: convert minimized rst7 → 1-frame DCD for MMPBSA.
        cpp_in = work / "to_dcd.in"
        cpp_in.write_text(
            "trajin complex.min.rst7\n"
            "trajout complex.dcd dcd\n"
            "go\n"
            "quit\n"
        )
        try:
            cp = subprocess.run(["cpptraj", "-p", "complex.prmtop",
                                 "-i", cpp_in.name],
                                capture_output=True, text=True,
                                timeout=CPPTRAJ_TIMEOUT, env=env,
                                cwd=str(work))
        except subprocess.TimeoutExpired:
            res.status = "tool_failed"
            res.note = "cpptraj timeout"
            return res
        if cp.returncode != 0 or not (work / "complex.dcd").exists():
            res.status = "tool_failed"
            res.note = cp.stdout[-1500:]
            return res

        # Step 5: MMPBSA.py.
        mmin = work / "mmpbsa.in"
        mmin.write_text(
            "Single-frame MM-GBSA igb=5\n"
            "&general\n"
            "   startframe=1, endframe=1, interval=1, keep_files=0,\n"
            "/\n"
            "&gb\n"
            "   igb=5, saltcon=0.150,\n"
            "/\n"
        )
        results_dat = work / "RESULTS.dat"
        cmd = ["MMPBSA.py", "-O", "-i", mmin.name,
               "-cp", "complex.prmtop",
               "-rp", "receptor.prmtop",
               "-lp", "ligand.prmtop",
               "-y", "complex.dcd",
               "-o", results_dat.name]
        try:
            mp = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=MMPBSA_TIMEOUT, env=env,
                                cwd=str(work))
        except subprocess.TimeoutExpired:
            res.status = "timeout"
            res.note = "MMPBSA.py timeout"
            return res
        if mp.returncode != 0 or not results_dat.exists():
            res.status = "mmpbsa_failed"
            tail = (mp.stdout + "\n" + mp.stderr)[-1800:]
            res.note = tail
            return res

        parsed = _parse_mmpbsa_results(results_dat)
        if parsed is None:
            res.status = "mmpbsa_failed"
            res.note = "could not parse RESULTS.dat"
            return res
        res.mm_gbsa_total = parsed["total"]
        res.mm_gbsa_vdw = parsed["vdw"]
        res.mm_gbsa_eel = parsed["eel"]
        res.mm_gbsa_egb = parsed["egb"]
        res.mm_gbsa_esurf = parsed["esurf"]
        res.mm_gbsa_gas = parsed["gas"]
        res.mm_gbsa_solv = parsed["solv"]
        res.status = "ok"

    except Exception as exc:  # pragma: no cover - defensive
        res.status = "tool_failed"
        res.note = f"unexpected: {exc}\n{traceback.format_exc()[-1200:]}"
    finally:
        res.runtime_s = time.time() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(asdict(res), indent=2))
        # Cleanup workdir on success only — keep failures for debugging.
        if res.status == "ok":
            shutil.rmtree(work, ignore_errors=True)

    return res


# ----------------------------------------------------------- orchestration --

def _load_top_k_jobs(generator: str, target: str, top_k: int):
    """Return list of (chain_idx, smiles, vina_score, vina_pose_score, sdf)."""
    poses_dir = POSES_ROOT / generator / target
    manifest = poses_dir / "manifest.parquet"
    if not manifest.exists():
        logger.warning("no manifest %s/%s", generator, target)
        return []
    df = pd.read_parquet(manifest)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        return []
    df = df.sort_values("vina_score", ascending=True).head(top_k)
    jobs = []
    for _, r in df.iterrows():
        chain_idx = int(r["chain_idx"])
        sdf = poses_dir / f"{chain_idx}.sdf"
        if not sdf.exists():
            continue
        jobs.append({
            "chain_idx": chain_idx,
            "smiles": str(r["smiles"]),
            "vina_score": float(r["vina_score"]),
            "vina_pose_score": (float(r["vina_pose_score"])
                                if pd.notnull(r.get("vina_pose_score")) else None),
            "sdf_path": sdf,
        })
    return jobs


def _per_target_parquet(generator: str, target: str, out_root: Path) -> Path:
    json_dir = out_root / generator / target
    rows = []
    if json_dir.exists():
        for jf in sorted(json_dir.glob("*.json")):
            try:
                d = json.loads(jf.read_text())
                rows.append(d)
            except Exception:
                continue
    pq = out_root / generator / f"{target}.parquet"
    pq.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(pq, index=False)
    return pq


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--generators", nargs="+", default=ALL_GENERATORS,
                   choices=ALL_GENERATORS,
                   help="Subset of generators to process.")
    p.add_argument("--targets", nargs="+", default=ALL_TARGETS,
                   choices=ALL_TARGETS,
                   help="Subset of targets to process.")
    p.add_argument("--top_k", type=int, default=10,
                   help="Number of poses per (gen,target) — by Vina score.")
    p.add_argument("--n_workers", type=int, default=8,
                   help="Process pool size for ligand-level fan-out.")
    p.add_argument("--out_root", type=Path,
                   default=Path("results/eval/phase7/mm_gbsa"))
    p.add_argument("--seed", type=int, default=42,
                   help="Reserved for reproducibility manifests.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    WORKDIR_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: per-target receptor preps (serial, cached). -------------
    receptor_preps = {}
    for target in args.targets:
        prep = prepare_receptor(target)
        if prep.status != "ok":
            logger.error("receptor prep failed for %s: %s — %s",
                         target, prep.status, prep.note[:200])
        receptor_preps[target] = prep

    # ---- Step 2: enumerate ligand jobs. ----------------------------------
    jobs = []
    for gen in args.generators:
        for tgt in args.targets:
            prep = receptor_preps[tgt]
            if prep.status != "ok":
                logger.warning("skipping %s/%s — receptor prep failed", gen, tgt)
                continue
            for j in _load_top_k_jobs(gen, tgt, args.top_k):
                out_dir = args.out_root / gen / tgt
                out_json = out_dir / f"{j['chain_idx']}.json"
                if out_json.exists():
                    try:
                        cached = json.loads(out_json.read_text())
                        if cached.get("status") not in (None, "tool_failed",
                                                         "timeout"):
                            continue
                    except Exception:
                        pass
                jobs.append({
                    "generator": gen,
                    "target": tgt,
                    "chain_idx": j["chain_idx"],
                    "smiles": j["smiles"],
                    "vina_score": j["vina_score"],
                    "vina_pose_score": j["vina_pose_score"],
                    "sdf_path": j["sdf_path"],
                    "receptor_amber_pdb": prep.receptor_amber_pdb,
                    "out_dir": out_dir,
                })

    logger.info("%d ligand jobs to run (%d workers, top_k=%d)",
                len(jobs), args.n_workers, args.top_k)

    if args.dry_run:
        for j in jobs[:20]:
            logger.info("DRY %s/%s/%s sdf=%s",
                        j["generator"], j["target"], j["chain_idx"],
                        j["sdf_path"])
        if len(jobs) > 20:
            logger.info("... %d more", len(jobs) - 20)
        return 0

    # ---- Step 3: fan out. ------------------------------------------------
    ok_count = 0
    fail_count = 0
    if args.n_workers <= 1:
        for job in jobs:
            r = run_one_ligand(**job)
            ok_count += int(r.status == "ok")
            fail_count += int(r.status != "ok")
            if (ok_count + fail_count) % 10 == 0:
                logger.info("progress %d/%d (ok=%d fail=%d)",
                            ok_count + fail_count, len(jobs),
                            ok_count, fail_count)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = [pool.submit(run_one_ligand, **job) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    r = fut.result()
                except Exception as exc:
                    logger.exception("worker crashed: %s", exc)
                    fail_count += 1
                    continue
                if r.status == "ok":
                    ok_count += 1
                else:
                    fail_count += 1
                if i % 10 == 0:
                    logger.info("progress %d/%d (ok=%d fail=%d)",
                                i, len(jobs), ok_count, fail_count)

    # ---- Step 4: per-target parquet rollup. ------------------------------
    for gen in args.generators:
        for tgt in args.targets:
            _per_target_parquet(gen, tgt, args.out_root)

    # ---- Step 5: run manifest. -------------------------------------------
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_sha = ""
    manifest = {
        "task_id": "01_mm_gbsa",
        "git_sha": git_sha,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "n_jobs_total": len(jobs),
        "n_ok": ok_count,
        "n_fail": fail_count,
    }
    (args.out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    logger.info("done: ok=%d fail=%d", ok_count, fail_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
