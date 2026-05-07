"""Phase-7 task 4 — explicit-solvent MD validation of docked Vina poses.

For one (generator, target, chain_idx) the script:

  1. Loads ``data/external/receptors/<target>/receptor_clean.pdb`` and
     trims to pocket-bearing chains (15 Å of box center) to keep large
     multimers (ALDH1, PKM2) tractable.
  2. Loads ``results/eval/phase7/poses/<gen>/<target>/<chain_idx>.sdf``
     as an RDKit mol with explicit Hs + 3D coords; recovers stereo from
     the 3D pose; passes to OpenFF; assigns AM1-BCC charges via NAGL.
  3. Builds an OpenMM system with ff14SB + TIP3P + GAFF-2.11, 10 Å
     solvent padding, 0.15 M NaCl, hydrogen-mass=1.5 amu, PME, HBonds.
  4. Minimizes (≤ 10000 iter), then NVT 100 ps with a 10 kJ/mol/nm²
     restraint on protein heavy atoms, then NPT 100 ps unrestrained.
  5. Production: ``--duration_ns`` (default 20) at 4 fs timestep,
     DCDReporter every 2500 steps (= 10 ps frame spacing).
  6. Analysis with mdtraj: image trajectory, ligand-RMSD vs the docked
     pose after protein-backbone alignment, contact persistence for
     residues within 4 Å of the ligand in the docked pose.

Outputs
-------

::

    results/eval/phase7/md/<gen>/<target>/<chain_idx>/
        topology.pdb          # equilibrated solvated system
        prod.dcd              # production trajectory
        prod.log              # OpenMM StateDataReporter
        metrics.json          # rmsd_lig, rmsf_lig, contact_persistence
        status.json           # status + timings + exception (if any)

Status codes
------------

* ``ok`` — minimisation, equilibration, production and analysis all
  finished cleanly.
* ``missing_input`` — receptor or pose SDF not found.
* ``ligand_charge_failed`` — RDKit / NAGL / GAFF prep raised.
* ``system_build_failed`` — Modeller, addSolvent or createSystem raised.
* ``equilibration_nan`` — NaN energy during minimisation or equil.
* ``production_nan`` — NaN energy during production.
* ``analysis_failed`` — trajectory exists but mdtraj analysis raised.
* ``timeout`` — wall budget exceeded.

The script always writes ``status.json`` even on failure so the outer
queue can resume cleanly.

Convention notes
----------------

* The original ``data/external/receptors/<target>/receptor_clean.pdb``
  is already HETATM-stripped — there are no cofactors in any of the
  15 receptors. Task 1 (MM-GBSA) was also run cofactor-free, so this
  script keeps the same policy for cross-task consistency. For
  KAT2A/IDH1/MAPK1 specifically this is a pre-registered limitation.
* CUDA-only; falls back to CPU OpenMM if CUDA platform not available
  but loudly warns since 4 fs HMR was tuned for GPU throughput.
* Idempotent: if ``status.json`` exists with status='ok', skip.

Spec: ``docs/validation/04_md_stability.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("run_md_validation")

# ---------------------------------------------------------------- constants --

ALL_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]
ALL_GENERATORS = ["thermofrag", "targetdiff", "rxnflow", "bbar"]

REPO_ROOT = Path(__file__).resolve().parent.parent
POSES_ROOT = REPO_ROOT / "results/eval/phase7/poses"
RECEPTOR_ROOT = REPO_ROOT / "data/external/receptors"
OUT_ROOT_DEFAULT = REPO_ROOT / "results/eval/phase7/md"
WORKDIR_ROOT = Path("/tmp/thermofrag_validation/04_md")

# AmberTools env: openmmforcefields' GAFFTemplateGenerator shells out to
# antechamber/parmchk2, which need AMBERHOME and the env's bin/ on PATH.
TF_EVAL_ROOT = "/home/zhao/miniconda3/envs/tf-eval"
os.environ.setdefault("AMBERHOME", TF_EVAL_ROOT)
os.environ["PATH"] = f"{TF_EVAL_ROOT}/bin:{os.environ.get('PATH', '')}"

POCKET_CHAIN_RADIUS = 15.0  # Å — same as run_mm_gbsa.py

# Reuse the prep helpers + cached receptor_amber.pdb from run_mm_gbsa
# so MD operates on the *same* protonated structure that MM-GBSA
# (Task 1) rescored. Avoids drifting between tasks on HIS protonation.
TASK1_RECEPTOR_CACHE = Path("/tmp/thermofrag_validation/01_mm_gbsa")

sys.path.insert(0, str(REPO_ROOT / "scripts"))
try:
    from run_mm_gbsa import (_trim_to_pocket_chains, prepare_receptor as
                              _task1_prepare_receptor)  # type: ignore
except Exception as exc:  # pragma: no cover
    logger.warning("could not import helpers from run_mm_gbsa: %s", exc)
    _trim_to_pocket_chains = None  # type: ignore
    _task1_prepare_receptor = None  # type: ignore

NAGL_MODEL = "openff-gnn-am1bcc-0.1.0-rc.3.pt"

# ----------------------------------------------------------------- helpers --


@dataclass
class MDResult:
    generator: str
    target: str
    chain_idx: int
    status: str = "pending"
    note: str = ""
    n_atoms: int = 0
    n_water: int = 0
    duration_ns: float = 0.0
    wall_seconds_equil: float = 0.0
    wall_seconds_prod: float = 0.0
    rmsd_mean: Optional[float] = None
    rmsd_std: Optional[float] = None
    rmsd_max: Optional[float] = None
    frac_below_3A: Optional[float] = None
    n_persistent_contacts: Optional[int] = None
    contact_persistence: Optional[dict] = None


def _load_top_k_jobs(generator: str, target: str, top_k: int):
    """Read the pose manifest and return top-K (chain_idx, smiles, sdf)."""
    import pandas as pd
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
    out = []
    for _, r in df.iterrows():
        c = int(r["chain_idx"])
        sdf = poses_dir / f"{c}.sdf"
        if not sdf.exists():
            continue
        out.append({
            "chain_idx": c,
            "smiles": str(r["smiles"]),
            "vina_score": float(r["vina_score"]),
            "sdf_path": sdf,
        })
    return out


def _resolve_receptor(target: str) -> Path:
    """Return path to an MD-ready receptor PDB (Hs + missing side chains rebuilt).

    Task 1 used pdb4amber **without** ``--add-missing-atoms`` because for
    MMPBSA single-point energies, naive heavy-atom completion produces
    severe VDW clashes that destroy the (Cmplx − Rcptr) − Lig delta.
    For MD this is fine — the 10k-iteration minimisation absorbs the
    clashes. So this routine runs a separate pdb4amber pass with
    ``--add-missing-atoms`` and caches it under the MD work-tree.

    Pocket trimming (ALDH1/PKM2) is reused from the Task 1 trimmer.
    """
    md_cache = WORKDIR_ROOT / target / "receptor"
    md_cache.mkdir(parents=True, exist_ok=True)
    out = md_cache / "receptor_md.pdb"
    if out.exists():
        return out.resolve()

    src = (RECEPTOR_ROOT / target / "receptor_clean.pdb").resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    # Pocket trim for large multimers, before pdb4amber.
    n_atoms_full = sum(1 for ln in src.read_text().splitlines()
                       if ln.startswith("ATOM "))
    trimmed = md_cache / "receptor_clean_pocket.pdb"
    if n_atoms_full > 10000 and _trim_to_pocket_chains is not None:
        box = json.loads((RECEPTOR_ROOT / target / "box.json").read_text())
        kept, n_after = _trim_to_pocket_chains(
            src, trimmed, tuple(box["center"]), POCKET_CHAIN_RADIUS)
        logger.info("[%s] trimmed receptor: kept chains %s, %d atoms (was %d)",
                    target, sorted(kept), n_after, n_atoms_full)
    else:
        shutil.copyfile(src, trimmed)

    # Build the MD-ready PDB. Heavy atoms only — Hs come from
    # OpenMM Modeller.addHydrogens later, with ideal geometry. Doing
    # H placement in OpenMM (instead of via pdb4amber --reduce) avoids
    # the residual VDW clashes that L-BFGS can't escape.
    import subprocess
    env = os.environ.copy()
    env["PATH"] = f"/home/zhao/miniconda3/envs/tf-eval/bin:{env.get('PATH', '')}"
    env["AMBERHOME"] = "/home/zhao/miniconda3/envs/tf-eval"
    env.setdefault("LANG", "C")
    raw = md_cache / "receptor_md_raw.pdb"
    cmd = [
        "pdb4amber",
        "-i", str(trimmed.resolve()),
        "-o", str(raw.resolve()),
        "--dry", "--no-conect",
        "--add-missing-atoms",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=900, env=env, cwd=str(md_cache))
    if res.returncode != 0 or not raw.exists():
        raise RuntimeError(
            f"pdb4amber failed for {target}: {res.stderr[-1500:]}")

    # Strip Hs — OpenMM Modeller.addHydrogens then places them with
    # ideal geometry from ff14SB templates and decides HIS protonation.
    # Also rename HIE/HID/HIP back to HIS so OpenMM doesn't see a
    # partially-protonated residue as the wrong template.
    n_in, n_out = 0, 0
    with raw.open() as fh, out.open("w") as fo:
        for ln in fh:
            if ln.startswith(("ATOM", "HETATM")):
                n_in += 1
                if ln[12:16].strip().startswith("H"):
                    continue
                # Normalise HIS protonation tags so Modeller picks fresh.
                resname = ln[17:20]
                if resname in ("HIE", "HID", "HIP"):
                    ln = ln[:17] + "HIS" + ln[20:]
                fo.write(ln)
                n_out += 1
            else:
                fo.write(ln)
    logger.info("[%s] receptor_md.pdb built — %d heavy atoms (stripped %d Hs)",
                target, n_out, n_in - n_out)
    return out.resolve()


# --------------------------------------------------------------- ligand prep --


def _prepare_ligand(sdf_path: Path):
    """Returns (openff Molecule, rdkit mol). NAGL AM1-BCC charges assigned."""
    from rdkit import Chem
    from openff.toolkit.topology import Molecule

    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not parse {sdf_path}")
    if not any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        # Pose SDFs are written with explicit Hs but guard anyway.
        mol = Chem.AddHs(mol, addCoords=True)
    Chem.AssignStereochemistryFrom3D(mol)
    omol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    omol.assign_partial_charges(NAGL_MODEL)
    return omol, mol


# --------------------------------------------------------------- system build --


def _build_system(receptor_pdb: Path, omol, padding_ang: float = 10.0,
                  with_hbonds: bool = True, solvate: bool = True):
    """Build a solvated OpenMM Modeller + system. Returns (modeller, system, ff).

    `with_hbonds=False` returns a system without bond constraints — used
    for the initial minimisation pass so the L-BFGS minimizer has full
    DOF to relieve clashes around the docked ligand. The production
    system uses HBonds + 4 fs HMR.
    """
    from openmm.app import PDBFile, Modeller, ForceField, PME, NoCutoff, HBonds
    from openmm import unit as u
    from openmmforcefields.generators import GAFFTemplateGenerator

    gaff = GAFFTemplateGenerator(molecules=[omol], forcefield="gaff-2.11")
    ff = ForceField("amber/ff14SB.xml", "amber/tip3p_standard.xml",
                    "amber/tip3p_HFE_multivalent.xml")
    ff.registerTemplateGenerator(gaff.generator)

    # Combine protein + ligand. receptor_pdb is heavy-atom-only
    # (pdb4amber --add-missing-atoms, no --reduce). Add Hs via OpenMM
    # Modeller — uses ideal geometry from ff14SB templates and
    # auto-picks HIS protonation, so we skip the clashy pdb4amber
    # `--reduce` Hs that L-BFGS could not escape.
    prot = PDBFile(str(receptor_pdb))
    modeller = Modeller(prot.topology, prot.positions)
    modeller.addHydrogens(ff, pH=7.0)
    lig_topology = omol.to_topology().to_openmm()
    lig_positions = omol.conformers[0].to_openmm()
    modeller.add(lig_topology, lig_positions)

    # Solvate (padding + 0.15 M NaCl) for the production system. The
    # pre-solvation complex is also useful for AMBER minimisation because
    # ParmEd can serialize protein/ligand bonds cleanly, while OpenMM's
    # TIP3P water constraints appear as untyped water bonds in ParmEd.
    if solvate:
        modeller.addSolvent(ff, model="tip3p",
                            padding=padding_ang * u.angstrom,
                            ionicStrength=0.15 * u.molar)

    constraints_arg = HBonds if with_hbonds else None
    hmass = 1.5 * u.amu if with_hbonds else 1.0 * u.amu
    system = ff.createSystem(modeller.topology,
                             nonbondedMethod=PME if solvate else NoCutoff,
                             nonbondedCutoff=1.0 * u.nanometer,
                             constraints=constraints_arg,
                             hydrogenMass=hmass)
    return modeller, system, ff


def _sander_minimise(modeller, system, workdir: Path,
                     generator: str, target: str, chain_idx: int):
    """Run AMBER sander minimisation on the OpenMM-built system.

    Round-trip: parmed writes the OpenMM `system + topology` as Amber
    prmtop + inpcrd, sander runs SD → CG with weak protein-heavy
    restraint, parmed reads the minimised inpcrd back. Returns the
    minimised positions ready for ``sim.context.setPositions``.
    """
    import subprocess
    import parmed
    from openmm import unit as u

    sander_dir = workdir / "sander_min"
    sander_dir.mkdir(parents=True, exist_ok=True)

    prmtop = sander_dir / "system.prmtop"
    inpcrd = sander_dir / "system.inpcrd"
    minrst = sander_dir / "min.rst7"
    minout = sander_dir / "min.out"
    minin = sander_dir / "min.in"
    refcrd = sander_dir / "ref.rst7"

    if prmtop.exists() and minrst.exists():
        new = parmed.load_file(str(prmtop), str(minrst))
        logger.info("[%s/%s/%d] reusing cached sander minimisation",
                    generator, target, chain_idx)
        return new.coordinates * u.angstrom

    # Dump the OpenMM solvated system as prmtop+inpcrd via parmed.
    # This must be an unconstrained minimisation system. ParmEd cannot
    # reliably serialize the HBond-constrained/HMR production system:
    # constrained bonds may have no HarmonicBondForce type, which trips
    # AmberParm.from_structure() with ``bond.type is None``.
    pmd_struct = parmed.openmm.load_topology(modeller.topology, system,
                                              xyz=modeller.positions)
    pmd_struct.box = pmd_struct.box  # ensures box vectors flow through
    pmd_struct.save(str(prmtop), overwrite=True)
    pmd_struct.save(str(inpcrd), format="rst7", overwrite=True)
    pmd_struct.save(str(refcrd), format="rst7", overwrite=True)

    # Sander minimisation input.
    has_box = pmd_struct.box is not None
    ntb = 1 if has_box else 0
    cut = 10.0 if has_box else 12.0
    minin.write_text(
        "minimisation\n"
        " &cntrl\n"
        "  imin=1, maxcyc=4000, ncyc=1000,\n"
        f"  ntb={ntb}, ntp=0,\n"
        "  ntr=1, restraint_wt=10.0,\n"
        "  restraintmask=':1-9999 & !@H= & !:WAT,Na+,Cl-,UNK,UNL,LIG,MOL',\n"
        f"  cut={cut:.1f},\n"
        "  ntpr=200, ntwx=0,\n"
        " /\n"
    )

    env = os.environ.copy()
    env["AMBERHOME"] = TF_EVAL_ROOT
    env["PATH"] = f"{TF_EVAL_ROOT}/bin:{env.get('PATH', '')}"
    env.setdefault("OMP_NUM_THREADS", "1")
    cmd = ["sander", "-O",
           "-i", str(minin),
           "-o", str(minout),
           "-p", str(prmtop),
           "-c", str(inpcrd),
           "-r", str(minrst),
           "-ref", str(refcrd)]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True,
                         timeout=1800, env=env, cwd=str(sander_dir))
    if res.returncode != 0 or not minrst.exists():
        raise RuntimeError(
            f"sander failed for {generator}/{target}/{chain_idx}: "
            f"returncode={res.returncode} stderr={res.stderr[-1500:]}")

    # Read minimised coordinates back and return as OpenMM Quantity.
    new = parmed.load_file(str(prmtop), str(minrst))
    coords = new.coordinates  # (N, 3) numpy in Å
    logger.info("[%s/%s/%d] sander min done in %.0fs",
                generator, target, chain_idx, time.time() - t0)
    return coords * u.angstrom


def _staged_openmm_minimise(modeller, system_min, system_prod, gpu_id: int,
                             generator: str, target: str, chain_idx: int):
    """Three-stage OpenMM minimisation of a solvated complex.

    The docked pose typically has ~tens of clashing water/ligand pairs.
    A naive minim on the full unconstrained system cannot escape these
    in 20k iterations. We stage the relaxation so each stage only has
    a tractable number of free DOF:

      A. Restrain *both* protein heavy atoms AND ligand heavy atoms with
         k=2000 kJ/mol/nm² → only solvent + ions move. Resolves the
         worst water-ligand and water-protein clashes from addSolvent.
      B. Release ligand restraint, keep protein restraint at 200
         kJ/mol/nm² → ligand can shift inside the pocket to relieve
         residual clashes from the docked geometry.
      C. Switch to the constrained production system with k=0; brief
         L-BFGS to settle to the production force-field's minimum.

    Returns positions (Quantity in nm) ready for the production sim.
    """
    from openmm.app import Simulation
    from openmm import (CustomExternalForce, LangevinMiddleIntegrator,
                        Platform, unit as u)
    import numpy as _np

    # Build harmonic restraint forces on system_min for staged use.
    # Use unique global param names (k_minP, k_minL) so they don't
    # collide with the production-stage restraint's global "k".
    expr_p = "0.5*k_minP*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    expr_l = "0.5*k_minL*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    force_prot = CustomExternalForce(expr_p)
    force_prot.addGlobalParameter("k_minP", 0.0)
    force_prot.addPerParticleParameter("x0")
    force_prot.addPerParticleParameter("y0")
    force_prot.addPerParticleParameter("z0")

    force_lig = CustomExternalForce(expr_l)
    force_lig.addGlobalParameter("k_minL", 0.0)
    force_lig.addPerParticleParameter("x0")
    force_lig.addPerParticleParameter("y0")
    force_lig.addPerParticleParameter("z0")

    positions = modeller.positions
    n_prot = n_lig = 0
    solvent_resnames = ("HOH", "WAT", "NA", "CL", "MG", "ZN", "K", "SOD",
                        "CLA", "POT")
    ligand_resnames = ("UNK", "UNL", "LIG", "MOL")

    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        rn = atom.residue.name
        if rn in solvent_resnames:
            continue
        idx = atom.index
        # Convert position to nm.
        if hasattr(positions[idx], "value_in_unit"):
            arr = positions[idx].value_in_unit(u.nanometer)
            xyz = [float(arr[0]), float(arr[1]), float(arr[2])]
        else:
            xyz = [float(positions[idx][0]),
                   float(positions[idx][1]),
                   float(positions[idx][2])]
        if rn in ligand_resnames:
            force_lig.addParticle(idx, xyz)
            n_lig += 1
        else:
            force_prot.addParticle(idx, xyz)
            n_prot += 1

    fp_idx = system_min.addForce(force_prot)
    fl_idx = system_min.addForce(force_lig)
    logger.info("[%s/%s/%d] staged min restraints: %d protein heavy, %d ligand heavy",
                generator, target, chain_idx, n_prot, n_lig)

    # Pick a usable platform for the minimisation simulation.
    integrator = LangevinMiddleIntegrator(300 * u.kelvin,
                                          1.0 / u.picosecond,
                                          1.0 * u.femtosecond)
    sim = None
    last_exc = None
    platform_order = (("CPU", {}),) if gpu_id < 0 else (
        ("CUDA",   {"Precision": "mixed", "DeviceIndex": str(gpu_id)}),
        ("OpenCL", {"Precision": "mixed", "DeviceIndex": str(gpu_id)}),
        ("CPU",    {}),
    )
    for plat_name, props in platform_order:
        try:
            platform = Platform.getPlatformByName(plat_name)
        except Exception as exc:
            last_exc = exc
            continue
        try:
            sim = Simulation(modeller.topology, system_min, integrator,
                             platform, props)
            sim.context.setPositions(modeller.positions)
            logger.info("[%s/%s/%d] min platform %s",
                        generator, target, chain_idx, plat_name)
            break
        except Exception as exc:
            last_exc = exc
            integrator = LangevinMiddleIntegrator(300 * u.kelvin,
                                                  1.0 / u.picosecond,
                                                  1.0 * u.femtosecond)
    if sim is None:
        raise RuntimeError(f"no usable OpenMM platform for min: {last_exc}")

    def _pe():
        return sim.context.getState(getEnergy=True).getPotentialEnergy(
            ).value_in_unit(u.kilojoule_per_mole)

    # Stage A: solvent only.
    sim.context.setParameter("k_minP", 2000.0)
    sim.context.setParameter("k_minL", 2000.0)
    sim.minimizeEnergy(
        tolerance=10 * u.kilojoule_per_mole / u.nanometer,
        maxIterations=10_000)
    pe_a = _pe()
    logger.info("[%s/%s/%d] stage-A (solvent-only) PE=%.3e kJ/mol",
                generator, target, chain_idx, pe_a)
    if pe_a != pe_a or pe_a > 1e9:
        # Stage A still unphysical → likely a bad starting pose.
        raise RuntimeError(f"stage-A PE non-physical: {pe_a}")

    # Stage B: protein restrained, ligand free.
    sim.context.setParameter("k_minP", 200.0)
    sim.context.setParameter("k_minL", 0.0)
    sim.minimizeEnergy(
        tolerance=5 * u.kilojoule_per_mole / u.nanometer,
        maxIterations=10_000)
    pe_b = _pe()
    logger.info("[%s/%s/%d] stage-B (lig-free) PE=%.3e kJ/mol",
                generator, target, chain_idx, pe_b)
    if pe_b != pe_b or pe_b > 1e8:
        raise RuntimeError(f"stage-B PE non-physical: {pe_b}")

    # Stage C: fully unrestrained on the unconstrained system.
    sim.context.setParameter("k_minP", 0.0)
    sim.context.setParameter("k_minL", 0.0)
    sim.minimizeEnergy(
        tolerance=1 * u.kilojoule_per_mole / u.nanometer,
        maxIterations=10_000)
    pe_c = _pe()
    logger.info("[%s/%s/%d] stage-C (free) PE=%.3e kJ/mol",
                generator, target, chain_idx, pe_c)
    if pe_c != pe_c or pe_c > 1e7:
        raise RuntimeError(f"stage-C PE non-physical: {pe_c}")

    state = sim.context.getState(getPositions=True)
    return state.getPositions(asNumpy=False)


def _add_protein_heavy_restraint(system, modeller, k_kj_mol_nm2: float = 1000.0):
    """Add a CustomExternalForce harmonic restraint on protein heavy atoms.

    Returns the force index so caller can later set k=0 to release.
    """
    from openmm import CustomExternalForce
    from openmm import unit as u

    def _xyz_nm(pos):
        """Return a position triple in OpenMM's native nanometer units."""
        if hasattr(pos, "x"):
            return pos.x, pos.y, pos.z
        if hasattr(pos, "value_in_unit"):
            arr = pos.value_in_unit(u.nanometer)
        else:
            arr = pos
        return float(arr[0]), float(arr[1]), float(arr[2])

    expr = "0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    force = CustomExternalForce(expr)
    force.addGlobalParameter("k", k_kj_mol_nm2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    positions = modeller.positions
    n_restrained = 0
    for atom in modeller.topology.atoms():
        if atom.element is None or atom.element.symbol == "H":
            continue
        # Restrain only protein heavy atoms (not waters, ions, ligand).
        if atom.residue.name in ("HOH", "WAT", "NA", "CL", "MG", "ZN", "K"):
            continue
        if atom.residue.name == "LIG":
            continue
        # Detect the auto-generated ligand resname from openff (default 'UNK')
        if atom.residue.name in ("UNK", "UNL"):
            continue
        idx = atom.index
        force.addParticle(idx, list(_xyz_nm(positions[idx])))
        n_restrained += 1
    system.addForce(force)
    return force, n_restrained


def _as_vec3_positions_nm(positions):
    """Normalize positions to a list of Vec3 in nanometers for Modeller."""
    from openmm import Vec3
    from openmm import unit as u

    if hasattr(positions, "value_in_unit"):
        arr = positions.value_in_unit(u.nanometer)
        return [Vec3(float(x), float(y), float(z)) for x, y, z in arr] * u.nanometer

    out = []
    for pos in positions:
        if hasattr(pos, "x"):
            out.append(Vec3(float(pos.x), float(pos.y), float(pos.z)))
        else:
            out.append(Vec3(float(pos[0]), float(pos[1]), float(pos[2])))
    return out * u.nanometer


# ------------------------------------------------------------------ analysis --


def _analyse_trajectory(topology_pdb: Path, dcd: Path, lig_resname_candidates) -> dict:
    """Compute lig RMSD vs frame 0 + contact persistence with mdtraj."""
    import numpy as np
    import mdtraj as md

    traj = md.load(str(dcd), top=str(topology_pdb))
    # Image so the ligand doesn't jump across PBC.
    traj = traj.image_molecules(inplace=False)

    # Identify ligand atoms.
    top = traj.topology
    lig_indices = []
    lig_resname = None
    for cand in lig_resname_candidates:
        sel = top.select(f"resname {cand}")
        if len(sel) > 0:
            lig_indices = sel
            lig_resname = cand
            break
    if len(lig_indices) == 0:
        # Fall back: anything that isn't standard protein/water/ion.
        std = ("HOH WAT NA CL MG ZN K SOD CLA POT ALA ARG ASN ASP CYS GLN GLU "
               "GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL HID HIE HIP "
               "CYX CYM ASH GLH LYN").split()

        for res in top.residues:
            if res.name not in std and not res.is_water:
                lig_indices = np.array([a.index for a in res.atoms])
                lig_resname = res.name
                break
    if len(lig_indices) == 0:
        raise ValueError("could not locate ligand atoms in topology")

    prot_bb = top.select("backbone")
    if len(prot_bb) == 0:
        raise ValueError("no protein backbone atoms found")

    # Heavy-atom-only ligand RMSD vs frame 0 after BB superposition.
    lig_heavy = np.array([i for i in lig_indices
                          if top.atom(int(i)).element.symbol != "H"])
    if len(lig_heavy) == 0:
        raise ValueError("no ligand heavy atoms")

    traj.superpose(traj, frame=0, atom_indices=prot_bb)
    diff = traj.xyz[:, lig_heavy, :] - traj.xyz[0:1, lig_heavy, :]
    rmsd_nm = np.sqrt((diff ** 2).sum(-1).mean(-1))
    rmsd_A = rmsd_nm * 10.0  # nm → Å

    # Contact persistence: residues within 4 Å of ligand in frame 0.
    cutoff_nm = 0.4
    pairs = []
    for res in top.residues:
        if res.is_water or res.name in ("NA", "CL", "MG", "ZN", "K"):
            continue
        if res.name == lig_resname:
            continue
        atoms = np.array([a.index for a in res.atoms
                          if a.element.symbol != "H"])
        if len(atoms) == 0:
            continue
        # Min distance from this residue to the ligand at frame 0.
        d0 = np.linalg.norm(
            traj.xyz[0, atoms[:, None], :] - traj.xyz[0, lig_heavy[None, :], :],
            axis=-1).min()
        if d0 < cutoff_nm:
            pairs.append((res, atoms))

    persistence = {}
    for res, atoms in pairs:
        d_per_frame = np.linalg.norm(
            traj.xyz[:, atoms[:, None], :] - traj.xyz[:, lig_heavy[None, :], :],
            axis=-1).min(axis=(1, 2))
        persistence[f"{res.name}{res.resSeq}"] = float((d_per_frame < cutoff_nm).mean())

    n_persistent = sum(1 for v in persistence.values() if v >= 0.6)
    return {
        "n_frames": int(traj.n_frames),
        "rmsd_mean": float(rmsd_A.mean()),
        "rmsd_std": float(rmsd_A.std()),
        "rmsd_max": float(rmsd_A.max()),
        "frac_below_3A": float((rmsd_A < 3.0).mean()),
        "n_persistent_contacts": int(n_persistent),
        "contact_persistence": persistence,
        "lig_resname": lig_resname,
    }


# -------------------------------------------------------------- main worker --


def run_one(generator: str, target: str, chain_idx: int,
            sdf_path: Path, smiles: str, vina_score: float,
            out_dir: Path, duration_ns: float = 20.0,
            equil_only: bool = False,
            padding_ang: float = 10.0,
            equil_ps: float = 100.0,
            gpu_id: int = 0) -> MDResult:
    """Driver for one ligand. Writes status.json + metrics.json."""
    import numpy as np
    from openmm.app import PDBFile, DCDReporter, StateDataReporter, Simulation
    from openmm import (LangevinMiddleIntegrator, MonteCarloBarostat, Platform,
                        unit as u)

    res = MDResult(generator=generator, target=target, chain_idx=chain_idx,
                   duration_ns=duration_ns)

    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = WORKDIR_ROOT / generator / target / str(chain_idx)
    workdir.mkdir(parents=True, exist_ok=True)

    status_path = out_dir / "status.json"
    metrics_path = out_dir / "metrics.json"
    topology_pdb = out_dir / "topology.pdb"
    prod_dcd = out_dir / "prod.dcd"
    prod_log = out_dir / "prod.log"

    try:
        # --- (1) receptor (cached, protonated, pocket-trimmed) -----------
        rec_pdb = _resolve_receptor(target)

        # --- (2) ligand prep ----------------------------------------------
        try:
            omol, _ = _prepare_ligand(sdf_path)
        except Exception as exc:
            res.status = "ligand_charge_failed"
            res.note = f"{type(exc).__name__}: {exc}"
            return res

        # --- (3) build solvated complex directly, no parmed/sander -------
        # Earlier versions used parmed → sander on the unsolvated complex,
        # then re-solvated. That route hit two failure modes:
        #   (a) parmed.Structure.save(prmtop) crashed with
        #       'NoneType.used' on certain ligand bond types
        #       (parmed/_amberparm.py:1560), failing before sander ran.
        #   (b) Re-solvating the minimised complex creates fresh waters
        #       at sub-VDW distances around the now-relaxed ligand. The
        #       subsequent OpenMM minim could not unwind the clashes,
        #       producing post-min PE > 1e8 kJ/mol or NaN at warm-up.
        # Both modes are sidestepped by minimising the *solvated*
        # production system (with HBonds + HMR + rigid water) in stages
        # — rigid-ligand → rigid-protein → free — directly. Important:
        # do the staged minim under the *same* constraint set that will
        # run during dynamics. Earlier we minimised under
        # constraints=None and only switched to HBonds+HMR for stepping;
        # the unconstrained minim let H atoms drift off the ideal X-H
        # bond manifold, and applyConstraints + 0.1 fs warm-up still
        # NaN'd at ~step 2200 on TP53 (the projection back to the
        # constraint manifold creates near-singular forces). With HBonds
        # on during minim the manifold is preserved end-to-end.
        t0 = time.time()
        try:
            modeller, _system_unused, ff = _build_system(
                rec_pdb, omol, padding_ang, with_hbonds=False, solvate=True)
        except Exception as exc:
            res.status = "system_build_failed"
            res.note = f"{type(exc).__name__}: {exc}"
            return res

        # Build the production system (HBonds + HMR) — also used for
        # staged minim so constraint manifold is consistent.
        from openmm.app import PME, HBonds
        system = ff.createSystem(modeller.topology,
                                 nonbondedMethod=PME,
                                 nonbondedCutoff=1.0 * u.nanometer,
                                 constraints=HBonds,
                                 hydrogenMass=1.5 * u.amu)

        # Stage A: rigid ligand + rigid protein → relax waters/ions only.
        # Stage B: rigid protein → ligand can relax in pocket.
        # Stage C: free minim under HBonds+HMR (production system).
        try:
            relaxed_positions = _staged_openmm_minimise(
                modeller, system, system, gpu_id,
                generator, target, chain_idx)
            modeller.positions = _as_vec3_positions_nm(relaxed_positions)
        except Exception as exc:
            res.status = "equilibration_nan"
            res.note = (f"staged_min: {type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()[-1000:]}")
            return res

        n_atoms = modeller.topology.getNumAtoms()
        n_water = sum(1 for r in modeller.topology.residues()
                      if r.name in ("HOH", "WAT"))
        res.n_atoms = n_atoms
        res.n_water = n_water
        logger.info("[%s/%s/%d] solvated system: %d atoms (%d waters)",
                    generator, target, chain_idx, n_atoms, n_water)

        # --- (4) restraint + integrator -----------------------------------
        rest_force, n_rest = _add_protein_heavy_restraint(
            system, modeller, k_kj_mol_nm2=1000.0)

        integrator = LangevinMiddleIntegrator(
            300 * u.kelvin, 1.0 / u.picosecond, 4.0 * u.femtosecond)

        # Pick a platform that actually works on this driver. The local
        # 3090 + driver 535 cannot load OpenMM 8.5.1's CUDA PTX
        # (compiled against CUDA 12.9, but driver supports up to 12.4),
        # so we try CUDA first and fall back to OpenCL → CPU.
        sim = None
        last_exc = None
        platform_order = (("CPU", {}),) if gpu_id < 0 else (
            ("CUDA",   {"Precision": "mixed", "DeviceIndex": str(gpu_id)}),
            ("OpenCL", {"Precision": "mixed", "DeviceIndex": str(gpu_id)}),
            ("CPU",    {}),
        )
        for plat_name, props in platform_order:
            try:
                platform = Platform.getPlatformByName(plat_name)
            except Exception as exc:
                last_exc = exc
                continue
            try:
                sim = Simulation(modeller.topology, system, integrator,
                                 platform, props)
                sim.context.setPositions(modeller.positions)
                logger.info("[%s/%s/%d] using platform %s",
                            generator, target, chain_idx, plat_name)
                break
            except Exception as exc:
                last_exc = exc
                logger.warning("platform %s unavailable: %s", plat_name, exc)
                # Re-create the integrator — it gets bound to a context
                # on Simulation construction and cannot be reused.
                integrator = LangevinMiddleIntegrator(
                    300 * u.kelvin, 1.0 / u.picosecond, 4.0 * u.femtosecond)
        if sim is None:
            raise RuntimeError(f"no usable OpenMM platform: {last_exc}")

        # Brief OpenMM minim to relax under HBonds + nonbonded settings
        # that match what production will use.
        sim.minimizeEnergy(
            tolerance=1 * u.kilojoule_per_mole / u.nanometer,
            maxIterations=20_000)

        import numpy as _np
        st = sim.context.getState(getEnergy=True, getForces=True)
        pe = st.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        forces = st.getForces(asNumpy=True).value_in_unit(
            u.kilojoule_per_mole / u.nanometer)
        max_force = float(_np.linalg.norm(forces, axis=1).max())
        logger.info("[%s/%s/%d] post-min PE = %.3e kJ/mol max|F|=%.2e kJ/mol/nm",
                    generator, target, chain_idx, pe, max_force)
        if pe != pe or pe > 1e7:
            res.status = "equilibration_nan"
            res.note = f"post-min PE not physical: {pe}"
            return res

        # Save the minimised state as topology.pdb (with box vectors set).
        state = sim.context.getState(getPositions=True,
                                     enforcePeriodicBox=True)
        with open(topology_pdb, "w") as fh:
            PDBFile.writeFile(modeller.topology, state.getPositions(), fh,
                              keepIds=True)

        # --- (6) gentle warm-up ramping T and dt --------------------------
        # Apply HBond constraints first so SHAKE/SETTLE has a clean
        # starting manifold. The schedule starts at 0.5 fs / 10 K — sub-
        # 0.5-fs steps on OpenCL mixed precision (the only platform
        # available on this CUDA-broken host) lose enough numerical
        # precision in the constraint solver that the integrator NaN's
        # within ~2200 steps even on a freshly minimised system. 0.5 fs
        # is the smallest step that's numerically robust here.
        sim.context.applyConstraints(1e-6)
        sim.context.computeVirtualSites()
        sim.context.setVelocitiesToTemperature(10 * u.kelvin)
        warm_integrator = sim.integrator
        warm_schedule = [
            (10,  0.50, 5_000, "0.5fs/10K"),    # 2.5 ps
            (50,  0.50, 5_000, "0.5fs/50K"),    # 2.5 ps
            (100, 0.50, 5_000, "0.5fs/100K"),   # 2.5 ps
            (200, 1.00, 5_000, "1fs/200K"),     # 5 ps
            (300, 1.00, 5_000, "1fs/300K"),     # 5 ps
            (300, 2.00, 5_000, "2fs/300K"),     # 10 ps
        ]
        for T, step_fs, n_steps, tag in warm_schedule:
            warm_integrator.setTemperature(T * u.kelvin)
            warm_integrator.setStepSize(step_fs * u.femtosecond)
            try:
                sim.step(n_steps)
            except Exception as exc:
                res.status = "equilibration_nan"
                res.note = f"warm {tag}: {type(exc).__name__}: {exc}"
                return res
            chk = sim.context.getState(getEnergy=True).getPotentialEnergy(
                ).value_in_unit(u.kilojoule_per_mole)
            logger.info("[%s/%s/%d] warm %s done PE=%.3e",
                        generator, target, chain_idx, tag, chk)
            if chk != chk:
                res.status = "equilibration_nan"
                res.note = f"NaN during warm-up stage {tag}"
                return res

        # Switch to 4 fs HMR for the rest of equil + production.
        warm_integrator.setStepSize(4.0 * u.femtosecond)

        equil_steps = max(1, int(round(equil_ps * 1000.0 / 4.0)))

        # --- (7) NVT equilibration with restraint at 4 fs -----------------
        sim.step(equil_steps)

        # --- (8) NPT equilibration unrestrained ---------------------------
        sim.context.setParameter("k", 0.0)
        system.addForce(MonteCarloBarostat(1 * u.bar, 300 * u.kelvin, 25))
        sim.context.reinitialize(preserveState=True)
        sim.step(equil_steps)
        res.wall_seconds_equil = float(time.time() - t0)
        logger.info("[%s/%s/%d] equil done in %.0fs",
                    generator, target, chain_idx, res.wall_seconds_equil)

        if equil_only:
            res.status = "ok"
            res.note = "equil_only"
            return res

        # --- (8) Production -----------------------------------------------
        n_steps = int(round(duration_ns * 1e6 / 4))  # 4 fs steps
        report_every = 2500  # 10 ps
        sim.reporters = []
        sim.reporters.append(DCDReporter(str(prod_dcd), report_every))
        sim.reporters.append(StateDataReporter(
            str(prod_log), report_every,
            step=True, time=True, potentialEnergy=True,
            kineticEnergy=True, temperature=True, volume=True,
            density=True, speed=True, separator="\t"))
        t1 = time.time()
        try:
            sim.step(n_steps)
        except Exception as exc:
            res.status = "production_nan"
            res.note = f"prod: {type(exc).__name__}: {exc}"
            return res
        res.wall_seconds_prod = float(time.time() - t1)
        logger.info("[%s/%s/%d] prod %.1f ns done in %.0fs (%.1f ns/day)",
                    generator, target, chain_idx, duration_ns,
                    res.wall_seconds_prod,
                    duration_ns / (res.wall_seconds_prod / 86400.0))

        # --- (9) analysis -------------------------------------------------
        try:
            metrics = _analyse_trajectory(topology_pdb, prod_dcd,
                                          ("UNK", "UNL", "LIG"))
        except Exception as exc:
            res.status = "analysis_failed"
            res.note = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
            return res

        res.rmsd_mean = metrics["rmsd_mean"]
        res.rmsd_std = metrics["rmsd_std"]
        res.rmsd_max = metrics["rmsd_max"]
        res.frac_below_3A = metrics["frac_below_3A"]
        res.n_persistent_contacts = metrics["n_persistent_contacts"]
        res.contact_persistence = metrics["contact_persistence"]
        res.status = "ok"

        metrics_path.write_text(json.dumps({
            **metrics,
            "smiles": smiles,
            "vina_score": vina_score,
            "duration_ns": duration_ns,
            "n_atoms": n_atoms,
            "n_water": n_water,
        }, indent=2))
        return res

    except Exception as exc:
        res.status = "exception"
        res.note = (f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()[-1500:]}")
        return res

    finally:
        status_path.write_text(json.dumps(asdict(res), indent=2))


# ----------------------------------------------------------------- main CLI --


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--generator", required=True, choices=ALL_GENERATORS)
    p.add_argument("--target", required=True, choices=ALL_TARGETS)
    p.add_argument("--chain_idx", type=int, default=None,
                   help="Specific chain_idx; if omitted, run top_k by Vina score.")
    p.add_argument("--top_k", type=int, default=3,
                   help="Number of top-scored poses to run (used when "
                        "--chain_idx not supplied).")
    p.add_argument("--duration_ns", type=float, default=20.0,
                   help="Production duration in ns (case studies use 50).")
    p.add_argument("--equil_ps", type=float, default=100.0,
                   help="NVT and NPT equilibration length in ps each.")
    p.add_argument("--padding_ang", type=float, default=10.0,
                   help="Solvent padding; reduce to 8 for OOM mitigation.")
    p.add_argument("--gpu_id", type=int, default=0,
                   help="GPU index. Use -1 to force CPU-only OpenMM.")
    p.add_argument("--out_root", type=Path, default=OUT_ROOT_DEFAULT)
    p.add_argument("--equil_only", action="store_true",
                   help="Stop after equilibration (smoke-test mode).")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if status.json exists with status=ok.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    if args.chain_idx is not None:
        # Single ligand requested — look up smiles / vina_score in manifest.
        import pandas as pd
        manifest = POSES_ROOT / args.generator / args.target / "manifest.parquet"
        df = pd.read_parquet(manifest)
        row = df[df["chain_idx"] == args.chain_idx]
        if row.empty:
            logger.error("chain_idx %d not in %s", args.chain_idx, manifest)
            return 1
        sdf = (POSES_ROOT / args.generator / args.target /
               f"{args.chain_idx}.sdf")
        if not sdf.exists():
            logger.error("missing sdf %s", sdf)
            return 1
        jobs = [{
            "chain_idx": args.chain_idx,
            "smiles": str(row.iloc[0]["smiles"]),
            "vina_score": float(row.iloc[0]["vina_score"]),
            "sdf_path": sdf,
        }]
    else:
        jobs = _load_top_k_jobs(args.generator, args.target, args.top_k)

    logger.info("MD on %s/%s — %d ligand(s), %.1f ns each",
                args.generator, args.target, len(jobs), args.duration_ns)

    n_ok = 0
    n_fail = 0
    for j in jobs:
        out_dir = args.out_root / args.generator / args.target / str(j["chain_idx"])
        status_path = out_dir / "status.json"
        if (not args.force) and status_path.exists():
            try:
                cached = json.loads(status_path.read_text())
                cached_is_equil_only = cached.get("note") == "equil_only"
                cache_satisfies_request = (
                    cached.get("status") == "ok"
                    and (args.equil_only or not cached_is_equil_only)
                )
                if cache_satisfies_request:
                    logger.info("[%s/%s/%d] cached ok — skip",
                                args.generator, args.target, j["chain_idx"])
                    n_ok += 1
                    continue
            except Exception:
                pass

        r = run_one(generator=args.generator, target=args.target,
                    chain_idx=j["chain_idx"],
                    sdf_path=j["sdf_path"],
                    smiles=j["smiles"], vina_score=j["vina_score"],
                    out_dir=out_dir,
                    duration_ns=args.duration_ns,
                    equil_only=args.equil_only,
                    padding_ang=args.padding_ang,
                    equil_ps=args.equil_ps,
                    gpu_id=args.gpu_id)
        if r.status == "ok":
            n_ok += 1
            logger.info("[%s/%s/%d] OK rmsd_mean=%.2f frac<3Å=%.2f n_persist=%s",
                        args.generator, args.target, j["chain_idx"],
                        r.rmsd_mean if r.rmsd_mean is not None else float("nan"),
                        r.frac_below_3A if r.frac_below_3A is not None else float("nan"),
                        r.n_persistent_contacts)
        else:
            n_fail += 1
            logger.error("[%s/%s/%d] FAIL %s — %s",
                         args.generator, args.target, j["chain_idx"],
                         r.status, r.note[:200])

    logger.info("done %s/%s: ok=%d fail=%d", args.generator, args.target,
                n_ok, n_fail)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
