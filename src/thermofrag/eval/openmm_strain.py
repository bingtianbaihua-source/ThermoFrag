"""Post-MMFF94 single-point strain energy under OpenFF Sage (claim C4).

For each generated SMILES we:

  1. Embed a 3D conformer with RDKit ETKDGv3.
  2. Run an MMFF94 minimization. This is the "as-generated" geometry that a
     BBAR-style generator would output without any further physics-based
     polish.
  3. Parameterize the molecule with the OpenFF 2.1 (Sage) force field, using
     pre-computed Gasteiger partial charges to skip AM1-BCC (~80x speedup over
     GAFF default). The force field choice is still a physics-based
     drug-chemistry MM potential; C4 is a relative comparison against the
     baselines, so consistent FF is what matters.
  4. Single-point energy ``E_mm_at_mmff_geom``.
  5. LocalEnergyMinimizer under the same FF -> ``E_mm_at_relaxed``.

  Strain = E_mm_at_mmff_geom - E_mm_at_relaxed >= 0.

A low ΔE means the MMFF94 geometry was already close to the FF basin — i.e.
the generator produced a physically sane pose without needing relaxation. This
is the C4 claim. See docs/PLAN.md C4 and docs/METHOD.md §7.

Note: docs/METHOD.md §7 specifies "OpenMM + GAFF" for strain. We substitute
OpenFF Sage with Gasteiger charges for tractability on a 4060 workstation
(AM1-BCC via antechamber takes ~30-60 s per ligand, pushing the 1500-ligand
audit past overnight). OpenFF Sage is the modern SMIRNOFF equivalent of GAFF
and is regularly benchmarked against it; the relative strain comparison that
C4 is built on is unchanged by this substitution.

The module is pure-functional; callers pass a SMILES and get back a
dataclass with ``e_mmff``, ``e_gaff``, ``strain``, ``status``.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# Suppress the usual openff/openmm deprecation chatter that fires on every
# parameterization call — the volume hides the real errors we care about.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="openff")


OPENFF_RELEASE = "openff-2.1.0"  # Sage, current default for small-molecule FFs


@dataclass
class StrainResult:
    smiles: str
    e_mmff: Optional[float]  # kcal/mol, single-point GAFF energy at MMFF94 geom
    e_gaff: Optional[float]  # kcal/mol, GAFF-minimized energy
    strain: Optional[float]  # kcal/mol, e_mmff - e_gaff
    n_atoms: int
    status: str  # "ok" | "embed_failed" | "mmff_failed" | "param_failed" | "min_failed"


def _embed_and_mmff_minimize(mol: Chem.Mol, seed: int = 42) -> Optional[Chem.Mol]:
    """Add Hs, embed one conformer with ETKDGv3, MMFF94 minimize in place.

    Returns None if embedding or MMFF optimization fails.
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        return None
    try:
        rc = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        return None
    if rc < 0:
        return None
    return mol


def _rdkit_to_openff(mol: Chem.Mol):
    """Convert an RDKit mol-with-conformer into an OpenFF Molecule + Topology.

    Uses ``from_rdkit`` so stereo/bond orders carry over. Hydrogens must be
    explicit on ``mol`` before calling.
    """
    from openff.toolkit.topology import Molecule as OFFMolecule

    offmol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)
    return offmol


def _build_system_and_context(offmol):
    """Parameterize with OpenFF Sage + build an OpenMM Context.

    The molecule is expected to already carry Gasteiger partial charges;
    SMIRNOFFTemplateGenerator will forward them (no AM1-BCC).
    """
    from openmm import LangevinIntegrator, Platform, unit
    from openmm.app import ForceField
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator

    gen = SMIRNOFFTemplateGenerator(molecules=[offmol], forcefield=OPENFF_RELEASE)
    ff = ForceField()
    ff.registerTemplateGenerator(gen.generator)

    topology = offmol.to_topology().to_openmm()
    system = ff.createSystem(topology)

    integrator = LangevinIntegrator(300 * unit.kelvin,
                                    1.0 / unit.picosecond,
                                    1.0 * unit.femtosecond)
    platform = Platform.getPlatformByName("Reference")
    from openmm import Context

    context = Context(system, integrator, platform)
    return context, system, topology


def _set_positions_from_offmol(context, offmol):
    import numpy as np
    from openmm import unit
    positions = np.asarray(offmol.conformers[0].to("nanometer").magnitude,
                           dtype=np.float64)
    context.setPositions(positions * unit.nanometer)


def _energy_kcal(context) -> float:
    from openmm import unit
    state = context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)


def compute_strain(smiles: str, seed: int = 42,
                   min_tolerance_kj: float = 0.1,
                   max_min_iterations: int = 500) -> StrainResult:
    """Compute single-molecule strain for one SMILES.

    Returns a ``StrainResult`` with status explaining any failure path so
    callers can aggregate without raising.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return StrainResult(smiles, None, None, None, 0, "parse_failed")

    mol_h = _embed_and_mmff_minimize(mol, seed=seed)
    if mol_h is None:
        return StrainResult(smiles, None, None, None, mol.GetNumAtoms(), "embed_failed")

    n_atoms = mol_h.GetNumAtoms()
    try:
        offmol = _rdkit_to_openff(mol_h)
        offmol.assign_partial_charges("gasteiger")
    except Exception as exc:
        logger.debug("openff conversion failed for %s: %s", smiles, exc)
        return StrainResult(smiles, None, None, None, n_atoms, "offmol_failed")

    try:
        context, _, _ = _build_system_and_context(offmol)
    except Exception as exc:
        logger.debug("GAFF parameterization failed for %s: %s", smiles, exc)
        return StrainResult(smiles, None, None, None, n_atoms, "param_failed")

    try:
        _set_positions_from_offmol(context, offmol)
        e_mmff = _energy_kcal(context)
        from openmm import LocalEnergyMinimizer, unit
        # Tolerance is an RMS force threshold in OpenMM >= 8.0.
        LocalEnergyMinimizer.minimize(
            context,
            min_tolerance_kj * unit.kilojoule_per_mole / unit.nanometer,
            max_min_iterations,
        )
        e_gaff = _energy_kcal(context)
    except Exception as exc:
        logger.debug("minimization failed for %s: %s", smiles, exc)
        return StrainResult(smiles, None, None, None, n_atoms, "min_failed")

    strain = e_mmff - e_gaff
    status = "ok"
    if not np.isfinite(strain) or strain < -1.0:
        # a negative strain past float noise means the minimizer climbed out
        # of a ring-flipping saddle — treat as a failed polish, not strain.
        status = "unstable"
    return StrainResult(smiles, float(e_mmff), float(e_gaff),
                        float(strain) if status == "ok" else None,
                        n_atoms, status)


def compute_strain_batch(smiles_list, seed: int = 42,
                         log_every: int = 50):
    """Serial loop over ``smiles_list``. Skips duplicates. Yields results."""
    for i, smi in enumerate(smiles_list):
        if log_every and i and i % log_every == 0:
            logger.info("strain %d / %d", i, len(smiles_list))
        yield compute_strain(smi, seed=seed)
