"""QMugs QM dataset loader — held-out benchmark for C1.

QMugs (Isert et al 2022, https://doi.org/10.1038/s41597-022-01390-7) is
distributed as one SDF file per conformer, organised under a directory of
ChEMBL-IDed sub-folders (``structures/CHEMBL<id>/conf_00.sdf`` etc). Each SDF
carries single-point QM energies as SDF properties; we read the DFT total
energy and the 3D coordinates via RDKit.

Relevant SDF property keys (QMugs v1.1):
  - ``DFT:TOTAL_ENERGY``   (Hartree, reference level ωB97X-D/def2-SVP)
  - ``GFN2:TOTAL_ENERGY``  (Hartree, semi-empirical, cheaper)

QMugs does **not** publish forces; the DFT calculation is single-point only,
so the shard stores energies and coordinates but omits force arrays. The
associated Dataset yields PyG ``Data`` with ``energy`` but no ``forces``.

Each processed shard is an .npz with concatenated arrays:
    z       [N_atoms_total]         atomic numbers
    pos     [N_atoms_total, 3]      angstrom
    energy  [N_mols]                kcal/mol (total DFT energy)
    ptr     [N_mols + 1]            cumulative atom indices
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


HARTREE_TO_KCAL = 627.5094740631


# Atomic-number set for the drug-like subset (H,C,N,O,F,P,S,Cl,Br per docs/DATA.md).
# QMugs spans a broader element range than SPICE drug-like, so we re-filter here.
DRUGLIKE_Z = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35})


@dataclass
class QMugsConformer:
    z: np.ndarray         # [N] atomic numbers
    pos: np.ndarray       # [N, 3] angstrom
    energy: float         # kcal/mol (DFT total)
    smiles: str
    source_path: str


def iter_qmugs_sdf_dir(
    root: str | Path,
    energy_key: str = "DFT:TOTAL_ENERGY",
    elements_allowed: frozenset[int] = DRUGLIKE_Z,
    max_heavy_atoms: int = 50,
    sanitize: bool = False,
) -> Iterator[QMugsConformer]:
    """Stream QMugs conformers from a directory tree of SDF files.

    The walk is recursive; any file with suffix ``.sdf`` is parsed. Molecules
    failing the element or heavy-atom filter are silently skipped. The energy
    property is parsed as a float Hartree and converted to kcal/mol.
    """
    from rdkit import Chem

    root = Path(root)
    for sdf in sorted(root.rglob("*.sdf")):
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=sanitize)
        for mol in supplier:
            if mol is None:
                continue
            try:
                z = np.fromiter(
                    (a.GetAtomicNum() for a in mol.GetAtoms()), dtype=np.int64
                )
                if not set(int(zi) for zi in z).issubset(elements_allowed):
                    continue
                if int((z != 1).sum()) > max_heavy_atoms:
                    continue
                pos = mol.GetConformer(0).GetPositions().astype(np.float32)  # Å
                props = mol.GetPropsAsDict()
                if energy_key not in props:
                    continue
                energy_h = float(props[energy_key])
                smiles = props.get("SMILES") or props.get("smiles") or Chem.MolToSmiles(mol)
                yield QMugsConformer(
                    z=z,
                    pos=pos,
                    energy=energy_h * HARTREE_TO_KCAL,
                    smiles=str(smiles),
                    source_path=str(sdf),
                )
            except Exception:
                # Skip silently: preprocessors log aggregate failure counts.
                continue


def save_qmugs_shard(conformers: list[QMugsConformer], out_path: Path) -> None:
    """Concat conformers into a single NPZ shard (no forces array)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z_list = [c.z for c in conformers]
    pos_list = [c.pos for c in conformers]
    e_list = [c.energy for c in conformers]
    sizes = np.array([z.shape[0] for z in z_list], dtype=np.int64)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    np.savez_compressed(
        out_path,
        z=np.concatenate(z_list).astype(np.int8),
        pos=np.concatenate(pos_list).astype(np.float32),
        energy=np.asarray(e_list, dtype=np.float32),
        ptr=ptr.astype(np.int64),
    )


class QMugsShard(Dataset):
    """Random-access Dataset over pre-sharded QMugs NPZs.

    Mirrors :class:`thermofrag.data.spice.SPICEShard` so PyG DataLoader /
    ``Trainer._eval_qm`` can consume both interchangeably. The returned Data
    has no ``forces`` attribute (QMugs is single-point), so force-regression
    code paths must guard on ``hasattr(batch, 'forces')``.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        energy_mean: float = 0.0,
        energy_std: float = 1.0,
    ):
        self.shard_dir = Path(shard_dir)
        self.shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No QMugs shards under {shard_dir}")
        self._opened = [np.load(s, mmap_mode="r") for s in self.shards]
        self._sizes = [int(o["energy"].shape[0]) for o in self._opened]
        self._cum = np.concatenate([[0], np.cumsum(self._sizes)])
        self.energy_mean = float(energy_mean)
        self.energy_std = float(energy_std)

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_i = int(np.searchsorted(self._cum, idx, side="right") - 1)
        local = idx - self._cum[shard_i]
        o = self._opened[shard_i]
        ptr = o["ptr"]
        a0, a1 = int(ptr[local]), int(ptr[local + 1])
        z = torch.from_numpy(np.asarray(o["z"][a0:a1]).astype(np.int64))
        pos = torch.from_numpy(np.asarray(o["pos"][a0:a1]).astype(np.float32))
        energy = float(o["energy"][local])
        energy_norm = (energy - self.energy_mean) / self.energy_std
        return Data(
            z=z,
            pos=pos,
            energy=torch.tensor(energy, dtype=torch.float32),
            energy_norm=torch.tensor(energy_norm, dtype=torch.float32),
            num_nodes=z.shape[0],
        )
