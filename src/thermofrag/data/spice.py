"""SPICE QM dataset loader.

SPICE HDF5 schema (one HDF5 group per molecule):
    atomic_numbers          [N_atoms] int
    conformations           [N_conf, N_atoms, 3] float, bohr
    dft_total_energy        [N_conf] float, hartree
    dft_total_gradient      [N_conf, N_atoms, 3] float, hartree/bohr
    formation_energy        [N_conf] float, hartree (atom-centered total energy)
    subset                  bytes
    smiles                  bytes

Each processed shard is an .npz with concatenated arrays:
    z       [N_atoms_total]         atomic numbers
    pos     [N_atoms_total, 3]      angstrom
    energy  [N_mols]                kcal/mol, atom-centered (formation energy)
    forces  [N_atoms_total, 3]      kcal/mol/angstrom
    ptr     [N_mols + 1]            cumulative atom indices

This is the common torch_geometric per-graph-concatenated layout.
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
BOHR_TO_ANGSTROM = 0.529177210903
# gradient hartree/bohr -> force kcal/mol/A (sign flipped because F = -grad E)
GRAD_HARTREE_BOHR_TO_FORCE_KCAL_ANG = -HARTREE_TO_KCAL / BOHR_TO_ANGSTROM


# Atomic-number set for the drug-like subset (H,C,N,O,F,P,S,Cl,Br per docs/DATA.md).
DRUGLIKE_Z = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35})


@dataclass
class SPICEConformer:
    z: np.ndarray         # [N] atomic numbers
    pos: np.ndarray       # [N, 3] angstrom
    energy: float         # kcal/mol (formation energy)
    forces: np.ndarray    # [N, 3] kcal/mol/A
    smiles: str
    subset: str


def iter_spice_hdf5(
    h5_path: str | Path,
    elements_allowed: frozenset[int] = DRUGLIKE_Z,
    max_heavy_atoms: int = 50,
    use_formation_energy: bool = True,
) -> Iterator[SPICEConformer]:
    """Stream SPICE conformers that pass the drug-like filter.

    A molecule is kept iff every atom is in ``elements_allowed`` and the number
    of heavy atoms (non-H) is at most ``max_heavy_atoms``. Every conformer of a
    kept molecule is yielded.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        for mol_name in f.keys():
            g = f[mol_name]
            z = g["atomic_numbers"][:].astype(np.int64)
            if not set(int(zi) for zi in z).issubset(elements_allowed):
                continue
            n_heavy = int((z != 1).sum())
            if n_heavy > max_heavy_atoms:
                continue
            conf = g["conformations"][:].astype(np.float32) * BOHR_TO_ANGSTROM  # [C, N, 3]
            grad = g["dft_total_gradient"][:].astype(np.float32)
            forces = grad * GRAD_HARTREE_BOHR_TO_FORCE_KCAL_ANG  # [C, N, 3]
            e_key = "formation_energy" if use_formation_energy else "dft_total_energy"
            energies = g[e_key][:].astype(np.float64) * HARTREE_TO_KCAL  # [C]
            smiles = _decode_bytes(g["smiles"][()])
            subset = _decode_bytes(g["subset"][()]) if "subset" in g else ""

            for c in range(conf.shape[0]):
                yield SPICEConformer(
                    z=z,
                    pos=conf[c],
                    energy=float(energies[c]),
                    forces=forces[c],
                    smiles=smiles,
                    subset=subset,
                )


def _decode_bytes(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.ndarray):
        return x.item().decode("utf-8", errors="replace") if x.dtype.kind == "S" else str(x)
    return str(x)


def save_shard(conformers: list[SPICEConformer], out_path: Path) -> None:
    """Concat conformers into a single NPZ shard."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    z_list = [c.z for c in conformers]
    pos_list = [c.pos for c in conformers]
    f_list = [c.forces for c in conformers]
    e_list = [c.energy for c in conformers]
    sizes = np.array([z.shape[0] for z in z_list], dtype=np.int64)
    ptr = np.concatenate([[0], np.cumsum(sizes)])
    np.savez_compressed(
        out_path,
        z=np.concatenate(z_list).astype(np.int8),
        pos=np.concatenate(pos_list).astype(np.float32),
        energy=np.asarray(e_list, dtype=np.float32),
        forces=np.concatenate(f_list).astype(np.float32),
        ptr=ptr.astype(np.int64),
    )


class SPICEShard(Dataset):
    """Random-access Dataset over pre-sharded SPICE NPZs.

    Shards are opened with mmap_mode='r' so keeping all of them live is cheap.
    Optional ``energy_mean`` / ``energy_std`` are standardization stats computed
    once from the training split and written into the manifest.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        energy_mean: float = 0.0,
        energy_std: float = 1.0,
        force_std: float = 1.0,
    ):
        self.shard_dir = Path(shard_dir)
        self.shards = sorted(self.shard_dir.glob("shard_*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"No SPICE shards under {shard_dir}")
        # Read metadata once (sizes per shard) but do NOT keep mmap handles open
        # in the parent: multiple DataLoader workers forking from a shared mmap
        # can hit "BadZipFile: Bad magic number" because the zip central
        # directory is read inconsistently by concurrent worker processes.
        # Each worker opens its own handles lazily on first __getitem__.
        self._sizes = []
        for s in self.shards:
            with np.load(s, mmap_mode="r") as o:
                self._sizes.append(int(o["energy"].shape[0]))
        self._cum = np.concatenate([[0], np.cumsum(self._sizes)])
        self._opened: list = [None] * len(self.shards)  # per-process lazy cache
        self.energy_mean = float(energy_mean)
        self.energy_std = float(energy_std)
        self.force_std = float(force_std)

    def _get_shard(self, shard_i: int):
        o = self._opened[shard_i]
        if o is None:
            o = np.load(self.shards[shard_i], mmap_mode="r")
            self._opened[shard_i] = o
        return o

    def __len__(self) -> int:
        return int(self._cum[-1])

    def __getitem__(self, idx: int) -> Data:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        shard_i = int(np.searchsorted(self._cum, idx, side="right") - 1)
        local = idx - self._cum[shard_i]
        o = self._get_shard(shard_i)
        ptr = o["ptr"]
        a0, a1 = int(ptr[local]), int(ptr[local + 1])
        z = torch.from_numpy(np.asarray(o["z"][a0:a1]).astype(np.int64))
        pos = torch.from_numpy(np.asarray(o["pos"][a0:a1]).astype(np.float32))
        forces = torch.from_numpy(np.asarray(o["forces"][a0:a1]).astype(np.float32))
        energy = float(o["energy"][local])
        energy_norm = (energy - self.energy_mean) / self.energy_std
        return Data(
            z=z,
            pos=pos,
            forces=forces,
            forces_norm=forces / self.force_std,
            energy=torch.tensor(energy, dtype=torch.float32),
            energy_norm=torch.tensor(energy_norm, dtype=torch.float32),
            num_nodes=z.shape[0],
        )
