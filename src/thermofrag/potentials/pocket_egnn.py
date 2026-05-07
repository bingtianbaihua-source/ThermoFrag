"""EGNN-over-Cα pocket encoder for TF-pocket v3.

TF-pocket v1 used a frozen ESM-2 mean-pooled residue embedding; v2 added
a V^pocket(m, p) coupling term on top. Both preserved v1's opaque
sequence-only pocket signal. v3 swaps the encoder for a small trainable
EGNN over pocket Cα atoms so the pocket representation is geometric
rather than sequence-only, and learned end-to-end with the μ head.

The encoder takes padded batches

    coords   (B, N, 3)    Cα positions, any frame (pocket is centered internally)
    aa_idx   (B, N)       residue type in the 20-letter alphabet used across
                          this codebase (``TARGETDIFF_AA_ORDER``, index 20 = "X")
    mask     (B, N) bool  True on valid residues, False on padding

and returns a fixed-dimensional SE(3)-invariant pocket vector of shape
``(B, embed_dim)`` that drops into the existing
``PocketConditionalChemicalPotentialHead`` without any other change to
the training stack.

Residue Cα geometry is cached offline as ``.npz`` via
``scripts/precompute_pocket_geometry.py``. At train time
``CrossDockedPocketGeomDataset`` (in ``src/thermofrag/data/pocket_geom.py``)
loads the cache; at sampler time we dump a per-target ``.npy`` of the
trained EGNN's output, so the sampler still consumes a single pocket
vector per LIT-PCBA target and ``scripts/sample.py`` needs no changes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

# Reuse the 20-letter alphabet used everywhere else in the codebase.
# TARGETDIFF_AA_ORDER[idx] is the 1-letter code for aa_type == idx; idx 20
# is reserved for unknown/non-standard residues ('X').
from .pocket_encoder import TARGETDIFF_AA_ORDER, _AA3_TO_1

_AA_LETTER_TO_IDX = {c: i for i, c in enumerate(TARGETDIFF_AA_ORDER)}
_AA_LETTER_TO_IDX["X"] = 20  # unknown / non-standard
N_AA_TYPES = 21


def aa_letter_to_idx(letter: str) -> int:
    return _AA_LETTER_TO_IDX.get(letter, 20)


def geometry_from_preprocessed_record(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract Cα (coords, aa_idx) from a TargetDiff-preprocessed CrossDocked record.

    Returns
    -------
    coords : (N_ca, 3) float32
    aa_idx : (N_ca,)   int64, values in [0, 20]; 20 marks unknown.
    """
    names = rec["protein_atom_name"]
    pos = rec["protein_pos"]
    if hasattr(pos, "cpu"):
        pos = pos.cpu().numpy()
    pos = np.asarray(pos, dtype=np.float32)
    aa = rec["protein_atom_to_aa_type"]
    if hasattr(aa, "cpu"):
        aa = aa.cpu().numpy()
    aa = np.asarray(aa, dtype=np.int64)
    ca_mask = np.array([n == "CA" for n in names], dtype=bool)
    coords = pos[ca_mask].astype(np.float32, copy=False)
    # In upstream, aa index runs 0..19 matching TARGETDIFF_AA_ORDER. Clamp
    # anything out of range (shouldn't happen, but belt-and-braces) to 20 / 'X'.
    aa_sel = aa[ca_mask]
    aa_sel = np.where((aa_sel >= 0) & (aa_sel < 20), aa_sel, 20).astype(np.int64)
    return coords, aa_sel


def extract_pocket_ca_geometry(
    receptor_pdb: str | Path,
    ligand_sdf: str | Path,
    cutoff_a: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (coords, aa_idx) for every residue whose Cα is within
    ``cutoff_a`` Å of any ligand heavy atom (matches TF-pocket v1's pocket10
    definition, measured from the Cα rather than any heavy atom so the output
    matches TargetDiff's pocket10 LMDB exactly on the 15 LIT-PCBA targets).
    """
    from Bio.PDB import PDBParser
    from rdkit import Chem

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", str(receptor_pdb))

    lig_path = Path(ligand_sdf)
    if lig_path.suffix.lower() == ".pdb":
        mol = Chem.MolFromPDBFile(str(lig_path), removeHs=True, sanitize=False)
    else:
        suppl = Chem.SDMolSupplier(str(lig_path), removeHs=True, sanitize=False)
        mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"no molecule in {lig_path}")
    conf = mol.GetConformer()
    lig_coords = np.asarray(
        [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumHeavyAtoms())],
        dtype=np.float32,
    )

    cut2 = float(cutoff_a) ** 2
    coords_out: list[np.ndarray] = []
    aa_out: list[int] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].strip() != "":  # hetero / water
                    continue
                ca = None
                for a in residue:
                    if a.element != "H":
                        pass
                    if a.get_name() == "CA":
                        ca = a
                        break
                if ca is None:
                    continue
                d2 = ((np.asarray(ca.coord, dtype=np.float32) - lig_coords) ** 2).sum(axis=-1)
                if d2.min() > cut2:
                    continue
                letter = _AA3_TO_1.get(residue.resname.upper(), "X")
                coords_out.append(np.asarray(ca.coord, dtype=np.float32))
                aa_out.append(aa_letter_to_idx(letter))
        break  # first model only
    if not coords_out:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64))
    return np.stack(coords_out, axis=0), np.asarray(aa_out, dtype=np.int64)


def pocket_geom_id(coords: np.ndarray, aa_idx: np.ndarray) -> str:
    """Stable short id keyed on Cα sequence (aa letters only).

    Matches the v1 ``pocket_id_from_sequence`` contract: two CrossDocked
    records with the same residue set share an id and therefore share a
    cached geometry (coordinates may differ slightly between poses, but
    since we center the pocket in the encoder and the Cα neighborhood is
    locally rigid, the signal is dominated by sequence + topology — the
    tiny pose-to-pose coordinate wobble is treated as noise).
    """
    letters = "".join(TARGETDIFF_AA_ORDER[i] if i < 20 else "X" for i in aa_idx.tolist())
    return hashlib.sha1(letters.encode()).hexdigest()[:16]


def save_pocket_geom(path: str | Path, coords: np.ndarray, aa_idx: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        coords=coords.astype(np.float32, copy=False),
        aa_idx=aa_idx.astype(np.int64, copy=False),
    )


def load_pocket_geom(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(str(path))
    return z["coords"].astype(np.float32, copy=False), z["aa_idx"].astype(np.int64, copy=False)


class EGNNPocketEncoder(nn.Module):
    """Small dense EGNN-style encoder over pocket Cα atoms.

    Design choices
    --------------
    * All-pairs dense messaging with a 10 Å cutoff mask. Pockets are small
      (≤ ~120 residues within 10 Å of a drug-like ligand) so the (N, N)
      matrix never blows up; avoiding ``torch_scatter`` / ``knn_graph``
      also means no Python-level sort-by-distance inside the model.
    * SE(3) invariance: node features depend only on pairwise distances
      (Gaussian-smeared), never on raw coordinates. We center per-sample so
      numeric scale is consistent across pockets, but the readout never
      reads absolute coordinates.
    * Output: one vector of size ``embed_dim`` per pocket (mean pooling
      over valid residues, then a final Linear).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 3,
        n_rbf: int = 16,
        cutoff_a: float = 10.0,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_layers = int(n_layers)
        self.n_rbf = int(n_rbf)
        self.cutoff_a = float(cutoff_a)

        self.aa_embed = nn.Embedding(N_AA_TYPES, embed_dim)
        # Gaussian RBF basis over distances in [0, cutoff_a].
        centers = torch.linspace(0.0, cutoff_a, n_rbf)
        widths = torch.full((n_rbf,), float(cutoff_a) / max(n_rbf - 1, 1))
        self.register_buffer("rbf_centers", centers)
        self.register_buffer("rbf_widths", widths)

        self.msg_mlps = nn.ModuleList()
        self.upd_mlps = nn.ModuleList()
        for _ in range(self.n_layers):
            self.msg_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * embed_dim + n_rbf, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                    nn.SiLU(),
                )
            )
            self.upd_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * embed_dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim),
                )
            )
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _rbf(self, d: torch.Tensor) -> torch.Tensor:
        # d: (..., 1) or (...); returns (..., n_rbf)
        c = self.rbf_centers
        w = self.rbf_widths
        return torch.exp(-((d.unsqueeze(-1) - c) ** 2) / (2.0 * w * w + 1e-12))

    def forward(
        self,
        coords: torch.Tensor,
        aa_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """coords (B, N, 3), aa_idx (B, N) long, mask (B, N) bool.

        Returns (B, embed_dim).
        """
        if coords.dim() != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must be (B, N, 3), got {tuple(coords.shape)}")
        B, N = coords.shape[0], coords.shape[1]
        mask_f = mask.to(coords.dtype)  # (B, N)

        # Per-sample center-of-mass of valid residues, to stabilize scale.
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
        com = (coords * mask_f.unsqueeze(-1)).sum(dim=1) / denom  # (B, 3)
        x = coords - com.unsqueeze(1)  # (B, N, 3)

        # Pairwise distances (B, N, N) and RBF features (B, N, N, n_rbf).
        diff = x.unsqueeze(2) - x.unsqueeze(1)  # (B, N, N, 3)
        d = torch.linalg.norm(diff, dim=-1).clamp_min(1e-6)  # (B, N, N)
        rbf = self._rbf(d)  # (B, N, N, n_rbf)

        # Pair mask: valid-valid, not self, within cutoff.
        m2 = mask.unsqueeze(-1) & mask.unsqueeze(-2)  # (B, N, N)
        eye = torch.eye(N, device=coords.device, dtype=torch.bool).unsqueeze(0)
        m2 = m2 & ~eye & (d < self.cutoff_a)
        m2_f = m2.to(coords.dtype).unsqueeze(-1)  # (B, N, N, 1)

        # Node features
        h = self.aa_embed(aa_idx.clamp(min=0, max=N_AA_TYPES - 1))  # (B, N, D)
        h = h * mask_f.unsqueeze(-1)

        for msg, upd in zip(self.msg_mlps, self.upd_mlps):
            hi = h.unsqueeze(2).expand(B, N, N, self.embed_dim)  # dst (row)
            hj = h.unsqueeze(1).expand(B, N, N, self.embed_dim)  # src (col)
            pair = torch.cat([hi, hj, rbf], dim=-1)               # (B, N, N, 2D+nrbf)
            mij = msg(pair)                                        # (B, N, N, D)
            mij = mij * m2_f
            mi = mij.sum(dim=2)                                    # (B, N, D) sum over src
            h = h + upd(torch.cat([h, mi], dim=-1))
            h = h * mask_f.unsqueeze(-1)

        # Mean pool valid residues, project to output.
        pooled = h.sum(dim=1) / denom  # (B, D)
        return self.out_proj(pooled)

    @torch.no_grad()
    def encode_single(self, coords: np.ndarray | torch.Tensor, aa_idx: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Convenience helper for one pocket; returns a 1-D tensor of size ``embed_dim``."""
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords.astype(np.float32, copy=False))
        if isinstance(aa_idx, np.ndarray):
            aa_idx = torch.from_numpy(aa_idx.astype(np.int64, copy=False))
        device = next(self.parameters()).device
        coords = coords.to(device).unsqueeze(0)
        aa_idx = aa_idx.to(device).unsqueeze(0).long()
        mask = torch.ones(aa_idx.shape, dtype=torch.bool, device=device)
        out = self.forward(coords, aa_idx, mask)
        return out.squeeze(0).detach()


def collate_pocket_geoms(
    geoms: Iterable[tuple[np.ndarray, np.ndarray]],
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of (coords, aa_idx) pairs into batched (coords, aa_idx, mask).

    Out shapes: coords (B, N_max, 3), aa_idx (B, N_max), mask (B, N_max) bool.
    Empty pockets are handled by padding to N_max >= 1 and mask=False.
    """
    gs = list(geoms)
    if not gs:
        return (
            torch.zeros(0, 0, 3),
            torch.zeros(0, 0, dtype=torch.long),
            torch.zeros(0, 0, dtype=torch.bool),
        )
    lens = [int(g[0].shape[0]) for g in gs]
    N_max = max(max(lens), 1)
    B = len(gs)
    coords = np.zeros((B, N_max, 3), dtype=np.float32)
    aa = np.full((B, N_max), 20, dtype=np.int64)
    mask = np.zeros((B, N_max), dtype=bool)
    for i, ((c, a), n) in enumerate(zip(gs, lens)):
        if n > 0:
            coords[i, :n] = c
            aa[i, :n] = a
            mask[i, :n] = True
    coords_t = torch.from_numpy(coords)
    aa_t = torch.from_numpy(aa)
    mask_t = torch.from_numpy(mask)
    if device is not None:
        coords_t = coords_t.to(device)
        aa_t = aa_t.to(device)
        mask_t = mask_t.to(device)
    return coords_t, aa_t, mask_t
