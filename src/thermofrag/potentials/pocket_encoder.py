"""Frozen pocket encoder for the TF-pocket variant.

Wraps ESM-2 (default: ``facebook/esm2_t33_650M_UR50D``) as the pocket
feature extractor. The model is loaded in inference mode; the encoder
never contributes gradients to the TF Hamiltonian. Pocket embeddings
are fixed-length (1280-d for esm2_t33) mean-pooled over the residues
within ``cutoff_a`` Å of the cognate ligand, matching TargetDiff's
10 Å pocket definition.

Typical use is offline precomputation — call ``encode_pocket`` once per
(receptor, cognate ligand) pair, cache the resulting vector to disk, and
load from LMDB during training. See ``scripts/precompute_pocket_embeddings.py``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


# Standard 3-letter → 1-letter amino acid codes. Unknown residues map to 'X'.
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
    "HID": "H", "HIE": "H", "HIP": "H",
    "CYX": "C", "CYM": "C",
}

# Order matches TargetDiff's PDBProtein.AA_NAME_SYM (vendor/targetdiff/utils/data.py):
# integer indices in preprocessed LMDB's protein_atom_to_aa_type map into this
# string — so position i here is the 1-letter code for aa_type == i.
TARGETDIFF_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

DEFAULT_ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
DEFAULT_ESM_DIM = 1280
DEFAULT_POCKET_CUTOFF_A = 10.0


def sequence_from_preprocessed_record(rec: dict) -> str:
    """Build the pocket's 1-letter residue sequence from a TargetDiff-preprocessed
    CrossDocked2020 LMDB record.

    Picks one residue per Cα atom (``protein_atom_name == 'CA'``) and decodes
    ``protein_atom_to_aa_type`` via ``TARGETDIFF_AA_ORDER``. Residue order
    follows atom order in the record, which is the pocket10 PDB order.
    """
    aa = rec["protein_atom_to_aa_type"]
    names = rec["protein_atom_name"]
    out = []
    for i, n in enumerate(names):
        if n == "CA":
            out.append(TARGETDIFF_AA_ORDER[int(aa[i])])
    return "".join(out)


def pocket_id_from_sequence(seq: str) -> str:
    """Stable short id keyed on the amino-acid sequence. Used to deduplicate
    pocket embeddings across poses that share a residue set: a 16-hex prefix
    of sha1(seq) is collision-free at our ~165k-pocket scale.
    """
    return hashlib.sha1(seq.encode()).hexdigest()[:16]


@dataclass
class PocketResidue:
    chain: str
    resseq: int
    name3: str  # 3-letter code

    @property
    def letter(self) -> str:
        return _AA3_TO_1.get(self.name3.upper(), "X")


def extract_pocket_residues(
    receptor_pdb: str | Path,
    ligand_sdf: str | Path,
    cutoff_a: float = DEFAULT_POCKET_CUTOFF_A,
) -> list[PocketResidue]:
    """Return the ordered list of residues with any heavy atom within
    ``cutoff_a`` Å of any heavy atom of the cognate ligand.

    Residue order follows PDB file order (chain, then resseq).
    """
    # Lazy imports keep this module importable without the full ML stack.
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
    selected: list[PocketResidue] = []
    seen = set()  # (chain, resseq, icode)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0].strip() != "":  # hetero / water
                    continue
                key = (chain.id, residue.id[1], residue.id[2])
                if key in seen:
                    continue
                # squared distance from any residue heavy atom to any ligand atom
                atom_coords = np.asarray(
                    [a.coord for a in residue if a.element != "H"], dtype=np.float32
                )
                if atom_coords.size == 0:
                    continue
                d2 = ((atom_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(axis=-1)
                if d2.min() <= cut2:
                    seen.add(key)
                    selected.append(
                        PocketResidue(chain=chain.id, resseq=residue.id[1], name3=residue.resname)
                    )
        break  # first model only
    return selected


def residues_to_sequence(residues: Iterable[PocketResidue]) -> str:
    return "".join(r.letter for r in residues)


class PocketEncoder(nn.Module):
    """Frozen ESM-2 pocket encoder producing a fixed-dim embedding per pocket.

    The underlying HuggingFace model is loaded lazily so importing this
    file is cheap. All parameters are frozen; ``forward`` runs under
    ``torch.no_grad`` internally.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ESM_MODEL,
        device: str | torch.device = "cpu",
        max_length: int = 1024,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = int(max_length)
        self._device = torch.device(device)
        self._tok = None
        self._model = None
        self._dim: int | None = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model.to(self._device)
        self._dim = int(model.config.hidden_size)

    @property
    def embed_dim(self) -> int:
        self._lazy_load()
        assert self._dim is not None
        return self._dim

    @torch.no_grad()
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Mean-pool ESM-2 residue embeddings (excluding CLS / EOS / pad).

        Returns a tensor of shape [embed_dim].
        """
        self._lazy_load()
        if not sequence:
            # Empty pocket shouldn't happen, but keep the call total.
            return torch.zeros(self.embed_dim, device=self._device)
        enc = self._tok(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state[0]  # [L, D]
        # Mask off CLS (position 0) and EOS (last attended position).
        mask = attention_mask[0].clone()
        mask[0] = 0
        eos_idx = int(attention_mask[0].sum().item()) - 1
        if 0 <= eos_idx < mask.shape[0]:
            mask[eos_idx] = 0
        m = mask.bool().unsqueeze(-1)
        resid = hidden.masked_select(m).view(-1, hidden.shape[-1])
        if resid.shape[0] == 0:
            return torch.zeros(self.embed_dim, device=self._device)
        return resid.mean(dim=0)

    @torch.no_grad()
    def encode_sequences_batch(self, sequences: list[str]) -> torch.Tensor:
        """Batched variant of ``encode_sequence``. Pads within the batch and
        returns a tensor of shape ``[B, embed_dim]``. CLS + EOS + pad positions
        are masked out of the mean, matching the single-sequence code path.
        """
        self._lazy_load()
        if not sequences:
            return torch.zeros(0, self.embed_dim, device=self._device)
        enc = self._tok(
            sequences,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
        )
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc["attention_mask"].to(self._device)
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, L, D]

        mask = attention_mask.clone()
        mask[:, 0] = 0                              # CLS
        last = attention_mask.sum(dim=1) - 1        # index of EOS per row
        mask[torch.arange(mask.shape[0], device=mask.device), last] = 0
        mask_f = mask.to(hidden.dtype).unsqueeze(-1)
        summed = (hidden * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom

    @torch.no_grad()
    def encode_pocket(
        self,
        receptor_pdb: str | Path,
        ligand_sdf: str | Path,
        cutoff_a: float = DEFAULT_POCKET_CUTOFF_A,
    ) -> torch.Tensor:
        residues = extract_pocket_residues(receptor_pdb, ligand_sdf, cutoff_a=cutoff_a)
        return self.encode_sequence(residues_to_sequence(residues))


def load_pocket_embed(path: str | Path) -> torch.Tensor:
    """Load a cached pocket embedding saved by the precompute script."""
    arr = np.load(str(path))
    return torch.from_numpy(arr.astype(np.float32))


def save_pocket_embed(path: str | Path, embed: torch.Tensor) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embed.detach().cpu().float().numpy())
