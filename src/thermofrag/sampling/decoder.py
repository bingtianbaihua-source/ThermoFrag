"""Fragment-graph → SMILES decoder for Phase-4 sampled pools.

The Phase-4 conditional MH sampler (``conditional_mh.py``) produces pools of
fragment graphs that share topology with a BRICS-decomposed seed molecule but
differ in per-node frag_id assignments. This module converts those samples back
into RDKit Mol / SMILES for downstream docking.

MVP scope
---------
Handles exactly two cases per sample, gated by the seed's BRICS topology:

* No flips (sampled frag_id list == seed's frag_id list):
  return the canonicalized seed SMILES.

* Only leaf flips with compatible anchor count:
  every position where ``sampled != seed`` must correspond to a seed BRICS unit
  of degree 1 (leaf) whose replacement fragment has ``n_anchors_mode == 1`` in
  the library. Each leaf is detached from the shared scaffold, a single-dummy
  variant of the new core is built (by trying each heavy atom with free
  valence), and merged back using BBAR's ``merge`` primitive (see
  ``vendor/bbar_fragmentation/utils.py``). If any merge fails, the whole sample
  is rejected.

All other samples (non-leaf flip, multi-anchor replacement, UNK, failed merge)
return ``None``. This yields roughly 15-25% of pool on LIT-PCBA samples,
enough for the Vina pipeline to start while a more permissive decoder is left
to Phase 5 polish.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType, Mol, RWMol

from thermofrag.data._brics_shim import ensure_brics


RDLogger.DisableLog("rdApp.*")


def _bbar():
    ensure_brics()
    from bbar_fragmentation.brics import brics_fragmentation  # type: ignore
    from bbar_fragmentation import utils as bbar_utils  # type: ignore
    return brics_fragmentation, bbar_utils


@dataclass
class DecodeResult:
    smiles: Optional[str]
    mode: str  # 'identical', 'leaf_flip', 'skip_topology', 'skip_unk',
               # 'skip_anchor_mismatch', 'skip_non_leaf', 'fail_decompose',
               # 'fail_merge', 'fail_sanitize'
    n_flips: int = 0


@dataclass
class FragmentLibraryIndex:
    """Minimal lookup over ``fragment_library.parquet``."""

    smi_to_id: dict[str, int] = field(default_factory=dict)
    id_to_smi: dict[int, str] = field(default_factory=dict)
    id_to_anchors: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "FragmentLibraryIndex":
        df = pd.read_parquet(path)
        return cls(
            smi_to_id=dict(zip(df["fragment_smi"], df["frag_id"].astype(int))),
            id_to_smi=dict(zip(df["frag_id"].astype(int), df["fragment_smi"])),
            id_to_anchors=dict(zip(df["frag_id"].astype(int), df["n_anchors_mode"].astype(int))),
        )


def _canonical_core(unit) -> str:
    return Chem.MolToSmiles(unit.to_rdmol(), canonical=True, isomericSmiles=False)


def _build_fragment_with_dummy(core_smi: str, bondtype_int: int) -> list[Mol]:
    """Return candidate fragment Mols, each with one dummy atom attached.

    For each heavy atom in ``core_smi`` with a free valence slot, produce a
    variant where that atom carries a ``*`` attached via ``bondtype_int``.
    Returned in atom-index order; the caller tries them sequentially.
    """
    _, bbar_utils = _bbar()
    core = Chem.MolFromSmiles(core_smi)
    if core is None:
        return []
    bondtype = BondType.values[bondtype_int]
    out: list[Mol] = []
    n = core.GetNumAtoms()
    for i in range(n):
        # Work on a fresh RWMol per candidate to isolate failure.
        rw = RWMol(core)
        try:
            bbar_utils.add_dummy_atom(rw, i, bondtype)
        except Exception:
            continue
        try:
            frag = rw.GetMol()
            Chem.SanitizeMol(frag)
        except Exception:
            continue
        # Must round-trip via SMILES to match BBAR's to_fragment convention
        # (rebalances H counts after dummy placement).
        smi = Chem.MolToSmiles(frag)
        parsed = Chem.MolFromSmiles(smi)
        if parsed is None:
            continue
        out.append(parsed)
    return out


def _merge_core_block(core: Mol, block: Mol, core_atom_idx: int, block_dummy_idx: int,
                      bbar_utils) -> Mol:
    """Local replacement for the broken ``vendor/bbar_fragmentation/utils.merge``.

    The vendored version references undefined ``scaffold`` / ``fragment`` names
    (sloppy rename from a refactor). We reimplement the documented behavior
    here: combine core+block, connect ``core[core_atom_idx]`` to the atom
    adjacent to the dummy in ``block`` with the dummy's bondtype, then remove
    the dummy.
    """
    rw = Chem.RWMol(Chem.CombineMols(core, block))
    dummy_abs_idx = core.GetNumAtoms() + block_dummy_idx
    dummy_atom = rw.GetAtomWithIdx(dummy_abs_idx)
    assert bbar_utils.check_dummy_atom(dummy_atom)
    bondtype = bbar_utils.get_dummy_bondtype(dummy_atom)
    neigh_idx = dummy_atom.GetNeighbors()[0].GetIdx()
    bbar_utils.create_bond(rw, core_atom_idx, neigh_idx, bondtype)
    rw.RemoveAtom(dummy_abs_idx)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def _attempt_merge(scaffold: Mol, fragment_candidates: list[Mol],
                   scaffold_atom_index: int, bbar_utils) -> Optional[Mol]:
    """Try to merge scaffold with each fragment candidate; return first success."""
    for frag in fragment_candidates:
        dummy_idx = bbar_utils.find_dummy_atom(frag)
        if dummy_idx is None:
            continue
        try:
            merged = _merge_core_block(scaffold, frag, scaffold_atom_index, dummy_idx, bbar_utils)
        except Exception:
            continue
        smi = Chem.MolToSmiles(merged)
        parsed = Chem.MolFromSmiles(smi)
        if parsed is not None:
            return parsed
    return None


def decode_sample(
    sample: dict,
    lib: FragmentLibraryIndex,
    seed_cache: dict[str, object] | None = None,
) -> DecodeResult:
    """Decode a single Phase-4 sample record to a SMILES string.

    ``sample`` must carry the keys ``seed_smiles`` and ``frag_id`` (list of
    ints, length = number of BRICS units in the seed).

    ``seed_cache`` (optional) caches the seed's BRICS decomposition so that
    decoding many samples from a single seed stays O(1) per sample.
    """
    brics_fragmentation, bbar_utils = _bbar()

    seed_smi = sample["seed_smiles"]
    sampled_ids = list(sample["frag_id"])

    cache_hit = seed_cache.get(seed_smi) if seed_cache is not None else None
    if cache_hit is None:
        try:
            g = brics_fragmentation(seed_smi)
        except Exception:
            if seed_cache is not None:
                seed_cache[seed_smi] = False
            return DecodeResult(None, "fail_decompose")
        seed_ids: list[int] = []
        seed_deg: list[int] = []
        unit_cores: list[str] = []
        for u in g.units:
            try:
                core = _canonical_core(u)
            except Exception:
                if seed_cache is not None:
                    seed_cache[seed_smi] = False
                return DecodeResult(None, "fail_decompose")
            seed_ids.append(lib.smi_to_id.get(core, 0))
            seed_deg.append(len(u.connections))
            unit_cores.append(core)
        cache_hit = (g, seed_ids, seed_deg, unit_cores)
        if seed_cache is not None:
            seed_cache[seed_smi] = cache_hit
    elif cache_hit is False:
        return DecodeResult(None, "fail_decompose")

    g, seed_ids, seed_deg, unit_cores = cache_hit  # type: ignore
    if len(sampled_ids) != len(seed_ids):
        return DecodeResult(None, "skip_topology")

    diffs = [i for i in range(len(seed_ids)) if seed_ids[i] != sampled_ids[i]]

    if not diffs:
        mol = Chem.MolFromSmiles(seed_smi)
        if mol is None:
            return DecodeResult(None, "fail_sanitize")
        return DecodeResult(Chem.MolToSmiles(mol), "identical", 0)

    # Strict MVP gate: every flip must be leaf->leaf-anchor-1.
    for i in diffs:
        new_id = sampled_ids[i]
        if new_id == 0:
            return DecodeResult(None, "skip_unk", len(diffs))
        if seed_deg[i] != 1:
            return DecodeResult(None, "skip_non_leaf", len(diffs))
        if lib.id_to_anchors.get(new_id, 0) != 1:
            return DecodeResult(None, "skip_anchor_mismatch", len(diffs))

    # Build the shared scaffold = seed with all leaf units removed.
    all_units = list(g.units)
    flip_units = {all_units[i] for i in diffs}
    scaffold_units = [u for u in all_units if u not in flip_units]
    if not scaffold_units:
        return DecodeResult(None, "skip_topology", len(diffs))

    atom_map: dict[int, int] = {}
    try:
        scaffold = g.get_submol(scaffold_units, atomMap=atom_map)
    except Exception:
        return DecodeResult(None, "fail_sanitize", len(diffs))

    # For each flipped leaf, find its unique connection to the scaffold and
    # graft the new fragment.
    for i in diffs:
        leaf = all_units[i]
        if len(leaf.connections) != 1:
            return DecodeResult(None, "skip_non_leaf", len(diffs))
        conn = leaf.connections[0]
        u0, u1 = conn.units
        if u0 is leaf:
            scaf_side = u1
            scaf_atom_src = conn.atom_indices[1]
        else:
            scaf_side = u0
            scaf_atom_src = conn.atom_indices[0]
        if scaf_side not in scaffold_units:
            return DecodeResult(None, "skip_non_leaf", len(diffs))
        try:
            scaf_atom_idx = atom_map[scaf_atom_src]
        except KeyError:
            return DecodeResult(None, "fail_sanitize", len(diffs))

        new_core_smi = lib.id_to_smi.get(sampled_ids[i])
        if new_core_smi is None or new_core_smi == "__UNK__":
            return DecodeResult(None, "skip_unk", len(diffs))

        bondtype_int = conn._bondtype
        frag_candidates = _build_fragment_with_dummy(new_core_smi, bondtype_int)
        if not frag_candidates:
            return DecodeResult(None, "fail_merge", len(diffs))

        merged = _attempt_merge(scaffold, frag_candidates, scaf_atom_idx, bbar_utils)
        if merged is None:
            return DecodeResult(None, "fail_merge", len(diffs))
        scaffold = merged
        # atom_map entries for surviving scaffold atoms stay valid because
        # merge() appends new atoms after the existing scaffold.

    smi = Chem.MolToSmiles(scaffold)
    parsed = Chem.MolFromSmiles(smi)
    if parsed is None:
        return DecodeResult(None, "fail_sanitize", len(diffs))
    return DecodeResult(Chem.MolToSmiles(parsed), "leaf_flip", len(diffs))


def decode_pool(
    samples: list[dict],
    lib: FragmentLibraryIndex,
) -> list[DecodeResult]:
    cache: dict[str, object] = {}
    return [decode_sample(s, lib, cache) for s in samples]
