"""Unit tests for thermofrag.sampling.decoder."""
from __future__ import annotations

import pytest
from rdkit import Chem, RDLogger

from thermofrag.data._brics_shim import ensure_brics
from thermofrag.sampling.decoder import (
    FragmentLibraryIndex,
    decode_sample,
)


RDLogger.DisableLog("rdApp.*")


LIB_PATH = "data/processed/fragment_library.parquet"


@pytest.fixture(scope="module")
def lib() -> FragmentLibraryIndex:
    return FragmentLibraryIndex.from_parquet(LIB_PATH)


@pytest.fixture(scope="module")
def bbar():
    ensure_brics()
    from bbar_fragmentation.brics import brics_fragmentation  # type: ignore
    return brics_fragmentation


def _seed_sample(seed: str, lib: FragmentLibraryIndex, brics_fragmentation) -> dict:
    """Build a no-flip sample record: sampled frag_id == seed's frag_id."""
    g = brics_fragmentation(seed)
    seed_ids: list[int] = []
    for u in g.units:
        core = Chem.MolToSmiles(u.to_rdmol(), canonical=True, isomericSmiles=False)
        seed_ids.append(lib.smi_to_id.get(core, 0))
    return {"seed_smiles": seed, "frag_id": seed_ids}


def test_identical_roundtrip(lib, bbar):
    """A sample whose frag_id matches the seed decomposition decodes to the seed."""
    seed = "Cc1cccc(OC(C)C(=O)NC2(C)CCS(=O)(=O)C2)c1"
    sample = _seed_sample(seed, lib, bbar)
    res = decode_sample(sample, lib)
    assert res.mode == "identical"
    assert Chem.CanonSmiles(res.smiles) == Chem.CanonSmiles(seed)
    assert res.n_flips == 0


def test_single_leaf_flip(lib, bbar):
    """Flipping one leaf to a 1-anchor core returns a valid SMILES different from seed."""
    seed = "Cc1cccc(OC(C)C(=O)NC2(C)CCS(=O)(=O)C2)c1"
    sample = _seed_sample(seed, lib, bbar)
    # The seed's node 4 is a leaf at the amide N; swap it for a 1-anchor fragment.
    # id 241 is C1CC2CCC1C2 (norbornane) with n_anchors_mode=1.
    assert lib.id_to_anchors[241] == 1
    sample["frag_id"][4] = 241
    res = decode_sample(sample, lib)
    assert res.mode == "leaf_flip", f"expected leaf_flip, got {res.mode}"
    assert res.n_flips == 1
    assert res.smiles is not None
    mol = Chem.MolFromSmiles(res.smiles)
    assert mol is not None
    # New SMILES must differ from seed.
    assert Chem.CanonSmiles(res.smiles) != Chem.CanonSmiles(seed)


def test_non_leaf_flip_rejected(lib, bbar):
    """A flip at a non-leaf position is skipped (returns None, mode=skip_non_leaf)."""
    seed = "Cc1cccc(OC(C)C(=O)NC2(C)CCS(=O)(=O)C2)c1"
    sample = _seed_sample(seed, lib, bbar)
    # Node 1 in this seed has degree 2 (internal). Flip it to any fragment.
    sample["frag_id"][1] = 241  # 1-anchor core, but slot is degree 2 → skip
    res = decode_sample(sample, lib)
    assert res.smiles is None
    assert res.mode in ("skip_non_leaf", "skip_anchor_mismatch")


def test_anchor_mismatch_rejected(lib, bbar):
    """A leaf position flipped to a multi-anchor fragment is rejected."""
    seed = "Cc1cccc(OC(C)C(=O)NC2(C)CCS(=O)(=O)C2)c1"
    sample = _seed_sample(seed, lib, bbar)
    # Find a 2-anchor fragment id and put it at node 4 (leaf).
    two_anchor_ids = [i for i, a in lib.id_to_anchors.items() if a == 2 and i != 0]
    assert two_anchor_ids
    sample["frag_id"][4] = two_anchor_ids[0]
    res = decode_sample(sample, lib)
    assert res.smiles is None
    assert res.mode == "skip_anchor_mismatch"


def test_unk_flip_rejected(lib, bbar):
    """Flipping to frag_id=0 (UNK) is explicitly rejected."""
    seed = "Cc1cccc(OC(C)C(=O)NC2(C)CCS(=O)(=O)C2)c1"
    sample = _seed_sample(seed, lib, bbar)
    sample["frag_id"][4] = 0
    res = decode_sample(sample, lib)
    assert res.smiles is None
    assert res.mode == "skip_unk"


def test_seed_cache_reuse(lib, bbar):
    """Passing a shared seed_cache across calls does not change the decoded SMILES."""
    seed = "Cc1cccc(OC(C)C(=O)NC2(C)CCS(=O)(=O)C2)c1"
    cache: dict = {}
    s1 = _seed_sample(seed, lib, bbar)
    r1 = decode_sample(s1, lib, seed_cache=cache)
    s2 = _seed_sample(seed, lib, bbar)
    s2["frag_id"][4] = 241
    r2 = decode_sample(s2, lib, seed_cache=cache)
    assert r1.mode == "identical"
    assert r2.mode == "leaf_flip"
    assert seed in cache
    # Running the identical again after cache warmup should stay identical.
    r3 = decode_sample(s1, lib, seed_cache=cache)
    assert r3.smiles == r1.smiles
