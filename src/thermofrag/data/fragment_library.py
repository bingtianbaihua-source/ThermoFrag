"""Build a BRICS fragment library from a SMILES corpus.

Wraps the vendored BBAR BRICS routine (vendor/bbar_fragmentation/) so we keep
exact compatibility with the reference fragmentation rules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow importing the vendored BBAR fragmentation code as `bbar_fragmentation`.
_VENDOR = Path(__file__).resolve().parents[3] / "vendor"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))


def build_library(smiles_path: Path, out_path: Path, min_freq: int = 5) -> None:
    """Read SMILES from smiles_path, BRICS-fragment, write parquet at out_path.

    The output schema written matches what ``scripts/build_zinc_fragments.py`` emits:
        frag_id: int              position in the vocabulary (0 = UNK)
        fragment_smi: str         canonical core SMILES (no anchor dummy)
        freq: int                 number of occurrences in corpus
        n_anchors_mode: int       modal attachment-slot count

    Thin wrapper over ``scripts/build_zinc_fragments.py`` that only emits the
    library file (no LMDB). Keeps ``scripts/build_zinc_fragments.py`` as the
    canonical entry point for the combined ZINC build.
    """
    import pandas as pd
    from collections import Counter

    from ._brics_shim import ensure_brics

    ensure_brics()
    from bbar_fragmentation.brics import brics_fragmentation  # type: ignore
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    smiles_list: list[str]
    if str(smiles_path).endswith(".csv"):
        smiles_list = pd.read_csv(smiles_path)["SMILES"].tolist()
    else:
        smiles_list = [line.strip().split()[0] for line in Path(smiles_path).read_text().splitlines() if line.strip()]

    freq: Counter = Counter()
    anchor_counts: dict[str, Counter] = {}
    for smi in smiles_list:
        try:
            g = brics_fragmentation(smi)
        except Exception:
            continue
        for u in g.units:
            try:
                core = Chem.MolToSmiles(u.to_rdmol(), canonical=True, isomericSmiles=False)
            except Exception:
                continue
            freq[core] += 1
            anchor_counts.setdefault(core, Counter())[len(u.connections)] += 1

    kept = [s for s, c in freq.most_common() if c >= min_freq]
    rows = [{"frag_id": 0, "fragment_smi": "__UNK__", "freq": 0, "n_anchors_mode": 0}]
    for i, s in enumerate(kept, start=1):
        a_mode = anchor_counts[s].most_common(1)[0][0]
        rows.append({"frag_id": i, "fragment_smi": s, "freq": int(freq[s]), "n_anchors_mode": int(a_mode)})
    pd.DataFrame(rows).to_parquet(out_path)


def load_library(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
