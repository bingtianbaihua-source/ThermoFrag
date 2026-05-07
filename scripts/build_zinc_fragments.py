"""Build the Phase-2 ZINC fragment dataset and fragment library.

Reads BBAR's pre-split ZINC CSV + split at /home/chaoxue/code/BBAR/data/ZINC/ (the
same drug-like subset BBAR uses), BRICS-decomposes every molecule via the
vendored ``bbar_fragmentation`` code, and writes:

  * ``data/processed/fragment_library.parquet``
        vocabulary of core fragment SMILES (the unit rdmol canonicalized, no
        anchor dummy). Columns:
            frag_id (int), fragment_smi (str), freq (int), n_anchors_mode (int)

  * ``data/processed/zinc_unconditional.lmdb``
        one LMDB entry per molecule, value = msgpack-encoded dict::

            {
              "split":  int (0=train, 1=val, 2=test),
              "smiles": str,
              "frag_id": List[int],             # length N_units
              "edge_index": List[Tuple[int,int]], # E directed edges
              "bond_type": List[int],           # bond-type enum 0..7
              "props": {"mw": float, "tpsa": float, "logp": float, "qed": float}
            }

The LMDB key is a zero-padded 10-digit index. A top-level "__meta__" key holds
schema metadata (n_molecules, n_fragments, split_counts).

Usage::

    python scripts/build_zinc_fragments.py \
        --zinc-dir /home/chaoxue/code/BBAR/data/ZINC \
        --out-dir  data/processed \
        --n-jobs 8 [--max-mols 20000]

Designed to be run once. A Phase-2 smoke run can use ``--max-mols 20000`` to
produce a small LMDB before committing to the full 250k mol build.
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from thermofrag.data._brics_shim import ensure_brics

ensure_brics()

# After the shim, these imports resolve.
from bbar_fragmentation.brics import brics_fragmentation  # type: ignore  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# BRICS connection.bondtype stores ``int(rdkit.Chem.BondType)``; the kit has
# only 22 unique values but we only ever see SINGLE/DOUBLE/TRIPLE/AROMATIC
# (+ rarer ones). Map them to 0..7 to stay within the CouplingPotential's
# n_bond_types=8 vocabulary.
_BOND_TYPE_TO_IDX = {
    1: 0,     # SINGLE
    2: 1,     # DOUBLE
    3: 2,     # TRIPLE
    12: 3,    # AROMATIC
    # Anything else collapses to 7 (OTHER).
}
_DEFAULT_BOND_IDX = 7


def bond_type_to_idx(raw_bondtype_int: int) -> int:
    return _BOND_TYPE_TO_IDX.get(int(raw_bondtype_int), _DEFAULT_BOND_IDX)


def _canonical_core(unit) -> str:
    mol = unit.to_rdmol()
    # Canonical SMILES, non-isomeric (dataset is achiral as-is).
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _count_anchors(unit) -> int:
    """Number of connections on this unit (anchor slots)."""
    return len(unit.connections)


def fragment_one(smiles: str) -> dict | None:
    """BRICS-decompose a SMILES and return a compact per-mol record.

    Returns None if decomposition fails or the graph has zero units.
    """
    try:
        g = brics_fragmentation(smiles)
    except Exception:  # pragma: no cover - BBAR code can raise on weird mols
        return None
    n = g.num_units
    if n == 0:
        return None

    core_smis: List[str] = []
    n_anchors: List[int] = []
    for u in g.units:
        try:
            core_smis.append(_canonical_core(u))
        except Exception:
            return None
        n_anchors.append(_count_anchors(u))

    unit_to_idx = {u: i for i, u in enumerate(g.units)}
    edges: List[Tuple[int, int]] = []
    btypes: List[int] = []
    for c in g.connections:
        u1, u2 = c.units
        i1, i2 = unit_to_idx[u1], unit_to_idx[u2]
        b = bond_type_to_idx(c._bondtype)
        edges.append((i1, i2))
        btypes.append(b)
        edges.append((i2, i1))
        btypes.append(b)

    return {
        "smiles": smiles,
        "core_smis": core_smis,
        "n_anchors": n_anchors,
        "edges": edges,
        "bond_type": btypes,
    }


def _worker_init(_=None):
    # Re-register shim in child processes.
    ensure_brics()


def _worker(args):
    idx, smiles = args
    rec = fragment_one(smiles)
    if rec is None:
        return idx, None
    return idx, rec


def build_vocab(records: List[dict | None], min_freq: int) -> tuple[dict[str, int], Counter]:
    """Collect core-SMILES frequencies across all records, filter by min_freq, assign ids.

    Returns (smi_to_id, freq_counter). A reserved id 0 is used for OOV / UNK.
    """
    freq: Counter = Counter()
    for r in records:
        if r is None:
            continue
        for s in r["core_smis"]:
            freq[s] += 1
    kept = [s for s, c in freq.most_common() if c >= min_freq]
    smi_to_id = {"__UNK__": 0}
    for s in kept:
        smi_to_id[s] = len(smi_to_id)
    return smi_to_id, freq


def write_lmdb(
    out_path: Path,
    records: List[dict | None],
    splits: List[int],
    props: List[dict],
    smi_to_id: dict[str, int],
) -> dict:
    import lmdb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 250k mols × ~300 B = ~75 MB; give generous headroom.
    map_size = 2 * 1024 * 1024 * 1024  # 2 GB

    env = lmdb.open(str(out_path), map_size=map_size, subdir=False, lock=False)

    n_kept = 0
    split_counts = [0, 0, 0]
    with env.begin(write=True) as txn:
        for i, rec in enumerate(records):
            if rec is None:
                continue
            frag_id = [smi_to_id.get(s, 0) for s in rec["core_smis"]]
            value = {
                "split": splits[i],
                "smiles": rec["smiles"],
                "frag_id": frag_id,
                "edge_index": rec["edges"],
                "bond_type": rec["bond_type"],
                "n_anchors": rec["n_anchors"],
                "props": props[i],
            }
            txn.put(f"{n_kept:010d}".encode(), pickle.dumps(value, protocol=4))
            split_counts[splits[i]] += 1
            n_kept += 1
        meta = {
            "n_molecules": n_kept,
            "n_fragments": len(smi_to_id),
            "split_counts": {"train": split_counts[0], "val": split_counts[1], "test": split_counts[2]},
            "schema": "pickle",
        }
        txn.put(b"__meta__", pickle.dumps(meta, protocol=4))
    env.close()
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zinc-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--max-mols", type=int, default=None)
    args = p.parse_args()

    data_csv = args.zinc_dir / "data.csv"
    split_csv = args.zinc_dir / "split.csv"
    print(f"[build] reading {data_csv}")
    df = pd.read_csv(data_csv)
    print(f"[build]   n_mols={len(df)}  columns={list(df.columns)}")
    splits = pd.read_csv(split_csv, header=None, names=["split", "idx"])
    split_map = {"train": 0, "val": 1, "test": 2}
    split_arr = np.full(len(df), -1, dtype=np.int8)
    for _, row in splits.iterrows():
        s = split_map[row["split"]]
        i = int(row["idx"])
        if 0 <= i < len(split_arr):
            split_arr[i] = s
    if (split_arr == -1).any():
        missing = int((split_arr == -1).sum())
        print(f"[build]   WARNING: {missing} rows missing a split; defaulting to train")
        split_arr[split_arr == -1] = 0

    if args.max_mols is not None:
        df = df.iloc[: args.max_mols].reset_index(drop=True)
        split_arr = split_arr[: args.max_mols]
        print(f"[build]   subsampled to {len(df)} mols")

    smiles_list = df["SMILES"].tolist()
    props_list = df[["mw", "tpsa", "logp", "qed"]].to_dict(orient="records")

    # Fragment in parallel.
    t0 = time.time()
    print(f"[build] BRICS-decomposing {len(smiles_list)} mols with n_jobs={args.n_jobs}")
    records: List[dict | None] = [None] * len(smiles_list)
    work = list(enumerate(smiles_list))
    if args.n_jobs == 1:
        _worker_init()
        for ia, smi in work:
            _, rec = _worker((ia, smi))
            records[ia] = rec
    else:
        with Pool(processes=args.n_jobs, initializer=_worker_init) as pool:
            for ia, rec in pool.imap_unordered(_worker, work, chunksize=200):
                records[ia] = rec

    n_ok = sum(r is not None for r in records)
    elapsed = time.time() - t0
    print(f"[build]   {n_ok}/{len(records)} decomposed OK in {elapsed:.1f}s ({n_ok/max(elapsed,1e-6):.1f} mol/s)")

    # Build vocab.
    smi_to_id, freq = build_vocab(records, min_freq=args.min_freq)
    print(f"[build] vocab: {len(smi_to_id)} fragments (incl. UNK) with min_freq={args.min_freq}")
    print(f"[build] coverage: top-20 fragments:")
    for s, c in freq.most_common(20):
        print(f"  {c:>7d}  {s}")

    # Collect anchor-count mode per fragment for library metadata.
    anchor_counts: dict[str, Counter] = {s: Counter() for s in smi_to_id if s != "__UNK__"}
    for r in records:
        if r is None:
            continue
        for s, a in zip(r["core_smis"], r["n_anchors"]):
            if s in anchor_counts:
                anchor_counts[s][a] += 1

    # Write parquet.
    lib_rows = []
    for s, fid in smi_to_id.items():
        if s == "__UNK__":
            lib_rows.append({"frag_id": fid, "fragment_smi": s, "freq": 0, "n_anchors_mode": 0})
            continue
        c = freq[s]
        a_mode = anchor_counts[s].most_common(1)[0][0] if anchor_counts[s] else 1
        lib_rows.append({"frag_id": fid, "fragment_smi": s, "freq": int(c), "n_anchors_mode": int(a_mode)})
    lib_df = pd.DataFrame(lib_rows).sort_values("frag_id").reset_index(drop=True)
    lib_path = args.out_dir / "fragment_library.parquet"
    lib_path.parent.mkdir(parents=True, exist_ok=True)
    lib_df.to_parquet(lib_path)
    print(f"[build] fragment_library -> {lib_path}  ({len(lib_df)} rows)")

    # Write LMDB.
    lmdb_path = args.out_dir / "zinc_unconditional.lmdb"
    meta = write_lmdb(lmdb_path, records, splits=split_arr.tolist(), props=props_list, smi_to_id=smi_to_id)
    print(f"[build] lmdb -> {lmdb_path}")
    print(f"[build] meta: {json.dumps(meta, indent=2)}")

    # Stats: distribution of #units per mol and edge counts.
    n_units = np.array([len(r["core_smis"]) for r in records if r is not None], dtype=np.int32)
    n_edges = np.array([len(r["edges"]) // 2 for r in records if r is not None], dtype=np.int32)
    print(
        f"[build] per-mol stats: n_units mean={n_units.mean():.2f} max={n_units.max()} "
        f"| n_edges(undirected) mean={n_edges.mean():.2f} max={n_edges.max()}"
    )


if __name__ == "__main__":
    main()
