"""Build the Phase-3 conditional LMDB with full 8-dim property vector phi per molecule.

Reads ``data/processed/zinc_unconditional.lmdb`` (Phase-2 output), computes the
property feature vector phi = [logP, qed, sa, tpsa, mw, hba, hbd, rotb] using
``thermofrag.data.properties.compute_phi``, and writes a new LMDB with the same
graph fields plus a ``phi`` array per record.

The output path defaults to ``data/processed/chembl_conditional.lmdb`` to match
configs/default.yaml. The name is historical (see docs/DATA.md); the actual
source here is ZINC + RDKit properties, which is sufficient for Phase-3's C2
calibration goal (mu(y) vs Wildman-Crippen / Bickerton weights).

Usage::

    python scripts/build_conditional_lmdb.py \
        --src data/processed/zinc_unconditional.lmdb \
        --dst data/processed/chembl_conditional.lmdb \
        --n-jobs 8
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from thermofrag.data.properties import REGISTRY, compute_phi

RDLogger.DisableLog("rdApp.*")

PROPERTIES = ["logP", "qed", "sa", "tpsa", "mw", "hba", "hbd", "rotb"]


def _compute_phi_from_smiles(smiles: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return compute_phi(mol, PROPERTIES)
    except Exception:
        return None


def _worker(item):
    key, smiles = item
    phi = _compute_phi_from_smiles(smiles)
    if phi is None:
        return key, None
    return key, phi.astype(np.float32)


def iter_records(src: Path):
    import lmdb

    env = lmdb.open(str(src), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        cur = txn.cursor()
        for k, v in cur:
            if k == b"__meta__":
                continue
            yield bytes(k), pickle.loads(v)
    env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("data/processed/zinc_unconditional.lmdb"))
    p.add_argument("--dst", type=Path, default=Path("data/processed/chembl_conditional.lmdb"))
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--max-mols", type=int, default=None)
    args = p.parse_args()

    assert all(name in REGISTRY for name in PROPERTIES), "property registry mismatch"

    # Gather all (key, smiles) pairs + full records up front. 250k records is fine.
    print(f"[build-conditional] scanning {args.src}")
    t0 = time.time()
    keys: list[bytes] = []
    records: list[dict] = []
    for k, rec in iter_records(args.src):
        keys.append(k)
        records.append(rec)
        if args.max_mols is not None and len(records) >= args.max_mols:
            break
    print(f"[build-conditional]   read {len(records)} records in {time.time()-t0:.1f}s")

    # Parallel phi computation.
    t0 = time.time()
    phis: dict[bytes, np.ndarray] = {}
    work = [(k, r["smiles"]) for k, r in zip(keys, records)]
    if args.n_jobs == 1:
        for item in work:
            k, phi = _worker(item)
            if phi is not None:
                phis[k] = phi
    else:
        with Pool(processes=args.n_jobs) as pool:
            for k, phi in pool.imap_unordered(_worker, work, chunksize=256):
                if phi is not None:
                    phis[k] = phi
    elapsed = time.time() - t0
    n_ok = len(phis)
    print(
        f"[build-conditional]   phi computed: {n_ok}/{len(records)} "
        f"in {elapsed:.1f}s ({n_ok/max(elapsed,1e-6):.1f} mol/s)"
    )

    # Write new LMDB.
    import lmdb

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    # Same order-of-magnitude size as source + 8 floats per rec ~= 32 bytes extra.
    map_size = 4 * 1024 * 1024 * 1024  # 4 GB
    env = lmdb.open(str(args.dst), map_size=map_size, subdir=False, lock=False)

    split_counts = [0, 0, 0]
    n_written = 0
    with env.begin(write=True) as txn:
        for k, rec in zip(keys, records):
            phi = phis.get(k)
            if phi is None:
                continue
            out = dict(rec)  # shallow copy; rec["props"] dict and graph lists are reused
            out["phi"] = phi.tolist()
            out["phi_properties"] = PROPERTIES
            txn.put(f"{n_written:010d}".encode(), pickle.dumps(out, protocol=4))
            split_counts[int(rec.get("split", 0))] += 1
            n_written += 1
        # Compute phi normalization stats across all written records.
        phi_mat = np.stack([phis[k] for k in keys if k in phis], axis=0)
        phi_mean = phi_mat.mean(axis=0).astype(np.float32)
        phi_std = phi_mat.std(axis=0).astype(np.float32)
        phi_std = np.maximum(phi_std, 1e-6)  # guard rare constant dims
        meta = {
            "n_molecules": n_written,
            "n_fragments": 0,  # filled below from the source meta if available
            "split_counts": {"train": split_counts[0], "val": split_counts[1], "test": split_counts[2]},
            "schema": "pickle",
            "phi_properties": PROPERTIES,
            "phi_dim": len(PROPERTIES),
            "phi_mean": phi_mean.tolist(),
            "phi_std": phi_std.tolist(),
        }
        # Preserve n_fragments from source meta so downstream code resolves vocab size.
        src_env = lmdb.open(str(args.src), subdir=False, readonly=True, lock=False)
        with src_env.begin() as stxn:
            src_meta = pickle.loads(stxn.get(b"__meta__"))
        src_env.close()
        meta["n_fragments"] = int(src_meta.get("n_fragments", 0))
        txn.put(b"__meta__", pickle.dumps(meta, protocol=4))
    env.close()

    print(f"[build-conditional] wrote {n_written} records -> {args.dst}")
    print(f"[build-conditional] meta: {json.dumps(meta, indent=2)}")

    # Summary stats on phi.
    phi_stack = np.stack([phis[k] for k in keys if k in phis], axis=0)
    print("[build-conditional] phi per-dim stats (mean / std / min / max):")
    for i, name in enumerate(PROPERTIES):
        c = phi_stack[:, i]
        print(f"  {name:>6s}  mean={c.mean():8.3f}  std={c.std():8.3f}  min={c.min():8.3f}  max={c.max():8.3f}")


if __name__ == "__main__":
    main()
