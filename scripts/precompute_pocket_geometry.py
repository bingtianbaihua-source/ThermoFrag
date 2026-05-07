"""Precompute pocket Cα geometry caches for TF-pocket v3 (EGNN variant).

For each unique pocket we store ``coords (N, 3) float32`` and
``aa_idx (N,) int64`` under ``<out>/<pocket_id>.npz``. The ``pocket_id``
is a 16-hex sha1 prefix over the 1-letter residue sequence — identical
to the ``pocket_id`` used in the preprocessed LMDB and in the ESM-2
``.npy`` embeddings shipped for v1/v2, so records and geometry caches
match up automatically.

Two input modes, mirroring ``scripts/precompute_pocket_embeddings.py``:

* LIT-PCBA receptors: ``--receptors data/external/receptors``. Each
  subdirectory must contain ``receptor_clean.pdb`` (or ``receptor.pdb``)
  and ``cognate_ligand.{sdf,pdb}``. Output keyed on the target name
  (e.g. ``VDR.npz``) so the sampler can look up by LIT-PCBA name.
* CrossDocked2020 preprocessed LMDB: ``--preprocessed-lmdb <lmdb>``. The
  script iterates every record, deduplicates on pocket sequence, and
  writes one ``.npz`` per unique pocket under ``<out>/<sha1>.npz``.

Usage::

    # LIT-PCBA
    python scripts/precompute_pocket_geometry.py \\
        --receptors data/external/receptors \\
        --out data/processed/pocket_geom/litpcba

    # CrossDocked (CPU-friendly; no model in the loop)
    python scripts/precompute_pocket_geometry.py \\
        --preprocessed-lmdb /path/to/crossdocked_*_pocket10_processed_*.lmdb \\
        --out data/processed/pocket_geom/crossdocked
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from thermofrag.potentials.pocket_egnn import (
    extract_pocket_ca_geometry,
    geometry_from_preprocessed_record,
    pocket_geom_id,
    save_pocket_geom,
)


def _find_receptor_and_ligand(target_dir: Path) -> tuple[Path, Path] | None:
    rec_candidates = ["receptor_clean.pdb", "receptor.pdb"]
    lig_candidates = ["cognate_ligand.sdf", "cognate_ligand.pdb"]
    rec = next((target_dir / n for n in rec_candidates if (target_dir / n).is_file()), None)
    lig = next((target_dir / n for n in lig_candidates if (target_dir / n).is_file()), None)
    if rec is None or lig is None:
        return None
    return rec, lig


def _run_litpcba(root: Path, out_dir: Path, cutoff_a: float, force: bool) -> int:
    n_ok = 0
    t0 = time.time()
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        pair = _find_receptor_and_ligand(sub)
        if pair is None:
            print(f"[precompute-geom] skip {sub.name}: receptor/ligand missing", file=sys.stderr)
            continue
        tid = sub.name
        out_path = out_dir / f"{tid}.npz"
        if out_path.exists() and not force:
            print(f"[precompute-geom] {tid}: cached")
            n_ok += 1
            continue
        try:
            coords, aa = extract_pocket_ca_geometry(pair[0], pair[1], cutoff_a=cutoff_a)
            save_pocket_geom(out_path, coords, aa)
            print(f"[precompute-geom] {tid}: n_res={coords.shape[0]}")
            n_ok += 1
        except Exception as e:
            print(f"[precompute-geom] {tid}: FAILED ({e})", file=sys.stderr)
    print(f"[precompute-geom] LIT-PCBA done: {n_ok} targets in {time.time()-t0:.1f}s")
    return n_ok


def _run_preprocessed_lmdb(
    lmdb_path: Path,
    out_dir: Path,
    force: bool,
    max_records: int | None,
) -> int:
    import lmdb as _lmdb  # lazy

    env = _lmdb.open(
        str(lmdb_path),
        subdir=False, readonly=True, lock=False,
        readahead=False, meminit=False,
        map_size=32 * 1024 ** 3,
    )
    unique: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    with env.begin() as txn:
        cur = txn.cursor()
        for i, (k, v) in enumerate(cur):
            rec = pickle.loads(v)
            try:
                coords, aa = geometry_from_preprocessed_record(rec)
            except Exception as e:
                print(f"[precompute-geom]   record {k}: {e}", file=sys.stderr)
                continue
            if coords.shape[0] == 0:
                continue
            pid = pocket_geom_id(coords, aa)
            if pid not in unique:
                unique[pid] = (coords, aa)
            if max_records is not None and len(unique) >= max_records:
                break
            if (i + 1) % 20000 == 0:
                print(f"[precompute-geom]   scan {i+1}  unique={len(unique)}  "
                      f"{time.time()-t0:.1f}s")
    print(f"[precompute-geom] scan done: {len(unique)} unique pockets "
          f"in {time.time()-t0:.1f}s")

    n_ok = 0
    t1 = time.time()
    n_total = len(unique)
    for j, (pid, (coords, aa)) in enumerate(unique.items()):
        out_path = out_dir / f"{pid}.npz"
        if out_path.exists() and not force:
            continue
        save_pocket_geom(out_path, coords, aa)
        n_ok += 1
        if (j + 1) % 1000 == 0:
            elapsed = time.time() - t1
            rate = (j + 1) / max(elapsed, 1e-6)
            eta = (n_total - j - 1) / max(rate, 1e-6)
            print(f"[precompute-geom]   save {j+1}/{n_total}  "
                  f"{rate:.1f}/s  eta {eta/60:.1f} min")
    print(f"[precompute-geom] wrote {n_ok} .npz files to {out_dir}")
    return n_ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--receptors", type=Path, help="LIT-PCBA-style receptor root")
    p.add_argument("--preprocessed-lmdb", type=Path,
                   help="TargetDiff-preprocessed CrossDocked LMDB (pocket10_processed)")
    p.add_argument("--out", type=Path, required=True,
                   help="output dir for .npz files")
    p.add_argument("--cutoff-a", type=float, default=10.0)
    p.add_argument("--max-records", type=int, default=None,
                   help="cap on unique pockets (smoke-test aid)")
    p.add_argument("--force", action="store_true", help="recompute even if .npz exists")
    args = p.parse_args()

    n_sources = sum(x is not None for x in (args.receptors, args.preprocessed_lmdb))
    if n_sources != 1:
        p.error("pass exactly one of --receptors, --preprocessed-lmdb")

    args.out.mkdir(parents=True, exist_ok=True)

    if args.preprocessed_lmdb is not None:
        n_ok = _run_preprocessed_lmdb(
            args.preprocessed_lmdb, args.out,
            force=args.force, max_records=args.max_records,
        )
    else:
        n_ok = _run_litpcba(args.receptors, args.out,
                            cutoff_a=float(args.cutoff_a), force=args.force)
    print(f"[precompute-geom] done: {n_ok} pockets -> {args.out}")


if __name__ == "__main__":
    main()
