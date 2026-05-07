"""Build a conditional LMDB from CrossDocked2020 for TF-pocket fine-tuning.

Consumes the TargetDiff-preprocessed CrossDocked LMDB
(``crossdocked_v1.1_rmsd1.0_pocket10_processed_*.lmdb``) directly, so no
raw PDB/SDF download is required. Each input record already has the
10 Å pocket atom table, the ligand SMILES, and Vina dock / QED / SA
attached — we just compute the 8-dim property vector phi from the SMILES,
derive a pocket id from the residue sequence (matching the id that
``scripts/precompute_pocket_embeddings.py --preprocessed-lmdb`` writes),
and repack into the trainer's expected record layout.

Output record schema (one per pose)::

    {
      "smiles": str,
      "phi": list[float],        # 8 entries in PROPERTIES order
      "phi_properties": list[str],
      "pocket_id": str,          # matches <pocket_embeds_dir>/<pocket_id>.npy
      "protein_filename": str,   # provenance (preprocessed-LMDB field)
      "ligand_filename": str,
      "vina_dock": float,        # carried over for optional loss weighting
      "split": int,              # 0 train / 1 val / 2 test
    }

The TargetDiff split ``.pt`` holds integer indices into the preprocessed
LMDB (train/val/test). The shipped split has train=99990 / val=0 /
test=100; we carve out a deterministic ~2% val slice from train so the
trainer has a held-out set for pocket_acc@B monitoring.

Usage::

    python scripts/build_crossdocked_lmdb.py \\
        --preprocessed-lmdb ~/下载/crossdocked_v1.1_rmsd1.0_pocket10_processed_dock_guide_final-002.lmdb \\
        --split-pt ~/下载/crossdocked_pocket10_pose_split_dock_guide.pt \\
        --pocket-embeds data/processed/pocket_embeds/crossdocked \\
        --out data/processed/crossdocked_conditional.lmdb
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

from thermofrag.data.properties import compute_phi
from thermofrag.potentials.pocket_encoder import (
    pocket_id_from_sequence,
    sequence_from_preprocessed_record,
)

RDLogger.DisableLog("rdApp.*")

PROPERTIES = ["logP", "qed", "sa", "tpsa", "mw", "hba", "hbd", "rotb"]


def _read_split(split_pt: Path) -> dict:
    """TargetDiff split .pt — keys train/val/test, values lists of ints."""
    try:
        import torch
        return torch.load(str(split_pt), map_location="cpu", weights_only=False)
    except Exception:
        with split_pt.open("rb") as f:
            return pickle.load(f)


def _carve_val_from_train(train_idx: list[int], frac: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    idxs = list(train_idx)
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(idxs) * frac)))
    val = sorted(idxs[:n_val])
    train = sorted(idxs[n_val:])
    return train, val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed-lmdb", type=Path, required=True,
                   help="TargetDiff-preprocessed CrossDocked LMDB (pocket10_processed)")
    p.add_argument("--split-pt", type=Path, required=True,
                   help="TargetDiff split .pt: dict train/val/test of int indices")
    p.add_argument("--pocket-embeds", type=Path, required=True,
                   help="dir of <pocket_id>.npy from precompute_pocket_embeddings.py")
    p.add_argument("--out", type=Path,
                   default=Path("data/processed/crossdocked_conditional.lmdb"))
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="fraction of train to carve into val when split.val is empty")
    p.add_argument("--val-seed", type=int, default=42)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--map-size-gb", type=int, default=8)
    p.add_argument("--require-pocket-embed", action="store_true",
                   help="skip records whose pocket .npy is missing (default: write anyway)")
    args = p.parse_args()

    split_raw = _read_split(args.split_pt)
    train_ids = [int(x) for x in split_raw.get("train", [])]
    val_ids = [int(x) for x in split_raw.get("val", [])]
    test_ids = [int(x) for x in split_raw.get("test", [])]
    if not val_ids and train_ids:
        train_ids, val_ids = _carve_val_from_train(train_ids, args.val_frac, args.val_seed)
        print(f"[cd-lmdb] carved val from train: "
              f"train={len(train_ids)} val={len(val_ids)} (seed={args.val_seed})")

    ordered: list[tuple[int, int]] = []  # (split_id, lmdb_idx)
    for idx in train_ids:
        ordered.append((0, idx))
    for idx in val_ids:
        ordered.append((1, idx))
    for idx in test_ids:
        ordered.append((2, idx))
    if args.max_records is not None:
        ordered = ordered[: args.max_records]

    print(f"[cd-lmdb] {len(ordered)} pose records "
          f"train={sum(1 for s,_ in ordered if s==0)} "
          f"val={sum(1 for s,_ in ordered if s==1)} "
          f"test={sum(1 for s,_ in ordered if s==2)}")

    import lmdb

    src_env = lmdb.open(
        str(args.preprocessed_lmdb),
        subdir=False, readonly=True, lock=False,
        readahead=False, meminit=False,
        map_size=32 * 1024 ** 3,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dst_env = lmdb.open(
        str(args.out),
        map_size=args.map_size_gb * 1024 ** 3,
        subdir=False,
        lock=False,
    )

    seen_pockets: set[str] = set()
    missing_pockets: set[str] = set()
    n_written = 0
    n_skip_missing = 0
    n_skip_badsmiles = 0
    split_counts = [0, 0, 0]
    phi_rows: list[np.ndarray] = []
    t0 = time.time()

    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        for sid, idx in ordered:
            raw = src_txn.get(str(idx).encode())
            if raw is None:
                continue
            rec = pickle.loads(raw)

            seq = sequence_from_preprocessed_record(rec)
            if not seq:
                continue
            pid = pocket_id_from_sequence(seq)
            embed_path = args.pocket_embeds / f"{pid}.npy"
            if not embed_path.is_file():
                missing_pockets.add(pid)
                if args.require_pocket_embed:
                    n_skip_missing += 1
                    continue

            smiles = rec.get("ligand_smiles")
            if not smiles:
                n_skip_badsmiles += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                n_skip_badsmiles += 1
                continue
            try:
                phi = compute_phi(mol, PROPERTIES).astype(np.float32)
                smiles_canon = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                n_skip_badsmiles += 1
                continue

            out_rec = {
                "smiles": smiles_canon,
                "phi": phi.tolist(),
                "phi_properties": PROPERTIES,
                "pocket_id": pid,
                "protein_filename": rec.get("protein_filename", ""),
                "ligand_filename": rec.get("ligand_filename", ""),
                "vina_dock": float(rec.get("vina_dock", 0.0)),
                "split": sid,
            }
            dst_txn.put(f"{n_written:010d}".encode(), pickle.dumps(out_rec, protocol=4))
            seen_pockets.add(pid)
            split_counts[sid] += 1
            phi_rows.append(phi)
            n_written += 1
            if n_written % 5000 == 0:
                print(f"[cd-lmdb]   {n_written} written ({time.time()-t0:.1f}s)")

        phi_mat = np.stack(phi_rows, axis=0)
        phi_mean = phi_mat.mean(axis=0).astype(np.float32)
        phi_std = np.maximum(phi_mat.std(axis=0), 1e-6).astype(np.float32)
        meta = {
            "n_records": n_written,
            "n_pockets": len(seen_pockets),
            "split_counts": {
                "train": split_counts[0],
                "val": split_counts[1],
                "test": split_counts[2],
            },
            "schema": "pickle",
            "phi_properties": PROPERTIES,
            "phi_dim": len(PROPERTIES),
            "phi_mean": phi_mean.tolist(),
            "phi_std": phi_std.tolist(),
            "pocket_embeds_dir": str(args.pocket_embeds),
            "source": "CrossDocked2020_TargetDiff_preprocessed",
            "source_lmdb": str(args.preprocessed_lmdb),
            "val_carved_from_train": (not split_raw.get("val")),
            "val_frac": args.val_frac,
            "val_seed": args.val_seed,
        }
        dst_txn.put(b"__meta__", pickle.dumps(meta, protocol=4))
    src_env.close()
    dst_env.close()

    print(f"[cd-lmdb] wrote {n_written} records -> {args.out}")
    print(f"[cd-lmdb] {len(seen_pockets)} unique pockets referenced, "
          f"{len(missing_pockets)} missing from {args.pocket_embeds}")
    print(f"[cd-lmdb] skipped: missing_embed={n_skip_missing} bad_smiles={n_skip_badsmiles}")
    print(f"[cd-lmdb] meta: {json.dumps({k:v for k,v in meta.items() if k not in ('phi_mean','phi_std')}, indent=2)}")


if __name__ == "__main__":
    main()
