"""Precompute pocket residue embeddings for TF-pocket training and eval.

Three input modes:

* LIT-PCBA: ``--receptors data/external/receptors`` — each subdirectory is a
  target containing ``receptor_clean.pdb`` (or ``receptor.pdb``) and
  ``cognate_ligand.pdb``/.sdf. Output keyed on target name.
* CrossDocked2020 raw: ``--index-csv`` with columns
  ``pocket_pdb,ligand_sdf,target_id``. Output keyed on ``target_id``.
* CrossDocked2020 preprocessed: ``--preprocessed-lmdb`` points at the
  TargetDiff-preprocessed LMDB (``crossdocked_v1.1_rmsd1.0_pocket10_*.lmdb``).
  No raw PDB/SDF is needed — residue sequences are reconstructed from each
  record's ``protein_atom_name``/``protein_atom_to_aa_type`` arrays, and the
  embedding for each unique sequence is saved once under
  ``<out_dir>/<sha1(sequence)[:16]>.npy``. The rewritten
  ``build_crossdocked_lmdb.py`` computes the same id so records and
  embeddings match up automatically.

Usage::

    # LIT-PCBA
    python scripts/precompute_pocket_embeddings.py \\
        --receptors data/external/receptors \\
        --out data/processed/pocket_embeds/litpcba

    # CrossDocked2020 via preprocessed LMDB (GPU recommended)
    python scripts/precompute_pocket_embeddings.py \\
        --preprocessed-lmdb /path/to/crossdocked_*_pocket10_processed_*.lmdb \\
        --out data/processed/pocket_embeds/crossdocked \\
        --device cuda --batch-size 32
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from thermofrag.potentials.pocket_encoder import (
    DEFAULT_ESM_MODEL,
    DEFAULT_POCKET_CUTOFF_A,
    PocketEncoder,
    extract_pocket_residues,
    pocket_id_from_sequence,
    residues_to_sequence,
    save_pocket_embed,
    sequence_from_preprocessed_record,
)


def _find_receptor_and_ligand(target_dir: Path) -> tuple[Path, Path] | None:
    rec_candidates = ["receptor_clean.pdb", "receptor.pdb"]
    lig_candidates = ["cognate_ligand.sdf", "cognate_ligand.pdb"]
    rec = next((target_dir / n for n in rec_candidates if (target_dir / n).is_file()), None)
    lig = next((target_dir / n for n in lig_candidates if (target_dir / n).is_file()), None)
    if rec is None or lig is None:
        return None
    return rec, lig


def _iter_litpcba(root: Path):
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        pair = _find_receptor_and_ligand(sub)
        if pair is None:
            print(f"[precompute-pockets] skip {sub.name}: receptor/ligand missing", file=sys.stderr)
            continue
        yield sub.name, pair[0], pair[1]


def _iter_index_csv(path: Path):
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["target_id"]
            rec = Path(row["pocket_pdb"])
            lig = Path(row["ligand_sdf"])
            if not rec.is_file() or not lig.is_file():
                print(f"[precompute-pockets] skip {tid}: {rec} or {lig} missing", file=sys.stderr)
                continue
            yield tid, rec, lig


def _run_preprocessed_lmdb(
    lmdb_path: Path,
    out_dir: Path,
    encoder: PocketEncoder,
    batch_size: int,
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
    # First pass: scan the LMDB once to collect *unique* pocket sequences. Each
    # CrossDocked pose record has a slightly different pocket (10 Å around that
    # particular pose), but many share the exact same residue set — deduping on
    # sha1(sequence) avoids running ESM-2 on the same sequence twice.
    t0 = time.time()
    unique: dict[str, tuple[str, int]] = {}  # pocket_id -> (sequence, sample_rec_idx)
    with env.begin() as txn:
        cur = txn.cursor()
        for i, (k, v) in enumerate(cur):
            rec = pickle.loads(v)
            seq = sequence_from_preprocessed_record(rec)
            if not seq:
                continue
            pid = pocket_id_from_sequence(seq)
            if pid not in unique:
                unique[pid] = (seq, int(k.decode()))
            if max_records is not None and len(unique) >= max_records:
                break
            if (i + 1) % 20000 == 0:
                print(f"[precompute-pockets]   scan {i+1}  unique={len(unique)}  "
                      f"{time.time()-t0:.1f}s")
    print(f"[precompute-pockets] scan done: {len(unique)} unique pockets "
          f"in {time.time()-t0:.1f}s")

    seq_log = (out_dir / "sequences.tsv").open("a")
    to_do: list[tuple[str, str]] = []
    for pid, (seq, _idx) in unique.items():
        out_path = out_dir / f"{pid}.npy"
        if out_path.exists() and not force:
            continue
        to_do.append((pid, seq))
    print(f"[precompute-pockets] {len(to_do)} sequences to embed "
          f"(skipping {len(unique)-len(to_do)} already cached)")

    t1 = time.time()
    n_ok = 0
    # Group sequences by length to minimize padding waste. (Only a coarse sort
    # — HuggingFace's padding=True will pad to the batch max.)
    to_do.sort(key=lambda it: len(it[1]))
    for start in range(0, len(to_do), batch_size):
        chunk = to_do[start : start + batch_size]
        seqs = [s for _, s in chunk]
        embeds = encoder.encode_sequences_batch(seqs)  # [B, D]
        embeds_np = embeds.detach().cpu().float().numpy()
        for (pid, seq), emb in zip(chunk, embeds_np):
            np.save(str(out_dir / f"{pid}.npy"), emb)
            seq_log.write(f"{pid}\t{len(seq)}\t{seq}\n")
            n_ok += 1
        seq_log.flush()
        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t1
            rate = (start + len(chunk)) / max(elapsed, 1e-6)
            eta = (len(to_do) - start - len(chunk)) / max(rate, 1e-6)
            print(f"[precompute-pockets]   embed {start+len(chunk)}/{len(to_do)}  "
                  f"{rate:.1f} seq/s  eta {eta/60:.1f} min")
    seq_log.close()
    return n_ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--receptors", type=Path, help="LIT-PCBA-style receptor root")
    p.add_argument("--index-csv", type=Path, help="CSV with pocket_pdb,ligand_sdf,target_id")
    p.add_argument("--preprocessed-lmdb", type=Path,
                   help="TargetDiff-preprocessed CrossDocked LMDB (pocket10_processed)")
    p.add_argument("--out", type=Path, required=True, help="output dir for .npy + sequences.tsv")
    p.add_argument("--model", type=str, default=DEFAULT_ESM_MODEL)
    p.add_argument("--cutoff-a", type=float, default=DEFAULT_POCKET_CUTOFF_A)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32,
                   help="batch size for --preprocessed-lmdb mode")
    p.add_argument("--max-records", type=int, default=None,
                   help="cap on unique pockets (smoke-test aid)")
    p.add_argument("--force", action="store_true", help="recompute even if .npy exists")
    args = p.parse_args()

    n_sources = sum(x is not None for x in (args.receptors, args.index_csv, args.preprocessed_lmdb))
    if n_sources != 1:
        p.error("pass exactly one of --receptors, --index-csv, --preprocessed-lmdb")

    args.out.mkdir(parents=True, exist_ok=True)
    encoder = PocketEncoder(model_name=args.model, device=args.device, max_length=args.max_length)
    print(f"[precompute-pockets] device={args.device} model={args.model}")

    if args.preprocessed_lmdb is not None:
        n_ok = _run_preprocessed_lmdb(
            args.preprocessed_lmdb, args.out, encoder,
            batch_size=args.batch_size, force=args.force,
            max_records=args.max_records,
        )
        print(f"[precompute-pockets] done: {n_ok} new pockets -> {args.out}")
        return

    iterator = _iter_litpcba(args.receptors) if args.receptors else _iter_index_csv(args.index_csv)
    seq_log = (args.out / "sequences.tsv").open("a")
    n_ok = 0
    t0 = time.time()
    for tid, rec, lig in iterator:
        out_path = args.out / f"{tid}.npy"
        if out_path.exists() and not args.force:
            print(f"[precompute-pockets] {tid}: cached ({out_path.stat().st_size} bytes)")
            n_ok += 1
            continue
        try:
            residues = extract_pocket_residues(rec, lig, cutoff_a=args.cutoff_a)
            seq = residues_to_sequence(residues)
            embed = encoder.encode_sequence(seq)
            save_pocket_embed(out_path, embed)
            seq_log.write(f"{tid}\t{len(residues)}\t{seq}\n")
            seq_log.flush()
            n_ok += 1
            print(f"[precompute-pockets] {tid}: n_res={len(residues)} dim={embed.shape[-1]}")
        except Exception as e:
            print(f"[precompute-pockets] {tid}: FAILED ({e})", file=sys.stderr)

    seq_log.close()
    print(
        f"[precompute-pockets] done: {n_ok} targets in {time.time()-t0:.1f}s "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
