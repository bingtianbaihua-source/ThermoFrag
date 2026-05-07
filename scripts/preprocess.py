"""Run dataset preprocessing.

Usage:
    python scripts/preprocess.py --dataset spice \
        --in data/raw/spice/SPICE-1.1.4.hdf5 \
        --out data/processed/spice \
        --n-target 200000 --shard-size 20000 --val-frac 0.05 --seed 42

Output layout (``out_dir``):
    train/shard_*.npz  + train/manifest.json
    val/shard_*.npz    + val/manifest.json    (only if val_frac > 0)
    manifest.json                             (top-level summary)

See docs/DATA.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

from thermofrag.data.qmugs import (
    QMugsConformer,
    iter_qmugs_sdf_dir,
    save_qmugs_shard,
)
from thermofrag.data.spice import (
    DRUGLIKE_Z,
    SPICEConformer,
    iter_spice_hdf5,
    save_shard,
)


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_split(
    conformers: list[SPICEConformer],
    out_dir: Path,
    shard_size: int,
    energy_stats: dict,
    common_meta: dict,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for i in range(0, len(conformers), shard_size):
        chunk = conformers[i : i + shard_size]
        p = out_dir / f"shard_{i // shard_size:04d}.npz"
        save_shard(chunk, p)
        shard_paths.append(p)

    manifest = {
        **common_meta,
        "split_size": len(conformers),
        "energy_stats": energy_stats,
        "shards": [
            {"path": p.name, "sha256": _sha256(p), "bytes": p.stat().st_size} for p in shard_paths
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def preprocess_spice(
    in_path: Path,
    out_dir: Path,
    n_target: int = 200_000,
    shard_size: int = 20_000,
    max_heavy_atoms: int = 50,
    seed: int = 42,
    val_frac: float = 0.0,
) -> dict:
    """Filter SPICE HDF5 to drug-like conformers, split by *molecule* into
    train/val (no conformer leakage), and write NPZ shards to sub-directories.

    Reservoir sampling produces an unbiased size-``n_target`` sample in a
    single streaming pass. Splitting is done at the SMILES level so all
    conformers of a molecule stay on the same side of the split; this is the
    statistically honest way to measure generalization on QM data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    reservoir: list[SPICEConformer] = []
    n_seen = 0
    n_kept_mols = 0
    last_mol = None

    progress = tqdm(
        iter_spice_hdf5(in_path, elements_allowed=DRUGLIKE_Z, max_heavy_atoms=max_heavy_atoms),
        desc="SPICE conformers",
        unit="conf",
        mininterval=2.0,
    )
    for conf in progress:
        if conf.smiles != last_mol:
            n_kept_mols += 1
            last_mol = conf.smiles
        if len(reservoir) < n_target:
            reservoir.append(conf)
        else:
            j = rng.randint(0, n_seen)
            if j < n_target:
                reservoir[j] = conf
        n_seen += 1

    # Group by SMILES so a whole molecule lands on one side.
    by_mol: dict[str, list[SPICEConformer]] = defaultdict(list)
    for c in reservoir:
        by_mol[c.smiles].append(c)
    mol_keys = list(by_mol.keys())
    rng.shuffle(mol_keys)

    n_val_target = int(round(val_frac * len(reservoir)))
    val_conformers: list[SPICEConformer] = []
    train_conformers: list[SPICEConformer] = []
    for k in mol_keys:
        if len(val_conformers) < n_val_target:
            val_conformers.extend(by_mol[k])
        else:
            train_conformers.extend(by_mol[k])
    rng.shuffle(train_conformers)
    rng.shuffle(val_conformers)

    # Energy + force stats come from the train split only; val inherits the same stats.
    e_train = np.asarray([c.energy for c in train_conformers], dtype=np.float64)
    energy_stats = {
        "mean": float(e_train.mean()) if e_train.size else 0.0,
        "std": float(e_train.std() + 1e-9) if e_train.size else 1.0,
    }
    # Per-component std of all atomic forces in the train set. A single scalar keeps
    # the normalization unambiguous across datasets; using the component (not vector
    # magnitude) std matches the loss which is MSE over vector components.
    if train_conformers:
        f_sq_sum = 0.0
        f_count = 0
        for c in train_conformers:
            f_sq_sum += float((c.forces ** 2).sum())
            f_count += int(c.forces.size)
        force_std = float((f_sq_sum / max(f_count, 1)) ** 0.5 + 1e-9)
    else:
        force_std = 1.0
    energy_stats["force_std"] = force_std

    common_meta = {
        "dataset": "SPICE",
        "source_file": str(in_path),
        "source_sha256": _sha256(in_path) if in_path.is_file() else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filter": {
            "elements_allowed_z": sorted(DRUGLIKE_Z),
            "max_heavy_atoms": max_heavy_atoms,
            "use_formation_energy": True,
        },
        "sampling": {
            "n_seen": n_seen,
            "n_target": n_target,
            "n_kept": len(reservoir),
            "seed": seed,
            "val_frac": val_frac,
        },
        "kept_unique_molecules_approx": n_kept_mols,
        "units": {
            "energy": "kcal/mol",
            "forces": "kcal/mol/angstrom",
            "pos": "angstrom",
        },
    }

    train_manifest = _write_split(
        train_conformers, out_dir / "train", shard_size, energy_stats, {**common_meta, "split": "train"}
    )
    val_manifest = None
    if val_conformers:
        val_manifest = _write_split(
            val_conformers, out_dir / "val", shard_size, energy_stats, {**common_meta, "split": "val"}
        )

    top = {
        **common_meta,
        "energy_stats": energy_stats,
        "splits": {
            "train": {"n": len(train_conformers), "dir": "train"},
            "val": {"n": len(val_conformers), "dir": "val"} if val_conformers else None,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(top, indent=2))
    return top


def preprocess_qmugs(
    in_dir: Path,
    out_dir: Path,
    n_target: int = 50_000,
    shard_size: int = 5_000,
    max_heavy_atoms: int = 50,
    energy_key: str = "DFT:TOTAL_ENERGY",
    seed: int = 42,
) -> dict:
    """Reservoir-sample one conformer per distinct SMILES from a QMugs SDF tree.

    QMugs is the held-out benchmark for C1 (energy MAE / Spearman). We take at
    most one conformer per molecule so the evaluation set is not inflated by
    near-duplicate conformer geometries — each sample is an independent
    draw from the molecule distribution.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    seen_smiles: set[str] = set()
    reservoir: list[QMugsConformer] = []
    n_seen = 0
    n_parse_fail = 0
    last_progress_update = 0

    progress = tqdm(
        iter_qmugs_sdf_dir(
            in_dir,
            energy_key=energy_key,
            max_heavy_atoms=max_heavy_atoms,
        ),
        desc="QMugs conformers",
        unit="conf",
        mininterval=2.0,
    )
    for conf in progress:
        # One conformer per molecule.
        if conf.smiles in seen_smiles:
            continue
        seen_smiles.add(conf.smiles)

        if len(reservoir) < n_target:
            reservoir.append(conf)
        else:
            j = rng.randint(0, n_seen)
            if j < n_target:
                reservoir[j] = conf
        n_seen += 1

    rng.shuffle(reservoir)

    shard_paths = []
    for i in range(0, len(reservoir), shard_size):
        chunk = reservoir[i : i + shard_size]
        p = out_dir / f"shard_{i // shard_size:04d}.npz"
        save_qmugs_shard(chunk, p)
        shard_paths.append(p)

    e_all = np.asarray([c.energy for c in reservoir], dtype=np.float64)
    energy_mean = float(e_all.mean()) if e_all.size else 0.0
    energy_std = float(e_all.std() + 1e-9) if e_all.size else 1.0

    manifest = {
        "dataset": "QMugs",
        "source_dir": str(in_dir),
        "energy_key": energy_key,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filter": {
            "elements_allowed_z": sorted(int(z) for z in {1, 6, 7, 8, 9, 15, 16, 17, 35}),
            "max_heavy_atoms": max_heavy_atoms,
            "one_conformer_per_molecule": True,
        },
        "sampling": {
            "n_seen_unique_mols": n_seen,
            "n_target": n_target,
            "n_kept": len(reservoir),
            "seed": seed,
        },
        "units": {"energy": "kcal/mol", "pos": "angstrom"},
        "energy_stats": {"mean": energy_mean, "std": energy_std},
        "shards": [
            {"path": p.name, "sha256": _sha256(p), "bytes": p.stat().st_size} for p in shard_paths
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["spice", "qmugs", "zinc", "chembl", "litpcba"])
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="input file or dir")
    p.add_argument("--out", dest="out_dir", type=Path, required=True, help="output dir")
    p.add_argument("--n-target", type=int, default=None,
                   help="number of samples to retain (dataset-specific default)")
    p.add_argument("--shard-size", type=int, default=None)
    p.add_argument("--max-heavy-atoms", type=int, default=50)
    p.add_argument("--val-frac", type=float, default=0.0, help="SPICE: fraction held out for validation (by molecule)")
    p.add_argument("--energy-key", type=str, default="DFT:TOTAL_ENERGY",
                   help="QMugs: SDF property key used as the target energy")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.dataset == "spice":
        n_target = args.n_target if args.n_target is not None else 200_000
        shard_size = args.shard_size if args.shard_size is not None else 20_000
        mf = preprocess_spice(
            args.in_path,
            args.out_dir,
            n_target=n_target,
            shard_size=shard_size,
            max_heavy_atoms=args.max_heavy_atoms,
            seed=args.seed,
            val_frac=args.val_frac,
        )
    elif args.dataset == "qmugs":
        n_target = args.n_target if args.n_target is not None else 50_000
        shard_size = args.shard_size if args.shard_size is not None else 5_000
        mf = preprocess_qmugs(
            args.in_path,
            args.out_dir,
            n_target=n_target,
            shard_size=shard_size,
            max_heavy_atoms=args.max_heavy_atoms,
            energy_key=args.energy_key,
            seed=args.seed,
        )
    else:
        raise SystemExit(f"preprocessor for '{args.dataset}' not implemented yet")
    print(json.dumps(mf, indent=2))


if __name__ == "__main__":
    main()
