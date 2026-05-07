"""BBAR baseline sampler for C3 / C4 generator-vs-generator arm.

Per BASELINES.md §2. BBAR has no pocket conditioning, so the fair protocol is
to generate a single pool per property-condition and dock the same pool
against every pocket. We use ``{logp, qed}`` because those are the two
conditions for which ThermoFrag already has LIT-PCBA target y-vectors; we set
each condition's target value to the aggregated mean of the 15 target vectors
(computed from ``results/eval/phase4/litpcba_targets/<t>/y_raw.npy``).

Output::

    results/eval/phase4_baselines/bbar/
        decoded/<target>.parquet    # target, chain_idx, smiles, condition
        manifest.json

Each of the 15 target parquets holds the same 2 × 1000 row pool — this is by
design and is documented both here and in docs/BASELINES.md §2.

Entry point: run inside the ``bbar`` conda env (python 3.11 + torch 2.3.1 +
torch-geometric 2.4.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

logger = logging.getLogger("sample_bbar")

THERMOFRAG_ROOT = Path(__file__).resolve().parents[1]
BBAR_ROOT = THERMOFRAG_ROOT / "vendor" / "bbar_upstream"

DEFAULT_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA",   "IDH1",  "KAT2A",   "MAPK1",     "MTORC1",
    "OPRK1", "PKM2",  "PPARG",   "TP53",      "VDR",
]

# Aggregated mean across the 15 LIT-PCBA target y_raw vectors (phi order:
# logP, qed, sa, tpsa, mw, hba, hbd, rotb). See scripts/sample_bbar.py header.
DEFAULT_CONDITIONS = {
    "logp": 3.7289,
    "qed":  0.6086,
}

CONDITION_TO_CONFIG = {
    "logp": "test/generation_config/logp.yaml",
    "qed":  "test/generation_config/qed.yaml",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize(smi: str):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _build_scaffold_list(generator):
    """Mirror upstream ``script/sample_denovo.py`` scaffold prep: every building
    block in the library (with attachment atom stripped) is a scaffold that can
    be used as a starting seed."""
    from rdkit import Chem

    scaffolds: list[str] = []
    for rdmol in generator.library.rdmol_list:
        rwmol = Chem.RWMol(rdmol)
        try:
            rwmol.RemoveAtom(0)
            rwmol.UpdatePropertyCache()
            mol = rwmol.GetMol()
            assert mol is not None
            Chem.SanitizeMol(mol)
            if mol.GetNumAtoms() <= 1:
                continue
            scaffolds.append(Chem.MolToSmiles(mol))
        except Exception:
            continue
    return scaffolds


def sample_pool(condition_name: str, condition_value: float, n_request: int,
                n_keep: int, seed: int, log_every: int = 250):
    """Build generator for ``condition_name`` and return up to ``n_keep``
    unique canonical SMILES plus a stats dict."""
    from omegaconf import OmegaConf
    from rdkit import Chem
    from bbar.generate import MoleculeBuilder
    from utils.seed import set_seed

    cfg_path = CONDITION_TO_CONFIG[condition_name]
    cfg = OmegaConf.load(cfg_path)

    t0 = time.time()
    generator = MoleculeBuilder(cfg)
    build_s = time.time() - t0

    target_props = list(generator.target_properties)
    assert target_props == [condition_name], (
        f"expected generator to want [{condition_name}] but got {target_props}"
    )
    condition = {condition_name: float(condition_value)}

    scaffolds = _build_scaffold_list(generator)
    logger.info(
        "%s: generator built in %.1fs, n_scaffolds=%d, generating up to %d",
        condition_name, build_s, len(scaffolds), n_request,
    )

    rnd = random.Random(seed)
    seen: set[str] = set()
    kept: list[str] = []
    n_fail = 0
    n_invalid = 0
    n_dup = 0

    t0 = time.time()
    for i in range(n_request):
        s_seed = seed + i
        set_seed(s_seed)
        scaffold_smi = rnd.choice(scaffolds)
        scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
        out_mol = generator.generate(scaffold_mol, condition)
        if out_mol is None:
            n_fail += 1
            continue
        raw_smi = Chem.MolToSmiles(out_mol)
        can = _canonicalize(raw_smi)
        if can is None:
            n_invalid += 1
            continue
        if can in seen:
            n_dup += 1
            continue
        seen.add(can)
        kept.append(can)
        if len(kept) >= n_keep:
            break
        if (i + 1) % log_every == 0:
            logger.info(
                "%s: request=%d kept=%d dup=%d invalid=%d fail=%d (%.1fs)",
                condition_name, i + 1, len(kept), n_dup, n_invalid, n_fail,
                time.time() - t0,
            )
    wall = time.time() - t0

    stats = {
        "condition": condition_name,
        "condition_value": float(condition_value),
        "n_requested": n_request,
        "n_generator_fail": n_fail,
        "n_invalid": n_invalid,
        "n_duplicate": n_dup,
        "n_kept": len(kept),
        "seconds": round(wall, 2),
    }
    return kept, stats


def write_decoded_parquet(pools: dict[str, list[str]], target: str,
                          out_pq: Path) -> None:
    """Write a single target parquet with all pools concatenated, one row per
    molecule. Columns: target, chain_idx, smiles, condition."""
    import pandas as pd

    rows: list[dict] = []
    chain_idx = 0
    for cond_name, smis in pools.items():
        for smi in smis:
            rows.append({
                "target":    target,
                "chain_idx": chain_idx,
                "smiles":    smi,
                "condition": cond_name,
            })
            chain_idx += 1
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_pq, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path,
                   default=THERMOFRAG_ROOT / "results" / "eval"
                           / "phase4_baselines" / "bbar")
    p.add_argument("--targets", nargs="*", default=None,
                   help="Subset of targets (default: all 15)")
    p.add_argument("--n-request", type=int, default=1500,
                   help="Upstream samples per condition "
                        "(trimmed to --n-keep unique canonical)")
    p.add_argument("--n-keep", type=int, default=1000,
                   help="Unique canonical SMILES to keep per condition")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--conditions", type=json.loads,
                   default=json.dumps(DEFAULT_CONDITIONS),
                   help='Override conditions as JSON, e.g. \'{"logp": 3.7289, "qed": 0.6086}\'')
    p.add_argument("--force", action="store_true",
                   help="Re-sample even if target parquet already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # BBAR's configs use ./test/... and ./data/... relative paths, so cd in.
    os.chdir(BBAR_ROOT)
    sys.path.insert(0, str(BBAR_ROOT / "script"))  # for utils.seed.set_seed

    targets = args.targets or DEFAULT_TARGETS
    decoded_dir = args.out_dir / "decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    # If all targets already exist and not --force, nothing to do.
    if not args.force:
        existing = [t for t in targets if (decoded_dir / f"{t}.parquet").exists()]
        if len(existing) == len(targets):
            logger.info("All %d target parquets already exist; use --force to resample", len(targets))
            return

    conditions: dict[str, float] = args.conditions
    for cond_name in conditions:
        if cond_name not in CONDITION_TO_CONFIG:
            raise SystemExit(f"Unknown condition '{cond_name}' "
                             f"(known: {list(CONDITION_TO_CONFIG)})")

    # Sample one pool per condition.
    pools: dict[str, list[str]] = {}
    pool_stats: dict[str, dict] = {}
    ckpt_meta: dict[str, dict] = {}
    total_t0 = time.time()
    for cond_name, cond_val in conditions.items():
        kept, stats = sample_pool(
            condition_name=cond_name,
            condition_value=cond_val,
            n_request=args.n_request,
            n_keep=args.n_keep,
            seed=args.seed,
        )
        pools[cond_name] = kept
        pool_stats[cond_name] = stats
        ckpt_path = BBAR_ROOT / "test" / "pretrained_model" / f"{cond_name}.tar"
        ckpt_meta[cond_name] = {
            "path": str(ckpt_path),
            "sha256": _sha256(ckpt_path),
            "condition_value": float(cond_val),
        }
        logger.info(
            "%s pool: kept %d / requested %d in %.1fs",
            cond_name, stats["n_kept"], stats["n_requested"], stats["seconds"],
        )

    # Replicate the combined pool across all 15 target parquets.
    per_target: dict[str, dict] = {}
    for target in targets:
        out_pq = decoded_dir / f"{target}.parquet"
        if out_pq.exists() and not args.force:
            logger.info("%s: parquet exists, skip", target)
            continue
        write_decoded_parquet(pools, target, out_pq)
        n_total = sum(len(v) for v in pools.values())
        per_target[target] = {
            "n_rows": n_total,
            "n_per_condition": {k: len(v) for k, v in pools.items()},
        }

    total_wall = time.time() - total_t0

    manifest = {
        "baseline": "bbar",
        "note": (
            "BBAR has no pocket input; the same SMILES pool is scored against "
            "all 15 pockets. Pairing is fair because both BBAR and ThermoFrag "
            "are pocket-agnostic at generation time (pocket enters only at "
            "Vina)."
        ),
        "checkpoints": ckpt_meta,
        "conditions": {k: float(v) for k, v in conditions.items()},
        "condition_values_rationale": (
            "Aggregated mean of logP and QED across the 15 LIT-PCBA y_raw "
            "vectors (results/eval/phase4/litpcba_targets/<t>/y_raw.npy, "
            "phi order: logP, qed, sa, tpsa, mw, hba, hbd, rotb)."
        ),
        "n_pool_per_condition": args.n_keep,
        "sampling": {
            "n_request_per_condition": args.n_request,
            "n_keep_per_condition": args.n_keep,
            "seed": args.seed,
            "window_size": 2000,
            "alpha": 0.75,
            "max_iteration": 10,
        },
        "library": {
            "path": str(BBAR_ROOT / "data" / "ZINC" / "library.csv"),
            "n_blocks": _count_lines(BBAR_ROOT / "data" / "ZINC" / "library.csv") - 1,
        },
        "per_condition_stats": pool_stats,
        "per_target": per_target,
        "bbar_git_sha": _try_git_sha(BBAR_ROOT),
        "wall_clock_seconds": round(total_wall, 2),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest → %s", manifest_path)


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _try_git_sha(repo: Path):
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        )
        return out.strip()
    except Exception:
        return None


if __name__ == "__main__":
    main()
