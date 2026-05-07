"""RxnFlow baseline sampler for C3 / C4 generator-vs-generator arm.

Per BASELINES.md §1. Loads the pocket-conditional RxnFlow checkpoint
``qvina-unif-0-64`` and generates molecules for each LIT-PCBA target, then writes
the same parquet artefacts ThermoFrag already produces under a parallel tree.

Output::

    results/eval/phase4_baselines/rxnflow/
        decoded/<target>.parquet    # target, chain_idx, smiles
        manifest.json               # model, env_dir, weight sha256, wall-clock

Note on env_dir substitution:
The pretrained ``qvina-unif-0-64`` weight was trained on the Enamine Comprehensive
Catalog (~1M blocks), which is only available under a commercial request from
Enamine. We substitute the public ZINCFrag-200k building-block library for
reproducibility. This is a controlled deviation that is recorded in the
manifest; expect a small quality hit versus the paper numbers.

Entry point: run inside the ``rxnflow`` conda env (python 3.12 + torch 2.5.1 +
pmnet-appl). ThermoFrag's own env cannot import rxnflow due to python/torch
version pins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("sample_rxnflow")

THERMOFRAG_ROOT = Path(__file__).resolve().parents[1]
RXNFLOW_ROOT = THERMOFRAG_ROOT / "vendor" / "rxnflow"
DEFAULT_TARGETS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1",
    "GBA",   "IDH1",  "KAT2A",   "MAPK1",     "MTORC1",
    "OPRK1", "PKM2",  "PPARG",   "TP53",      "VDR",
]


def _parse_temperature(s: str) -> tuple[str, list[float]]:
    parts = s.split("-")
    dist = parts[0]
    params = [float(x) for x in parts[1:]]
    assert dist in ("constant", "uniform", "loguniform", "gamma", "beta"), dist
    return dist, params


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonicalize(smi: str) -> Optional[str]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def build_sampler(model_path: str, env_dir: Path, subsampling: float,
                  temperature: str, device: str, seed: int,
                  num_from_policy: int = 100):
    """Instantiate a ProxySampler mirroring ``scripts/sampling_zeroshot.py``."""
    from rxnflow.config import Config, init_empty
    from rxnflow.tasks.multi_pocket import ProxySampler
    from rxnflow.utils.download import download_pretrained_weight

    cfg = init_empty(Config())
    cfg.seed = seed
    cfg.env_dir = str(env_dir)
    cfg.algo.num_from_policy = num_from_policy
    cfg.algo.action_subsampling.sampling_ratio = subsampling

    ckpt = download_pretrained_weight(model_path)
    sampler = ProxySampler(cfg, ckpt, device)
    dist, dparams = _parse_temperature(temperature)
    sampler.update_temperature(dist, dparams)
    return sampler, ckpt


def sample_one_target(sampler, target: str, rec_dir: Path, n_request: int,
                      n_keep: int) -> tuple[list[str], dict]:
    """Sample ``n_request`` molecules, keep the first ``n_keep`` valid canonical
    SMILES. Returns the canonical list and a stats dict."""
    import json as _json

    protein_path = rec_dir / "receptor_clean.pdb"
    if not protein_path.exists():
        raise FileNotFoundError(protein_path)
    box = _json.loads((rec_dir / "box.json").read_text())
    center = tuple(float(x) for x in box["center"])

    t0 = time.time()
    sampler.set_pocket(str(protein_path), center, None)
    raw = sampler.sample(n_request, calc_reward=False)
    t_sample = time.time() - t0

    seen: set[str] = set()
    canonical: list[str] = []
    invalid = 0
    dup = 0
    for item in raw:
        smi = item.get("smiles")
        if not smi:
            invalid += 1
            continue
        can = _canonicalize(smi)
        if can is None:
            invalid += 1
            continue
        if can in seen:
            dup += 1
            continue
        seen.add(can)
        canonical.append(can)
        if len(canonical) >= n_keep:
            break

    stats = {
        "n_requested": n_request,
        "n_returned": len(raw),
        "n_invalid": invalid,
        "n_duplicate": dup,
        "n_kept": len(canonical),
        "seconds": round(t_sample, 2),
    }
    return canonical, stats


def write_decoded_parquet(smiles: list[str], target: str, out_pq: Path) -> None:
    import pandas as pd

    out_pq.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "target":    [target] * len(smiles),
        "chain_idx": list(range(len(smiles))),
        "smiles":    smiles,
    })
    df.to_parquet(out_pq, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--receptors", type=Path,
                   default=THERMOFRAG_ROOT / "data" / "external" / "receptors")
    p.add_argument("--out-dir", type=Path,
                   default=THERMOFRAG_ROOT / "results" / "eval"
                             / "phase4_baselines" / "rxnflow")
    p.add_argument("--env-dir", type=Path,
                   default=RXNFLOW_ROOT / "data" / "envs" / "zincfrag",
                   help="Pre-built RxnFlow env_dir (see vendor/rxnflow/data/README.md)")
    p.add_argument("--model", type=str, default="qvina-unif-0-64")
    p.add_argument("--temperature", type=str, default="uniform-16-64")
    p.add_argument("--subsampling", type=float, default=0.1)
    p.add_argument("--n-request", type=int, default=1200,
                   help="Upstream samples per target (will be trimmed to --n-keep valid canonical)")
    p.add_argument("--n-keep", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--targets", nargs="*", default=None,
                   help="Subset of targets (default: all 15)")
    p.add_argument("--force", action="store_true",
                   help="Re-sample even if target parquet already exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # RxnFlow caches pretrained weights + env_dir relative to cwd, so run from
    # the vendor root.
    os.chdir(RXNFLOW_ROOT)
    sys.path.insert(0, str(RXNFLOW_ROOT / "src"))

    if not args.env_dir.exists():
        raise SystemExit(
            f"env_dir {args.env_dir} not found. Build via "
            f"vendor/rxnflow/scripts/b_create_env.py first."
        )

    targets = args.targets or DEFAULT_TARGETS
    decoded_dir = args.out_dir / "decoded"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building sampler (device=%s, model=%s)", args.device, args.model)
    sampler, ckpt_path = build_sampler(
        model_path=args.model,
        env_dir=args.env_dir,
        subsampling=args.subsampling,
        temperature=args.temperature,
        device=args.device,
        seed=args.seed,
    )
    ckpt_sha = _sha256(Path(ckpt_path))

    per_target: dict[str, dict] = {}
    total_t0 = time.time()
    for target in targets:
        out_pq = decoded_dir / f"{target}.parquet"
        if out_pq.exists() and not args.force:
            logger.info("%s: parquet exists, skip (use --force to resample)", target)
            continue
        rec_dir = args.receptors / target
        if not rec_dir.exists():
            logger.error("%s: receptor dir missing %s", target, rec_dir)
            continue
        try:
            smiles, stats = sample_one_target(
                sampler, target, rec_dir,
                n_request=args.n_request, n_keep=args.n_keep,
            )
        except Exception as exc:
            logger.exception("%s: sampling failed: %s", target, exc)
            per_target[target] = {"error": str(exc)}
            continue
        write_decoded_parquet(smiles, target, out_pq)
        per_target[target] = stats
        logger.info(
            "%s: kept %d / %d valid canonical smiles in %.1fs",
            target, stats["n_kept"], stats["n_requested"], stats["seconds"],
        )

    total_wall = time.time() - total_t0

    manifest = {
        "baseline": "rxnflow",
        "model": args.model,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "env_dir": str(args.env_dir),
        "env_dir_note": (
            "ZINCFrag-200k substituted for the Enamine Catalog that the upstream "
            "qvina-unif-0-64 was trained on (Enamine requires commercial request). "
            "Documented as controlled deviation for reproducibility."
        ),
        "sampling": {
            "temperature": args.temperature,
            "subsampling_ratio": args.subsampling,
            "n_request_per_target": args.n_request,
            "n_keep_per_target": args.n_keep,
            "device": args.device,
            "seed": args.seed,
        },
        "rxnflow_git_sha": _try_git_sha(RXNFLOW_ROOT),
        "per_target": per_target,
        "wall_clock_seconds": round(total_wall, 2),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest → %s", manifest_path)


def _try_git_sha(repo: Path) -> Optional[str]:
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
