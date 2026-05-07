#!/usr/bin/env python
"""Smina (Vinardo) and GNINA (CNN) cross-rescoring of generator poses.

Spec: ``docs/validation/02_multi_scoring.md``.

For each (generator, target, chain_idx in top-K), re-score the existing
docked pose with Smina (Vinardo scoring) and/or GNINA (CNN scoring).
Outputs a parquet per (generator, target) with columns:

  chain_idx, smiles, vina_score,
  smina_score, gnina_affinity, gnina_cnn_score, gnina_cnn_affinity,
  status

Both rescorers run in ``--score_only`` mode — we are auditing whether the
ranking is invariant under scoring-function choice, not re-searching the
pose.

CLI:

  --tool smina | gnina | both
  --top_k     int = 30
  --gpu_id    int = 0           (only used by gnina)
  --n_workers int = 8           (only used by smina)

Idempotent: skips (generator, target) if the per-target parquet exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

def _find_smina() -> Path:
    """Locate the smina binary. Prefer the conda-shipped one in tf-eval;
    fall back to the static binary in ``vendor/smina/``.
    """
    candidates = [
        Path("/home/zhao/miniconda3/envs/tf-eval/bin/smina"),
        Path("/home/zhao/code/ThermoFrag/vendor/smina/smina.static"),
    ]
    for c in candidates:
        if c.exists() and os.access(c, os.X_OK):
            return c
    return candidates[0]


SMINA_BIN = _find_smina()
GNINA_BIN_CANDIDATES = [
    Path("/home/zhao/code/ThermoFrag/vendor/gnina/gnina"),
    Path("/home/zhao/miniconda3/envs/tf-gnina/bin/gnina"),
    Path("/usr/local/bin/gnina"),
]

TARGETS_15 = ["ADRB2", "ALDH1", "ESR_ago", "ESR_antago", "FEN1", "GBA",
              "IDH1", "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG",
              "TP53", "VDR"]
GENERATORS_4 = ["thermofrag", "targetdiff", "rxnflow", "bbar"]


def _find_gnina() -> Path | None:
    for p in GNINA_BIN_CANDIDATES:
        if p.exists() and os.access(p, os.X_OK):
            return p
    return None


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Smina rescoring
# ---------------------------------------------------------------------------


SMINA_AFFINITY_RE = re.compile(r"Affinity:\s*([-+\d.eE]+)")


def smina_score_one(receptor_pdbqt: Path, pose_pdb: Path) -> dict:
    """Run smina --score_only --scoring vinardo on one pose."""
    if not SMINA_BIN.exists():
        return {"status": "smina_missing"}
    cmd = [
        str(SMINA_BIN),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(pose_pdb),
        "--score_only",
        "--scoring", "vinardo",
        "--autobox_ligand", str(pose_pdb),
        "--cpu", "1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"status": "smina_timeout"}
    if proc.returncode != 0 and not proc.stdout:
        return {"status": "smina_failed",
                "stderr": proc.stderr[:200]}
    m = SMINA_AFFINITY_RE.search(proc.stdout or "")
    if not m:
        return {"status": "smina_parse_failed"}
    return {"status": "ok", "smina_score": float(m.group(1))}


# ---------------------------------------------------------------------------
# GNINA rescoring
# ---------------------------------------------------------------------------


GNINA_AFFINITY_RE = re.compile(r"Affinity:\s*([-+\d.eE]+)")
GNINA_CNN_SCORE_RE = re.compile(r"CNNscore:\s*([-+\d.eE]+)")
GNINA_CNN_AFFINITY_RE = re.compile(r"CNNaffinity:\s*([-+\d.eE]+)")


def gnina_score_one(gnina_bin: Path, receptor_pdbqt: Path,
                    pose_pdb: Path, gpu_id: int = 0) -> dict:
    cmd = [
        str(gnina_bin),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(pose_pdb),
        "--score_only",
        "--cnn_scoring", "rescore",
        "--cnn", "dense_ensemble",
        "--gpu", "--gpu_id", str(gpu_id),
        "--no_gpu", "false",  # tolerate older flag names
    ]
    # The flag set above is the union of common GNINA versions; let
    # subprocess fail loudly if the binary rejects it.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)})
    except subprocess.TimeoutExpired:
        return {"status": "gnina_timeout"}
    if proc.returncode != 0:
        return {"status": "gnina_failed", "stderr": proc.stderr[:200]}
    out = proc.stdout or ""
    aff = GNINA_AFFINITY_RE.search(out)
    cnn = GNINA_CNN_SCORE_RE.search(out)
    cnn_aff = GNINA_CNN_AFFINITY_RE.search(out)
    if not aff:
        return {"status": "gnina_parse_failed"}
    return {
        "status": "ok",
        "gnina_affinity": float(aff.group(1)),
        "gnina_cnn_score": float(cnn.group(1)) if cnn else None,
        "gnina_cnn_affinity": float(cnn_aff.group(1)) if cnn_aff else None,
    }


# ---------------------------------------------------------------------------
# Per-target driver
# ---------------------------------------------------------------------------


def process_target(generator: str, target: str, args: argparse.Namespace,
                   gnina_bin: Path | None) -> Path:
    out_dir = args.out_root / generator
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"{target}.parquet"
    if out_parquet.exists() and not args.force:
        print(f"[skip] {generator}/{target}: {out_parquet.name} exists")
        return out_parquet

    rec_dir = args.repo_root / "data" / "external" / "receptors" / target
    receptor_pdbqt = rec_dir / "receptor.pdbqt"

    pose_dir = args.poses_root / generator / target
    manifest_path = pose_dir / "manifest.parquet"
    if not manifest_path.exists():
        print(f"[skip] {generator}/{target}: no pose manifest")
        return out_parquet
    manifest = pd.read_parquet(manifest_path)
    manifest_ok = (manifest[manifest["status"] == "ok"]
                   .sort_values("vina_score").head(args.top_k))

    rows = []
    smina_jobs = []
    if args.tool in ("smina", "both"):
        # Submit smina jobs in parallel via process pool
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {}
            for _, m in manifest_ok.iterrows():
                pose_pdb = pose_dir / f"{int(m['chain_idx'])}.pdb"
                if not pose_pdb.exists():
                    rows.append({"chain_idx": int(m["chain_idx"]),
                                 "smiles": m["smiles"],
                                 "vina_score": float(m["vina_score"]),
                                 "status": "no_pose"})
                    continue
                futures[pool.submit(smina_score_one, receptor_pdbqt, pose_pdb)] = m
            for fut in as_completed(futures):
                m = futures[fut]
                res = fut.result()
                rows.append({"chain_idx": int(m["chain_idx"]),
                             "smiles": m["smiles"],
                             "vina_score": float(m["vina_score"]),
                             **res})

    if args.tool in ("gnina", "both"):
        if gnina_bin is None:
            print(f"[warn] gnina binary not found; skipping CNN rescoring")
        else:
            # GNINA serial on one GPU
            for _, m in manifest_ok.iterrows():
                pose_pdb = pose_dir / f"{int(m['chain_idx'])}.pdb"
                if not pose_pdb.exists():
                    continue
                res = gnina_score_one(gnina_bin, receptor_pdbqt, pose_pdb,
                                      gpu_id=args.gpu_id)
                # Merge into existing row if smina ran first, else append
                merged = False
                for row in rows:
                    if row["chain_idx"] == int(m["chain_idx"]):
                        row.update({k: v for k, v in res.items()
                                    if k != "status"})
                        # Keep status='ok' if either succeeded
                        if row.get("status") == "ok" or res.get("status") == "ok":
                            row["status"] = "ok"
                        merged = True
                        break
                if not merged:
                    rows.append({"chain_idx": int(m["chain_idx"]),
                                 "smiles": m["smiles"],
                                 "vina_score": float(m["vina_score"]),
                                 **res})

    df = pd.DataFrame(rows).sort_values("vina_score").reset_index(drop=True)
    df.to_parquet(out_parquet, index=False)
    n_ok = (df["status"] == "ok").sum()
    print(f"[done] {generator}/{target}: {n_ok}/{len(df)} ok -> {out_parquet}")
    return out_parquet


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def aggregate(args, out_root: Path) -> dict:
    from scipy.stats import spearmanr, wilcoxon

    rows_all = []
    for gen in GENERATORS_4:
        for tgt in TARGETS_15:
            p = out_root / gen / f"{tgt}.parquet"
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["generator"] = gen
            df["target"] = tgt
            rows_all.append(df)
    if not rows_all:
        return {"task_id": "02_multi_scoring", "n_files": 0,
                "notes": "no per-(gen,target) parquets found"}
    full = pd.concat(rows_all, ignore_index=True)
    ok = full[full["status"] == "ok"]

    summary = {
        "task_id": "02_multi_scoring",
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "n_rows": int(len(full)),
        "n_ok": int(len(ok)),
    }

    # Sanity: cross-scorer Spearman correlations on pooled top-K
    thresholds = {}
    if "smina_score" in ok.columns and ok["smina_score"].notna().any():
        sub = ok.dropna(subset=["smina_score", "vina_score"])
        if len(sub) > 5:
            r, _ = spearmanr(sub["smina_score"], sub["vina_score"])
            thresholds["spearman_smina_vina"] = {
                "target": ">= 0.7",
                "observed": round(float(r), 3),
                "pass": bool(r >= 0.7),
            }
    if "gnina_affinity" in ok.columns and ok["gnina_affinity"].notna().any():
        sub = ok.dropna(subset=["gnina_affinity", "vina_score"])
        if len(sub) > 5:
            r, _ = spearmanr(sub["gnina_affinity"], sub["vina_score"])
            thresholds["spearman_gnina_vina"] = {
                "target": ">= 0.4",
                "observed": round(float(r), 3),
                "pass": bool(r >= 0.4),
            }

    # TF beats TargetDiff under each scorer (paired Wilcoxon per target)
    def _per_target_win(score_col: str) -> dict:
        wins = 0
        targets_evaluated = 0
        per_target = {}
        for tgt in sorted(ok["target"].unique()):
            tf = ok[(ok["target"] == tgt) & (ok["generator"] == "thermofrag")][score_col].dropna().values
            td = ok[(ok["target"] == tgt) & (ok["generator"] == "targetdiff")][score_col].dropna().values
            n = min(len(tf), len(td), 10)
            if n < 5:
                continue
            tf = sorted(tf)[:n]
            td = sorted(td)[:n]
            try:
                stat, p = wilcoxon(tf, td)
            except Exception:
                continue
            targets_evaluated += 1
            tf_better = sum(1 for a, b in zip(tf, td) if a < b)
            if tf_better > n / 2 and p < 0.05:
                wins += 1
            per_target[tgt] = {"tf_top10_mean": float(sum(tf) / n),
                               "td_top10_mean": float(sum(td) / n),
                               "wilcoxon_p": float(p),
                               "tf_wins": bool(tf_better > n / 2 and p < 0.05)}
        return {"wins": wins, "n_evaluated": targets_evaluated,
                "per_target": per_target}

    if "smina_score" in ok.columns:
        sm = _per_target_win("smina_score")
        thresholds["tf_vs_td_smina"] = {
            "target": ">= 10/15 sig wins",
            "observed": f"{sm['wins']}/{sm['n_evaluated']}",
            "pass": bool(sm["wins"] >= 10),
        }
        summary["smina_per_target"] = sm["per_target"]
    if "gnina_affinity" in ok.columns:
        gn = _per_target_win("gnina_affinity")
        thresholds["tf_vs_td_gnina"] = {
            "target": ">= 10/15 sig wins",
            "observed": f"{gn['wins']}/{gn['n_evaluated']}",
            "pass": bool(gn["wins"] >= 10),
        }
        summary["gnina_per_target"] = gn["per_target"]

    summary["thresholds"] = thresholds

    print()
    print("=" * 70)
    print("Task 2 (multi-scoring) summary")
    print("=" * 70)
    for name, t in thresholds.items():
        sym = "PASS" if t["pass"] else "FAIL"
        print(f"  {name:30s}  target={t['target']!s:20s}  observed={t['observed']!s:14s}  {sym}")
    print()

    agg_dir = out_root.parent / "AGGREGATE"
    agg_dir.mkdir(parents=True, exist_ok=True)
    (agg_dir / "02_multi_scoring_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[write] {agg_dir / '02_multi_scoring_summary.json'}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generator", default="all", choices=GENERATORS_4 + ["all"])
    p.add_argument("--target", default="all", choices=TARGETS_15 + ["all"])
    p.add_argument("--top_k", type=int, default=30)
    p.add_argument("--tool", default="both", choices=["smina", "gnina", "both"])
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--out_root", type=Path,
                   default=Path("results/eval/phase7/multi_scoring"))
    p.add_argument("--poses_root", type=Path,
                   default=Path("results/eval/phase7/poses"))
    p.add_argument("--repo_root", type=Path, default=Path("/home/zhao/code/ThermoFrag"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--aggregate_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(args, out_root)
        return

    if args.dry_run:
        print(f"[dry] tool={args.tool} top_k={args.top_k} "
              f"generator={args.generator} target={args.target} out={out_root}")
        return

    gnina_bin = _find_gnina() if args.tool in ("gnina", "both") else None
    if args.tool in ("gnina", "both") and gnina_bin is None:
        print("[warn] no GNINA binary found; will only run smina")
    if args.tool in ("smina", "both") and not SMINA_BIN.exists():
        print(f"[warn] smina not found at {SMINA_BIN}; cannot rescore.")
        if args.tool == "smina":
            sys.exit(2)

    gens = GENERATORS_4 if args.generator == "all" else [args.generator]
    tgts = TARGETS_15 if args.target == "all" else [args.target]

    t0 = time.time()
    for gen in gens:
        for tgt in tgts:
            process_target(gen, tgt, args, gnina_bin)
    wall = time.time() - t0

    aggregate(args, out_root)

    manifest = {
        "git_sha": _git_sha(args.repo_root),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.executable,
        "wall_seconds": wall,
        "completed_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    out_root.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
