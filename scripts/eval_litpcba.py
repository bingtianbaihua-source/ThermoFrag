"""Run LIT-PCBA evaluation: generate, dock with Vina, rerank with DiffDock.

Usage:
    python scripts/eval_litpcba.py --config configs/default.yaml --checkpoint results/checkpoints/joint.pt

Outputs:
    results/litpcba/<target>/{generated.smi, vina_scores.csv, diffdock_confidence.csv}
"""
from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--targets", default="all")
    p.add_argument("--n_per_target", type=int, default=1000)
    args = p.parse_args()
    print(f"[eval_litpcba] targets={args.targets} n_per_target={args.n_per_target}")
    raise SystemExit("Not implemented; see docs/FIGURES.md Fig 7")


if __name__ == "__main__":
    main()
