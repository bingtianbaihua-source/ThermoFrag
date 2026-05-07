#!/usr/bin/env bash
# Reproduce all 8 main figures and SI figures end-to-end from cached checkpoints.
#
# Prerequisites:
#   - data/processed/ has zinc_unconditional.lmdb and chembl_conditional.lmdb
#     (built by scripts/build_zinc_fragments.py + scripts/build_conditional_lmdb.py)
#   - results/checkpoints/ has qm_final.pt, coupling_final.pt, joint_final.pt
#   - data/external/LIT-PCBA.tar.gz is present
#   - Phase-4 sample pools + decoded parquets live under results/eval/phase4/
#
# This script only re-runs the downstream evaluators; it does NOT retrain.
# Training is orchestrated from configs/phase{1,2,3}.yaml + scripts/train.py.
#
# Outputs land in results/eval/phase1..5/ and results/figures/ symlinks.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[repro] 0 — directory check"
mkdir -p results/{logs,figures,eval/phase5}

# ---------------------------------------------------------------------------
# Fig 2 (C1): QM energy consistency
# ---------------------------------------------------------------------------
echo "[repro] Fig 2 / C1 QM energy consistency"
# qm_last.pt (step 178000) is the canonical checkpoint — step 178140 was degraded
# by a late-training loss spike. Build the recalibrated checkpoint from it.
python scripts/recalibrate_qm.py \
    --ckpt-in  results/checkpoints/qm_last.pt \
    --ckpt-out results/checkpoints/qm_recalibrated.pt \
    --train    data/processed/spice/train \
    --max-samples 20000
python scripts/eval_qm.py \
    --ckpt results/checkpoints/qm_last.pt \
    --data data/processed/spice \
    --dataset spice \
    --out  results/eval/phase1/
python scripts/plot_fig2.py \
    --ckpt-recal results/checkpoints/qm_recalibrated.pt \
    --ckpt-raw   results/checkpoints/qm_last.pt \
    --spice-val  data/processed/spice/val \
    --out        results/eval/phase1_recal

# ---------------------------------------------------------------------------
# Fig 3: temperature sweep
# ---------------------------------------------------------------------------
echo "[repro] Fig 3 temperature sweep"
python scripts/sweep_temperature.py \
    --n 256 --mh-steps 60 --batch-size 128 \
    --betas 0.1 0.3 1.0 3.0 10.0 \
    --target-axis qed --target-value 0.8 --hit-sigma 0.5

# ---------------------------------------------------------------------------
# Fig 5 (C2): chemical-potential interpretability
# ---------------------------------------------------------------------------
echo "[repro] Fig 5 / C2 chemical potential"
python scripts/eval_chempot.py \
    --ckpt results/checkpoints/joint_final.pt \
    --lmdb data/processed/chembl_conditional.lmdb \
    --out results/eval/phase3/
python scripts/plot_fig5.py

# ---------------------------------------------------------------------------
# Fig 6 (C5): OOD AUROC + Pareto reachability
# ---------------------------------------------------------------------------
echo "[repro] Fig 6 / C5 OOD AUROC + Pareto"
python scripts/eval_ood_auroc.py
python scripts/eval_pareto.py

# ---------------------------------------------------------------------------
# Fig 7 (C3): LIT-PCBA Vina head-to-head
# ---------------------------------------------------------------------------
echo "[repro] Fig 7 / C3 receptor prep + Vina"
python scripts/prep_litpcba_receptors.py
python scripts/dock_vina.py --workers "${VINA_WORKERS:-6}" --exhaustiveness 8
# No-μ ablation Vina (C6 pair)
python scripts/dock_nomu_ablation.py --workers "${VINA_WORKERS:-6}" --exhaustiveness 8
# Compose the paired summary / box plot
python scripts/eval_c3_vs_litpcba.py
python scripts/eval_c3_c4_c6_summary.py

# ---------------------------------------------------------------------------
# Fig 8 (C4): post-MMFF strain distribution
# ---------------------------------------------------------------------------
echo "[repro] Fig 8 / C4 strain audit"
python scripts/eval_strain.py --workers "${VINA_WORKERS:-6}"

# ---------------------------------------------------------------------------
# SI: detailed-balance numerical check
# ---------------------------------------------------------------------------
echo "[repro] SI S1 detailed-balance"
python scripts/eval_detailed_balance.py

# ---------------------------------------------------------------------------
# SI: ablation pools (no-μ and no-coupling) — sampled, decoded, strained
# ---------------------------------------------------------------------------
echo "[repro] SI ablation pools"
python scripts/sample_nomu_ablation.py \
    --n 1000 --mh-steps 60 --batch-size 128 \
    --out results/eval/phase5/nomu_samples/pool.pkl
python scripts/sample_nocoupling_ablation.py \
    --n 1000 --mh-steps 60 --batch-size 128 \
    --out results/eval/phase5/nocoupling_samples/pool.pkl

echo "[repro] — done. Figures in results/eval/phase*/"
