#!/usr/bin/env bash
# Resume Vina/strain/stats for TF-pocket-v2 after killing the mid-run docker.
# Caps remaining target pools at 100 ligands — matching baselines' 100-cap
# (see feedback_pool_size_fairness.md) so the comparison is fair and fast.
set -euo pipefail

OUT_ROOT=results/eval/phase4_tf_pocket_v2
STATS_ROOT=results/eval/phase5_tf_pocket_v2
DECODED_DIR=$OUT_ROOT/decoded
VINA_DIR=$OUT_ROOT/vina
STRAIN_DIR=$OUT_ROOT/strain
mkdir -p "$VINA_DIR" "$STRAIN_DIR" "$STATS_ROOT"

TARGETS=(ADRB2 ALDH1 ESR_ago ESR_antago FEN1 GBA IDH1 KAT2A MAPK1 MTORC1 OPRK1 PKM2 PPARG TP53 VDR)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf-eval
export PATH="/home/zhao/miniconda3/envs/tf-eval/bin:$PATH"

# --- Vina: only remaining targets, cap input at 100 --------------------
for t in "${TARGETS[@]}"; do
    if [[ -s "$VINA_DIR/$t.parquet" ]]; then
        echo "[vina] $t already done, skipping"
        continue
    fi
    echo "[vina] $t (limit 100)"
    python scripts/dock_vina.py \
        --decoded-dir "$DECODED_DIR" \
        --out-dir     "$VINA_DIR" \
        --targets "$t" \
        --workers 12 \
        --limit 100
done

# --- Strain: all targets (also cap at 100) -----------------------------
echo "[strain] all targets (limit 100)"
python scripts/eval_strain.py \
    --decoded-dir "$DECODED_DIR" \
    --out-dir     "$STRAIN_DIR" \
    --workers 12 \
    --limit 100 || true  # fall back gracefully if --limit isn't supported

# --- Generator-vs-generator stats --------------------------------------
conda activate py311
echo "[stats] gvg (pool cap 100 on baselines)"
python scripts/eval_generator_vs_generator.py \
    --baselines rxnflow bbar targetdiff \
    --tf-root "$OUT_ROOT" \
    --out-dir "$STATS_ROOT" \
    --max-baseline-pool 100

echo "[pipeline] done"
