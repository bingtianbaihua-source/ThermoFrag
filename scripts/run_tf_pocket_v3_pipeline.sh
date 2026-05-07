#!/usr/bin/env bash
# Full TF-pocket-v3 eval pipeline: dump-per-target → sample → decode →
# dock → strain → stats on the 15 LIT-PCBA targets. Writes into
# results/eval/phase4_tf_pocket_v3 and results/eval/phase5_tf_pocket_v3.
#
# Usage (from repo root):
#   bash scripts/run_tf_pocket_v3_pipeline.sh [CKPT]
# Default CKPT = results/checkpoints/tf_pocket_v3_final.pt
set -euo pipefail

CKPT=${1:-results/checkpoints/tf_pocket_v3_final.pt}
N_CHAINS=${N_CHAINS:-1000}
MH_STEPS=${MH_STEPS:-60}
BATCH_SIZE=${BATCH_SIZE:-128}
BETA=${BETA:-1.0}
DOCK_WORKERS=${DOCK_WORKERS:-12}
STRAIN_WORKERS=${STRAIN_WORKERS:-12}

JOINT_CKPT=results/checkpoints/joint_final.pt
CONFIG=configs/phase3.yaml
DATA=data/processed/chembl_conditional.lmdb
LIB=data/processed/fragment_library.parquet
TARGETS_DIR=results/eval/phase4/litpcba_targets

GEOM_DIR=data/processed/pocket_geom/litpcba
POCKET_EMB_DIR=data/processed/pocket_embeds/litpcba_v3

OUT_ROOT=results/eval/phase4_tf_pocket_v3
STATS_ROOT=results/eval/phase5_tf_pocket_v3
SAMPLES_DIR=$OUT_ROOT/samples
DECODED_DIR=$OUT_ROOT/decoded
VINA_DIR=$OUT_ROOT/vina
STRAIN_DIR=$OUT_ROOT/strain
mkdir -p "$SAMPLES_DIR" "$DECODED_DIR" "$VINA_DIR" "$STRAIN_DIR" "$STATS_ROOT" "$POCKET_EMB_DIR"

TARGETS=(ADRB2 ALDH1 ESR_ago ESR_antago FEN1 GBA IDH1 KAT2A MAPK1 MTORC1 OPRK1 PKM2 PPARG TP53 VDR)

source ~/miniconda3/etc/profile.d/conda.sh

echo "[pipeline-v3] checkpoint: $CKPT"
echo "[pipeline-v3] output root: $OUT_ROOT"

# --- 0. Dump per-target pocket vectors from the v3 EGNN -----------------
conda activate py311
echo "[pipeline-v3] dumping LIT-PCBA pocket embeddings from EGNN"
python scripts/dump_pocket_v3_litpcba.py \
    --ckpt "$CKPT" \
    --in-dir  "$GEOM_DIR" \
    --out-dir "$POCKET_EMB_DIR"

# --- 1. Sample -----------------------------------------------------------
for t in "${TARGETS[@]}"; do
    y_file=$TARGETS_DIR/$t/y_raw.npy
    pocket=$POCKET_EMB_DIR/$t.npy
    out=$SAMPLES_DIR/$t.pkl
    if [[ -s "$out" ]]; then
        echo "[sample] $t already done, skipping"
        continue
    fi
    echo "[sample] $t"
    python scripts/sample.py \
        --checkpoint "$JOINT_CKPT" \
        --config "$CONFIG" \
        --data "$DATA" \
        --library "$LIB" \
        --pocket-ckpt "$CKPT" \
        --pocket-embed "$pocket" \
        --y-file "$y_file" \
        --n "$N_CHAINS" \
        --mh-steps "$MH_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --beta "$BETA" \
        --seed 0 \
        --out "$out"
done

# --- 2. Decode ----------------------------------------------------------
echo "[decode] all targets"
python scripts/decode_samples.py \
    --samples-dir "$SAMPLES_DIR" \
    --lib "$LIB" \
    --out-dir "$DECODED_DIR"

# --- 3. Vina docking ----------------------------------------------------
conda activate tf-eval
export PATH="/home/zhao/miniconda3/envs/tf-eval/bin:$PATH"
for t in "${TARGETS[@]}"; do
    if [[ -s "$VINA_DIR/$t.parquet" ]]; then
        echo "[vina] $t already done, skipping"
        continue
    fi
    echo "[vina] $t"
    python scripts/dock_vina.py \
        --decoded-dir "$DECODED_DIR" \
        --out-dir     "$VINA_DIR" \
        --targets "$t" \
        --workers "$DOCK_WORKERS"
done

# --- 4. Strain (OpenMM GAFF) --------------------------------------------
echo "[strain] all targets"
python scripts/eval_strain.py \
    --decoded-dir "$DECODED_DIR" \
    --out-dir     "$STRAIN_DIR" \
    --workers "$STRAIN_WORKERS"

# --- 5. Generator-vs-generator stats ------------------------------------
conda activate py311
echo "[stats] gvg"
python scripts/eval_generator_vs_generator.py \
    --baselines rxnflow bbar targetdiff \
    --tf-root "$OUT_ROOT" \
    --out-dir "$STATS_ROOT" \
    --max-baseline-pool 100

echo "[pipeline-v3] done"
echo "  samples:  $SAMPLES_DIR"
echo "  decoded:  $DECODED_DIR"
echo "  vina:     $VINA_DIR"
echo "  strain:   $STRAIN_DIR"
echo "  stats:    $STATS_ROOT"
