#!/usr/bin/env bash
# Chains: wait-for-sampling → Vina → strain → generator-vs-generator stats.
# Assumes scripts/sample_rxnflow.py is already running or done.
#
# Usage: bash scripts/run_rxnflow_baseline_pipeline.sh &
# Log:   results/logs/rxnflow_pipeline.log
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

LOG=results/logs/rxnflow_pipeline.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

# Activate py310 conda env so subprocess PATH contains `mk_prepare_ligand.py`,
# `vina`, and any other CLI tools dock_vina.py / eval_strain.py invoke.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate py310

DECODED_DIR=results/eval/phase4_baselines/rxnflow/decoded
VINA_DIR=results/eval/phase4_baselines/rxnflow/vina
STRAIN_DIR=results/eval/phase4_baselines/rxnflow/strain

echo "[pipeline] $(date -Iseconds) start"
echo "[pipeline] env: $(which python) | vina=$(which vina) | meeko=$(which mk_prepare_ligand.py)"

echo "[pipeline] waiting for sample_rxnflow.py to finish..."
while pgrep -f "scripts/sample_rxnflow.py" > /dev/null; do sleep 30; done
n_parquet=$(ls -1 "$DECODED_DIR"/*.parquet 2>/dev/null | wc -l)
echo "[pipeline] sampling done, ${n_parquet} parquets present"

if [[ "$n_parquet" -lt 15 ]]; then
  echo "[pipeline] ERROR: expected 15 decoded parquets, got $n_parquet"
  exit 1
fi

echo "[pipeline] Vina docking"
python scripts/dock_vina.py \
  --decoded-dir "$DECODED_DIR" \
  --out-dir     "$VINA_DIR" \
  --exhaustiveness 8 \
  2>&1 | tail -40

echo "[pipeline] OpenMM strain"
python scripts/eval_strain.py \
  --decoded-dir "$DECODED_DIR" \
  --out-dir     "$STRAIN_DIR" \
  2>&1 | tail -40

echo "[pipeline] generator-vs-generator stats"
python scripts/eval_generator_vs_generator.py \
  --baselines rxnflow \
  2>&1 | tail -40

echo "[pipeline] $(date -Iseconds) DONE"
