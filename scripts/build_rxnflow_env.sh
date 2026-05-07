#!/usr/bin/env bash
# Build a RxnFlow env_dir from ZINCFrag-200k (public reproducible subset).
# Runs from the rxnflow conda env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate rxnflow env if not already active
if [[ "${CONDA_DEFAULT_ENV:-}" != "rxnflow" ]]; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate rxnflow
  else
    echo "[build_env] ERROR: conda not found; activate rxnflow env manually first"
    exit 1
  fi
fi

cd "$PROJECT_ROOT/vendor/rxnflow"
mkdir -p data/envs

# b_create_env.py lives under data/scripts but imports rxnflow modules (installed);
# run from vendor root so its relative imports resolve.
cd data
if [ ! -f envs/zincfrag/.done ]; then
  python scripts/b_create_env.py \
    -b building_blocks/zincfrag.smi.gz \
    -o envs/zincfrag \
    -t templates/real.txt \
    --cpu "${CPU:-16}"
  touch envs/zincfrag/.done
else
  echo "[build_env] envs/zincfrag already exists"
fi
