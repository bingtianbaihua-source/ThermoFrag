#!/usr/bin/env bash
# Bootstrap ThermoFrag on a new machine.
#
# Does:
#   1. Detect project root (the parent of this scripts/ dir).
#   2. Template vendor/rxnflow/environment.yml with the absolute project path
#      (conda env yaml cannot use env vars in pip `-e` entries).
#   3. Build the `rxnflow` conda env from the templated yaml (if missing).
#   4. Re-plant cached PharmacoNet weights from vendor/_cached_weights/pmnet/
#      into ~/.local/share/pmnet/ where the library expects them.
#
# Does NOT create the ThermoFrag main `py310` env — that needs the upstream
# environment.yml from the repo root. See docs/MIGRATION.md §3.
#
# Usage:
#   bash scripts/setup_new_machine.sh                # full setup
#   bash scripts/setup_new_machine.sh --weights-only # just replant weights
#   bash scripts/setup_new_machine.sh --env-only     # just build rxnflow env
set -euo pipefail

MODE=${1:-full}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "[setup] PROJECT_ROOT=$PROJECT_ROOT"

replant_weights() {
  local src="$PROJECT_ROOT/vendor/_cached_weights/pmnet"
  local dst="$HOME/.local/share/pmnet"
  if [[ ! -d "$src" ]]; then
    echo "[setup] WARN: cached weights missing at $src — skipping"
    return
  fi
  mkdir -p "$dst/tacogfn_proxy"
  cp -n "$src/pmnet.tar" "$dst/pmnet.tar" 2>/dev/null && echo "[setup] placed pmnet.tar"
  cp -n "$src/tacogfn_proxy/model-QVina-ZINCDock15M.ckpt" \
        "$dst/tacogfn_proxy/model-QVina-ZINCDock15M.ckpt" 2>/dev/null \
        && echo "[setup] placed tacogfn_proxy ckpt"
}

build_rxnflow_env() {
  local template="$PROJECT_ROOT/vendor/rxnflow/environment.yml"
  local rendered="$PROJECT_ROOT/vendor/rxnflow/environment.rendered.yml"

  if [[ ! -f "$template" ]]; then
    echo "[setup] ERROR: $template not found"
    exit 1
  fi

  sed "s|{{PROJECT_ROOT}}|$PROJECT_ROOT|g" "$template" > "$rendered"
  echo "[setup] rendered $rendered"

  if [[ -z "${CONDA_EXE:-}" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
      echo "[setup] ERROR: conda not found. Install miniconda then rerun."
      exit 1
    fi
  fi

  if conda env list 2>/dev/null | awk '{print $1}' | grep -qx rxnflow; then
    echo "[setup] env 'rxnflow' already exists — delete it first if you want to rebuild"
  else
    echo "[setup] creating conda env 'rxnflow' (this takes ~10 min)"
    conda env create -f "$rendered"
  fi

  rm -f "$rendered"
}

build_env_dir() {
  local env_dir="$PROJECT_ROOT/vendor/rxnflow/data/envs/zincfrag"
  if [[ -f "$env_dir/.done" ]]; then
    echo "[setup] zincfrag env_dir already built"
    return
  fi
  echo "[setup] building zincfrag env_dir (~1 min)"
  bash "$PROJECT_ROOT/scripts/build_rxnflow_env.sh"
}

smoke_test() {
  echo "[setup] smoke-test: import rxnflow + pmnet_appl, check CUDA"
  "$HOME/miniconda3/envs/rxnflow/bin/python" - <<'PY'
import torch, rxnflow, pmnet_appl  # noqa: F401
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("cuda dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY
}

case "$MODE" in
  --weights-only)
    replant_weights
    ;;
  --env-only)
    build_rxnflow_env
    smoke_test
    ;;
  full)
    replant_weights
    build_rxnflow_env
    build_env_dir
    smoke_test
    ;;
  *)
    echo "usage: $0 [--weights-only|--env-only]"
    exit 2
    ;;
esac

echo "[setup] done"
