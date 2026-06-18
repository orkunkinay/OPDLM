#!/bin/bash
#SBATCH --job-name=opdlm_setup
#SBATCH --partition=Teaching
#SBATCH --output=logs/opdlm_setup_%j.out
#SBATCH --error=logs/opdlm_setup_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200_3g.71gb:1

# Download OPDLM datasets and model assets for cluster runs.
#
# Run from the repo root:
#   mkdir -p logs
#   bash cluster/setup_assets.sh
#
# Or submit as a Slurm job if your cluster allows network access from jobs:
#   mkdir -p logs
#   sbatch cluster/setup_assets.sh
#
# Common overrides:
#   MODEL_SIZE=8B bash cluster/setup_assets.sh
#   DOWNLOAD_MODELS=false bash cluster/setup_assets.sh
#   PREPARE_CODEFORCES=false bash cluster/setup_assets.sh
#   PROJECT_DIR=/path/to/repo MODEL_ROOT=/scratch/$USER/opdlm_models bash cluster/setup_assets.sh
#   BOOTSTRAP_ASSET_ENV=false bash cluster/setup_assets.sh
#   CONDA_ENV=opdlm sbatch cluster/setup_assets.sh
#   CONDA_SH=$HOME/miniconda3/etc/profile.d/conda.sh sbatch cluster/setup_assets.sh

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-}
if [ -z "$PROJECT_DIR" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/rl.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
  elif [ -f "rl.py" ]; then
    PROJECT_DIR="$(pwd)"
  else
    echo "ERROR: Could not infer PROJECT_DIR."
    echo "Run from the repo root, or set PROJECT_DIR=/path/to/repo."
    exit 1
  fi
fi

cd "$PROJECT_DIR"
mkdir -p logs data

CONDA_ENV=${CONDA_ENV-opdlm}
CONDA_SH=${CONDA_SH:-}
if [ -z "$CONDA_SH" ]; then
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/mambaforge/etc/profile.d/conda.sh" \
    "$HOME/miniforge3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
      CONDA_SH="$candidate"
      break
    fi
  done
fi

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -n "$CONDA_ENV" ]; then
  # Slurm jobs are non-interactive, so `conda activate` needs the shell hook.
  if [ -f "$CONDA_SH" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  fi
  if command -v conda >/dev/null 2>&1 && conda activate "$CONDA_ENV"; then
    :
  else
    echo "WARNING: conda env '$CONDA_ENV' could not be activated; using current Python environment"
  fi
else
  echo "WARNING: .venv/bin/activate not found and conda is unavailable or disabled; using current Python environment"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

PYTHON_BIN=${PYTHON_BIN:-}
if [ -z "$PYTHON_BIN" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    echo "ERROR: required command 'python' or 'python3' not found."
    echo "Load Python first, or submit with a Python environment on PATH."
    exit 1
  fi
fi

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN
  HF_TOKEN=$(cat "$HOME/.hf_token")
fi

DATA_DIR=${DATA_DIR:-data}
MODEL_ROOT=${MODEL_ROOT:-pretrained_models}
MODEL_SIZE=${MODEL_SIZE:-4B}
DOWNLOAD_DATA=${DOWNLOAD_DATA:-true}
DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-true}
PREPARE_CODEFORCES=${PREPARE_CODEFORCES:-true}
PREPARE_LCB=${PREPARE_LCB:-false}
VERIFY_DATA=${VERIFY_DATA:-$DOWNLOAD_DATA}
BOOTSTRAP_ASSET_ENV=${BOOTSTRAP_ASSET_ENV:-true}
ASSET_ENV_DIR=${ASSET_ENV_DIR:-.venv-assets}

export HF_HOME=${HF_HOME:-$PROJECT_DIR/$MODEL_ROOT/.hf_cache}
mkdir -p "$DATA_DIR" "$MODEL_ROOT/BD3LM" "$MODEL_ROOT/Qwen" "$HF_HOME"

python_has_module() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import importlib.util
import sys

raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)
PY
}

ensure_python_modules() {
  local missing=()
  for module in "$@"; do
    if ! python_has_module "$module"; then
      missing+=("$module")
    fi
  done

  if [ "${#missing[@]}" -eq 0 ]; then
    return
  fi

  if [ "$BOOTSTRAP_ASSET_ENV" != "true" ]; then
    echo "ERROR: Python is missing required module(s): ${missing[*]}"
    echo "Activate the project environment, install requirements.txt, or re-run with BOOTSTRAP_ASSET_ENV=true."
    exit 1
  fi

  echo "===== BOOTSTRAP ASSET ENV ====="
  echo "Missing Python module(s): ${missing[*]}"
  echo "Creating lightweight asset environment at $ASSET_ENV_DIR"
  "$PYTHON_BIN" -m venv "$ASSET_ENV_DIR"
  # shellcheck disable=SC1091
  source "$ASSET_ENV_DIR/bin/activate"
  PYTHON_BIN=python
  python -m pip install --upgrade pip

  local packages=("huggingface_hub[cli]")
  if [ "$PREPARE_CODEFORCES" = "true" ]; then
    packages+=("datasets")
  fi
  python -m pip install "${packages[@]}"
}

download_hf() {
  local repo=$1
  local repo_type=$2
  local local_dir=$3

  echo "===== DOWNLOAD: $repo -> $local_dir ====="
  mkdir -p "$local_dir"
  "$PYTHON_BIN" - "$repo" "$repo_type" "$local_dir" <<'PY'
import sys

from huggingface_hub import snapshot_download

repo_id, repo_type, local_dir = sys.argv[1:4]
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=local_dir,
)
PY
}

if [ "$DOWNLOAD_DATA" = "true" ] || [ "$DOWNLOAD_MODELS" = "true" ]; then
  ensure_python_modules huggingface_hub
fi
if [ "$PREPARE_CODEFORCES" = "true" ]; then
  ensure_python_modules datasets
fi

echo "===== SETUP INFO ====="
echo "Project dir: $PROJECT_DIR"
echo "Data dir: $DATA_DIR"
echo "Model root: $MODEL_ROOT"
echo "HF_HOME: $HF_HOME"
echo "Python: $(command -v "$PYTHON_BIN")"
echo "Model size: $MODEL_SIZE"
echo "Download data: $DOWNLOAD_DATA"
echo "Download models: $DOWNLOAD_MODELS"
echo "Prepare Codeforces: $PREPARE_CODEFORCES"
echo "Prepare LCB: $PREPARE_LCB"
echo "Verify data: $VERIFY_DATA"
echo "======================"

if [ "$DOWNLOAD_DATA" = "true" ]; then
  download_hf "divelab/opdlm_eval_data" "dataset" "$DATA_DIR"
  download_hf "divelab/opdlm_train_data" "dataset" "$DATA_DIR"
fi

case "$MODEL_SIZE" in
  4B)
    STUDENT_REPO="divelab/Qwen3-4B-a2d-init"
    TEACHER_REPO="Qwen/Qwen3-4B"
    STUDENT_DIR="$MODEL_ROOT/BD3LM/Qwen3-4B-a2d-init"
    TEACHER_DIR="$MODEL_ROOT/Qwen/Qwen3-4B"
    ;;
  8B)
    STUDENT_REPO="divelab/Qwen3-8B-a2d-init"
    TEACHER_REPO="Qwen/Qwen3-8B"
    STUDENT_DIR="$MODEL_ROOT/BD3LM/Qwen3-8B-a2d-init"
    TEACHER_DIR="$MODEL_ROOT/Qwen/Qwen3-8B"
    ;;
  0.6B|1.7B)
    STUDENT_REPO=""
    TEACHER_REPO="Qwen/Qwen3-$MODEL_SIZE"
    STUDENT_DIR="$MODEL_ROOT/BD3LM/Qwen3-$MODEL_SIZE-a2d-init"
    TEACHER_DIR="$MODEL_ROOT/Qwen/Qwen3-$MODEL_SIZE"
    ;;
  *)
    echo "ERROR: MODEL_SIZE must be one of: 4B, 8B, 0.6B, 1.7B"
    exit 1
    ;;
esac

if [ "$DOWNLOAD_MODELS" = "true" ]; then
  if [ -n "$STUDENT_REPO" ]; then
    download_hf "$STUDENT_REPO" "model" "$STUDENT_DIR"
  else
    echo "WARNING: no public A2D init download is configured for MODEL_SIZE=$MODEL_SIZE."
    echo "The repo documents direct A2D init downloads for 4B and 8B; smaller init models need local conversion."
    echo "Expected student path after conversion: $STUDENT_DIR"
  fi
  download_hf "$TEACHER_REPO" "model" "$TEACHER_DIR"
fi

if [ "$PREPARE_CODEFORCES" = "true" ]; then
  "$PYTHON_BIN" data/prepare_codeforces.py --data-dir "$DATA_DIR"
fi

if [ "$PREPARE_LCB" = "true" ]; then
  "$PYTHON_BIN" prepare_lcb_data.py
fi

if [ "$VERIFY_DATA" = "true" ]; then
  echo "===== VERIFY DATA FILES ====="
  "$PYTHON_BIN" - "$DATA_DIR" <<'PY'
import json
import os
import sys

data_dir = sys.argv[1]
required = ["opdlm_train.json", "MATH500.json", "GSM8K.json", "MATH_train_traceRL.json"]
optional = ["Codeforces.json", "Codeforces_train.json", "LCB_v5.json", "LCB_v6.json"]

missing = []
for name in required + optional:
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        label = "missing optional" if name in optional else "missing"
        print(f"{label}: {path}")
        if name in required:
            missing.append(name)
        continue
    try:
        with open(path) as f:
            rows = json.load(f)
        n = len(rows) if hasattr(rows, "__len__") else "unknown"
    except Exception as exc:
        n = f"unreadable: {exc}"
    print(f"ok: {path} ({n} rows)")

if missing:
    raise SystemExit(f"Required data files missing: {', '.join(missing)}")
PY
fi

echo "===== MODEL PATHS ====="
echo "STUDENT=$PROJECT_DIR/$STUDENT_DIR"
echo "TEACHER=$PROJECT_DIR/$TEACHER_DIR"
if [ -d "$STUDENT_DIR" ]; then
  if [ -f "$STUDENT_DIR/config.json" ]; then
    echo "ok: $STUDENT_DIR/config.json"
  else
    echo "missing: $STUDENT_DIR/config.json"
  fi
else
  echo "missing student dir: $STUDENT_DIR"
fi
if [ -d "$TEACHER_DIR" ]; then
  if [ -f "$TEACHER_DIR/config.json" ]; then
    echo "ok: $TEACHER_DIR/config.json"
  else
    echo "missing: $TEACHER_DIR/config.json"
  fi
else
  echo "missing teacher dir: $TEACHER_DIR"
fi

echo "===== NEXT STEP ====="
echo "Run training with local model paths:"
echo "  MODEL_SIZE=$MODEL_SIZE sbatch cluster/teaching_h200_train.sh dataset.train_dataset=opdlm_train"
echo
echo "Or pass explicit paths:"
echo "  sbatch --export=ALL,STUDENT=$PROJECT_DIR/$STUDENT_DIR,TEACHER=$PROJECT_DIR/$TEACHER_DIR cluster/teaching_h200_train.sh dataset.train_dataset=opdlm_train"
