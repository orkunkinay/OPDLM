#!/bin/bash
#SBATCH --job-name=opdlm_train
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --output=logs/opdlm_train_%j.out
#SBATCH --error=logs/opdlm_train_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --signal=USR1@300

# Real RL-distillation training job for this repo (rl.py under accelerate).
# Re-submitting this script with the same --job-name resumes the same run
# automatically (experiment.auto_resume=true + a stable RUN_DIR), and
# SIGUSR1 (sent by Slurm 300s before the time limit) triggers a clean
# emergency checkpoint. Nothing here hardcodes a user-specific absolute path —
# edit the variables in the CONFIG block below for your cluster.

set -euo pipefail

mkdir -p logs checkpoints outputs

echo "===== JOB INFO ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Job name: ${SLURM_JOB_NAME:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "===================="

echo "===== ENVIRONMENT ====="
python --version || true
which python || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not_set}"
nvidia-smi || true
echo "======================="

# ── Activate your environment (edit as needed) ───────────────────────────────
# source .venv/bin/activate
# eval "$(conda shell.bash hook)" && conda activate opdlm
module add cuda || true

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

echo "===== PYTORCH CHECK ====="
python - <<'PY'
import sys
print("Python:", sys.version)
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("GPU memory GB:", round(props.total_memory / 1024**3, 2))
except Exception as e:
    print("Torch check failed:", repr(e))
PY
echo "========================="

# ── CONFIG (edit these) ──────────────────────────────────────────────────────
CONFIG=${CONFIG:-configs/rl_bd3lm.yaml}
NUM_GPUS=${NUM_GPUS:-1}
# Where runs are stored. A stable, job-name-based RUN_DIR lets a re-submitted
# job resume the same run (do NOT put %j / SLURM_JOB_ID in here).
export EXP_BASE=${EXP_BASE:-experiments}
export RUN_DIR=${RUN_DIR:-${EXP_BASE}/${SLURM_JOB_NAME:-opdlm_train}}
# Student / teacher models — set to your local paths or HF repo ids.
STUDENT=${STUDENT:-}
TEACHER=${TEACHER:-}

# All ranks must agree on the run timestamp (used for fresh-run naming).
export RUN_TIMESTAMP=$(date +"%m%d_%H%M%S")

ACCEL_CONFIG="accelerate_configs/1_node_${NUM_GPUS}_gpus_deepspeed_zero3.yaml"
[ -f "$ACCEL_CONFIG" ] || ACCEL_CONFIG="accelerate_configs/1gpu_debug.yaml"

EXTRA_ARGS=()
[ -n "$STUDENT" ] && EXTRA_ARGS+=("model.pretrained_model=$STUDENT")
[ -n "$TEACHER" ] && EXTRA_ARGS+=("model.teacher_model=$TEACHER")

mkdir -p "$RUN_DIR"
echo "Run dir (stable, resumable): $RUN_DIR"

accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${EXPERIMENT_PORT:-29500}" \
  --config_file "$ACCEL_CONFIG" \
  rl.py \
  config="$CONFIG" \
  experiment.auto_resume=true \
  experiment.project="$RUN_DIR" \
  experiment.save_every=10 \
  experiment.log_memory_every=50 \
  "${EXTRA_ARGS[@]}"

echo "===== JOB FINISHED ====="
echo "End time: $(date)"
