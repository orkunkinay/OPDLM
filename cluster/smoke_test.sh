#!/bin/bash
#SBATCH --job-name=opdlm_smoke
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:3g.71gb:1
#SBATCH --output=logs/opdlm_smoke_%j.out
#SBATCH --error=logs/opdlm_smoke_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --signal=USR1@120

# Tiny end-to-end smoke test: runs a couple of RL steps with small token
# budgets and a small rollout batch, then checkpoints. Use it to catch
# environment / config / checkpoint problems before launching an expensive
# real job. It exercises the same rollout -> reward -> train -> checkpoint ->
# resume path; the scientific hyperparameters are only shrunk here, not in the
# base config (config discipline).

set -euo pipefail

mkdir -p logs checkpoints outputs

echo "Node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
date
nvidia-smi || true

# ── Activate your environment (edit as needed) ───────────────────────────────
# source .venv/bin/activate
# eval "$(conda shell.bash hook)" && conda activate opdlm
module add cuda || true

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ── CONFIG (edit these) ──────────────────────────────────────────────────────
CONFIG=${CONFIG:-configs/rl_bd3lm.yaml}
NUM_GPUS=${NUM_GPUS:-1}
export EXP_BASE=${EXP_BASE:-experiments}
# Fresh smoke dir per submission so resume is exercised cleanly on a re-run.
export RUN_DIR=${RUN_DIR:-${EXP_BASE}/${SLURM_JOB_NAME:-opdlm_smoke}_${SLURM_JOB_ID:-local}}
STUDENT=${STUDENT:-}
TEACHER=${TEACHER:-}
export RUN_TIMESTAMP=$(date +"%m%d_%H%M%S")

ACCEL_CONFIG="accelerate_configs/1_node_${NUM_GPUS}_gpus_deepspeed_zero3.yaml"
[ -f "$ACCEL_CONFIG" ] || ACCEL_CONFIG="accelerate_configs/1gpu_debug.yaml"

EXTRA_ARGS=()
[ -n "$STUDENT" ] && EXTRA_ARGS+=("model.pretrained_model=$STUDENT")
[ -n "$TEACHER" ] && EXTRA_ARGS+=("model.teacher_model=$TEACHER")

mkdir -p "$RUN_DIR"
echo "Smoke run dir: $RUN_DIR"

# Smoke overrides (tiny job): 2 RL steps, tiny token budgets, small rollout,
# checkpoint every step, no W&B, eval off. These live here — NOT in the base
# config — so real runs keep their hyperparameters.
accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${EXPERIMENT_PORT:-29501}" \
  --config_file "$ACCEL_CONFIG" \
  rl.py \
  config="$CONFIG" \
  experiment.auto_resume=true \
  experiment.project="$RUN_DIR" \
  experiment.stop_RL_step=2 \
  experiment.save_every=1 \
  experiment.eval_every=1000 \
  experiment.log_memory_every=1 \
  evaluation.run_before_training=false \
  wandb.enabled=false \
  dataset.num_data_epochs=1 \
  rollout.num_task_per_step=4 \
  rollout.max_active=4 \
  max_token_schedule.start=64 \
  max_token_schedule.end=128 \
  max_token_schedule.ramp_steps=2 \
  "${EXTRA_ARGS[@]}"

echo "Smoke test finished. Inspect $RUN_DIR/ckpt and $RUN_DIR/run_metadata.json"
