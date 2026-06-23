#!/bin/bash
#SBATCH --job-name=opdlm_train
#SBATCH --partition=Teaching
#SBATCH --output=logs/opdlm_train_%j.out
#SBATCH --error=logs/opdlm_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --signal=USR1@300

# Slurm launcher for this repo on the Teaching H200 MIG partition.
#
# Submit from the repo root after pushing/pulling this branch there:
#   mkdir -p logs
#   sbatch cluster/teaching_h200_train.sh
#
# Useful overrides:
#   sbatch --export=ALL,PROJECT_DIR=/path/to/phoenix cluster/teaching_h200_train.sh
#   sbatch --export=ALL,MODEL_ROOT=/scratch/$USER/opdlm_models cluster/teaching_h200_train.sh
#   sbatch --export=ALL,STUDENT=/path/to/a2d-init,TEACHER=/path/to/Qwen3-4B cluster/teaching_h200_train.sh
#   sbatch --export=ALL,WANDB_ENABLED=false cluster/teaching_h200_train.sh
#   sbatch --export=ALL,MODEL_SIZE=8B cluster/teaching_h200_train.sh dataset.train_dataset=opdlm_train
#   sbatch cluster/teaching_h200_train.sh model.attn_backend=sdpa rollout.max_active=32

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-}
if [ -z "$PROJECT_DIR" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/rl.py" ]; then
    PROJECT_DIR="$SLURM_SUBMIT_DIR"
  else
    echo "ERROR: Could not infer PROJECT_DIR."
    echo "Submit from the repo root, or pass --export=ALL,PROJECT_DIR=/path/to/repo."
    exit 1
  fi
fi

cd "$PROJECT_DIR"
mkdir -p logs checkpoints outputs

echo "===== JOB INFO ====="
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Job name: ${SLURM_JOB_NAME:-opdlm_train}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working dir: $(pwd)"
echo "===================="

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate opdlm
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
  source venv/bin/activate
else
  echo "ERROR: No Python environment found."
  exit 1
fi

echo "Python: $(which python)"
python -c "import torch, numpy; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN
  HF_TOKEN=$(cat "$HOME/.hf_token")
fi

if [ -f "$HOME/.wandb_api_key" ]; then
  export WANDB_API_KEY
  WANDB_API_KEY=$(cat "$HOME/.wandb_api_key")
fi

module add cuda || true

echo "===== NODE / GPU INFO ====="
hostname
nvidia-smi
echo "==========================="

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("GPU memory GB:", props.total_memory / 1024**3)
else:
    raise RuntimeError("CUDA unavailable")
PY

CONFIG=${CONFIG:-configs/rl_bd3lm.yaml}
NUM_GPUS=${NUM_GPUS:-1}
ACCEL_CONFIG=${ACCEL_CONFIG:-accelerate_configs/1_node_${NUM_GPUS}_gpus_deepspeed_zero3.yaml}
if [ ! -f "$ACCEL_CONFIG" ]; then
  ACCEL_CONFIG=accelerate_configs/1gpu_debug.yaml
fi

# Stable run directory: re-submitting this script with the same job name resumes.
export EXP_BASE=${EXP_BASE:-experiments}
export RUN_DIR=${RUN_DIR:-${EXP_BASE}/${SLURM_JOB_NAME:-opdlm_train}}
export RUN_TIMESTAMP
RUN_TIMESTAMP=$(date +"%m%d_%H%M%S")

MODEL_SIZE=${MODEL_SIZE:-4B}
MODEL_ROOT=${MODEL_ROOT:-pretrained_models}
STUDENT=${STUDENT:-}
TEACHER=${TEACHER:-}
WANDB_ENABLED=${WANDB_ENABLED:-true}
EXPERIMENT_PORT=${EXPERIMENT_PORT:-29500}
ROLLOUT_BASE_PORT=${ROLLOUT_BASE_PORT:-29000}

LOCAL_STUDENT="$MODEL_ROOT/BD3LM/Qwen3-${MODEL_SIZE}-a2d-init"
LOCAL_TEACHER="$MODEL_ROOT/Qwen/Qwen3-${MODEL_SIZE}"
if [ -z "$STUDENT" ] && [ -d "$LOCAL_STUDENT" ]; then
  STUDENT="$LOCAL_STUDENT"
fi
if [ -z "$TEACHER" ] && [ -d "$LOCAL_TEACHER" ]; then
  TEACHER="$LOCAL_TEACHER"
fi

EXTRA_ARGS=()
if [ -n "$STUDENT" ]; then
  EXTRA_ARGS+=("model.pretrained_model=$STUDENT")
fi
if [ -n "$TEACHER" ]; then
  EXTRA_ARGS+=("model.teacher_model=$TEACHER")
fi

mkdir -p "$RUN_DIR"
echo "Run dir: $RUN_DIR"
echo "Accelerate config: $ACCEL_CONFIG"
echo "Model size selector: $MODEL_SIZE"
echo "Student override: ${STUDENT:-using config default}"
echo "Teacher override: ${TEACHER:-using config default}"

accelerate launch \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "$EXPERIMENT_PORT" \
  --config_file "$ACCEL_CONFIG" \
  rl.py \
  config="$CONFIG" \
  experiment.auto_resume=true \
  experiment.project="$RUN_DIR" \
  experiment.save_every=10 \
  experiment.log_memory_every=50 \
  rollout.base_port="$ROLLOUT_BASE_PORT" \
  wandb.enabled="$WANDB_ENABLED" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "===== JOB FINISHED ====="
echo "End time: $(date)"
