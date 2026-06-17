# Running OPDLM on a Slurm GPU cluster

This guide covers running the RL-distillation trainer (`rl.py`) on a Slurm
cluster so that jobs are **resumable, observable, reproducible, and
preemption-tolerant**. It documents only the cluster plumbing — the training
algorithm, model, data, and hyperparameters are unchanged.

The cluster plumbing lives in:

| Path | What it adds |
|------|--------------|
| `cluster/smoke_test.sh` | Tiny 2-step end-to-end Slurm job |
| `cluster/train.sh` | Real, resumable training Slurm job |
| `cluster/cluster_utils.py` | Checkpoint discovery, auto-resume, run metadata, CUDA-memory logging |
| `tests/test_cluster_utils.py` | CPU-only tests for the helpers above |

---

## Setup

1. **Create / activate the environment** (see `README.md` §1 for the full
   install). The cluster scripts have a commented activation block — edit it:

   ```bash
   # in cluster/train.sh and cluster/smoke_test.sh
   # source .venv/bin/activate
   # eval "$(conda shell.bash hook)" && conda activate opdlm
   ```

2. **Point the scripts at your models.** Either edit the `STUDENT` / `TEACHER`
   variables in the script, or export them at submit time:

   ```bash
   sbatch --export=ALL,STUDENT=/path/to/a2d-init,TEACHER=/path/to/Qwen3-4B cluster/train.sh
   ```

   If you leave them empty, the values from `configs/rl_bd3lm.yaml` are used.

3. **Secrets / env vars (only if you use them):**
   - `WANDB_API_KEY` — needed only if W&B is enabled (`wandb.enabled=true`,
     the config default). Disable with `wandb.enabled=false` for offline runs.
   - `HF_TOKEN` / `HF_HOME` — only if a model or dataset is gated / remote.
   - `EXP_BASE` — base directory for run outputs (default `experiments/`).
   - `RUN_DIR` — the stable per-run directory used for auto-resume (default
     `${EXP_BASE}/${SLURM_JOB_NAME}`).

The SBATCH header (`--partition`, `--gres`, `--time`, `--mem`) in each script is
a template — adjust it for your queue and GPU type.

### Attention backend fallback

The default config keeps the original FlashAttention behavior with
`model.attn_backend=flash`. If your cluster cannot build or import
`flash-attn`, rerun with PyTorch SDPA:

```bash
python rl.py config=configs/rl_bd3lm.yaml model.attn_backend=sdpa
```

---

## Smoke test

Always run this first. It executes the full `rollout → reward → train →
checkpoint → resume` path with tiny token budgets and a 4-task rollout for 2 RL
steps, so it catches environment/config/checkpoint problems in minutes instead
of after an expensive job has started:

```bash
sbatch cluster/smoke_test.sh
```

The smaller settings (`stop_RL_step=2`, tiny `max_token_schedule`,
`rollout.num_task_per_step=4`, `wandb.enabled=false`, eval off) are passed as
CLI overrides **inside the script** — they are not written into the base config,
so real runs keep their scientific hyperparameters.

---

## Real training

```bash
sbatch cluster/train.sh
# multi-GPU: sbatch --gres=gpu:8 --export=ALL,NUM_GPUS=8 cluster/train.sh
```

`cluster/train.sh` launches `rl.py` under `accelerate` with
`experiment.auto_resume=true` and a **stable** run directory
(`${EXP_BASE}/${SLURM_JOB_NAME}`). Re-submitting the same script (same job
name) continues the same run rather than starting a new one.

---

## Monitoring

```bash
squeue -u $USER                         # your queued / running jobs
tail -f logs/opdlm_train_<JOB_ID>.out   # progress, prints, tqdm, memory logs
tail -f logs/opdlm_train_<JOB_ID>.err   # tracebacks / crashes
```

At startup the run logs the command, resolved config path, output/checkpoint
directory, resume mode, device, CUDA availability, GPU name + memory, package
versions, and git commit. During training it logs step, loss, LR, grad norm,
GPU-hours, and (per step) the max-token / dynamic-threshold schedule. Enable
periodic CUDA-memory logging with `experiment.log_memory_every=N` (the
`train.sh` template sets `50`); startup / engine-init / pre-exit memory is
always logged.

A reproducibility snapshot is written to `<RUN_DIR>/run_metadata.json`
(timestamp, command, resolved config, git commit + dirty flag, Python / Torch /
CUDA versions, hostname, Slurm job id, seed).

---

## Resume

- **Where checkpoints live:** under the run directory `RUN_DIR`:
  - `ckpt/<optimized_name>/` — latest model weights + tokenizer (rewritten every
    step; this is what JetEngine reloads and what eval loads).
  - `ckpt/metadata.json` — records the last completed RL step (`current_epoch`).
  - `training_state/<optimized_name>/` — DeepSpeed optimizer + LR-scheduler
    state (saved on the `save_every` cadence and on emergency shutdown).
  - `temp_data/data_iter_state.json` — epoch-mode data-iterator position.
  - Checkpoint writes are **atomic** (write to `*.tmp`, then rename) so a job
    killed mid-save never corrupts the live checkpoint.

- **Automatic resume:** with `experiment.auto_resume=true` (set by the scripts),
  a re-submitted job detects a valid checkpoint in `RUN_DIR`, reads the last
  step from `ckpt/metadata.json`, and continues at the next step. If no valid
  checkpoint exists yet, it starts fresh but keeps the same `RUN_DIR` so the
  *next* submit can resume. Invalid / incomplete checkpoints are skipped rather
  than crashing the job.

- **Resume from a specific run directory:**

  ```bash
  sbatch --export=ALL,RUN_DIR=experiments/my_old_run cluster/train.sh
  ```

- **Disable resume / force a clean start:** point at a new `RUN_DIR`, or run
  `rl.py` directly with `experiment.auto_resume=false experiment.start_from_scratch=true`.

> Note: the epoch-mode data iterator restores its *position* (index + epoch
> counter), not the exact reshuffle order, so progress and step counting are
> preserved across a restart.

---

## Killing a job safely

```bash
scancel <JOB_ID>
```

The job installs handlers for **SIGTERM** (sent by `scancel`) and **SIGUSR1**
(sent by Slurm `--signal=USR1@300`, i.e. 300 s before the time limit). On either
signal the trainer:

1. logs that the signal was received,
2. finishes the current RL step (it never interrupts a step mid-way, so the
   saved state is always consistent),
3. forces a full checkpoint (model + optimizer + LR scheduler),
4. flushes logs and exits cleanly.

Re-submitting `cluster/train.sh` then resumes from that emergency checkpoint.

---

## Tests

CPU-only, no GPU or real model required:

```bash
python -m pytest tests/test_cluster_utils.py     # if pytest is installed
python tests/test_cluster_utils.py               # standalone fallback
```

They cover checkpoint validation, latest-checkpoint discovery, corrupted /
incomplete checkpoint skipping, auto-resume step recovery, the memory logger not
crashing without CUDA, and run-metadata writing.

---

## Common failures

| Symptom | Likely cause & fix |
|---------|--------------------|
| `GPU training was requested, but torch.cuda.is_available() is False` | Submitted without a GPU or CUDA not loaded. Add `--gres=gpu:...` and `module add cuda`. |
| `CUDA out of memory` | Lower `rollout.max_active` / `rollout.num_task_per_step`, lower the max-token schedule, reduce `training.batch_size_lm` and raise `training.gradient_accumulation_steps`, keep DeepSpeed offload on, or use a smaller model / larger MIG slice. |
| Missing env var (e.g. `WANDB_RUN_ID`) | W&B is enabled but not configured. Set `WANDB_API_KEY`, or run with `wandb.enabled=false`. |
| W&B login problem | `wandb login` once, or set `WANDB_MODE=offline` / `wandb.enabled=false`. |
| Corrupted / incomplete checkpoint | Auto-resume skips invalid checkpoints automatically; if needed, point `RUN_DIR` at an earlier good run, or start fresh. |
| Wrong working directory | Submit from the repo root so `config=configs/...` and `data/...` resolve; the scripts `cd` nowhere and rely on `pwd`. |
| `config` file not found | Pass `config=configs/rl_bd3lm.yaml` (relative to the submit directory) — OmegaConf errors early if it is missing. |
