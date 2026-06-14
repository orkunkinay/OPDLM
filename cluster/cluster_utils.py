"""Small, dependency-light helpers for running this repo on a Slurm cluster.

Everything here is CPU-safe and import-light so it can be unit-tested without a
GPU. The functions cover three concerns:

  * checkpoint discovery / validation / auto-resume
  * run-metadata + reproducibility snapshots
  * CUDA memory observability

The training entrypoint (``rl.py``) wires these in; the heavy
checkpoint-writing logic itself still lives in ``train/rl_sdar.py``.
"""

import json
import os
import platform
import socket
import subprocess
import sys
import time


# ──────────────────────────────────────────────────────────────────────────
# Git / environment helpers
# ──────────────────────────────────────────────────────────────────────────
def get_git_commit(repo_dir=None):
    """Return the current git commit hash, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def git_is_dirty(repo_dir=None):
    """Return True if the git working tree has uncommitted changes, None if unknown."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_dir, stderr=subprocess.DEVNULL
        )
        return len(out.decode().strip()) > 0
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint discovery / validation / auto-resume
# ──────────────────────────────────────────────────────────────────────────
def is_valid_checkpoint(ckpt_dir):
    """A checkpoint directory is usable if it has an HF ``config.json`` plus at
    least one weight shard (``*.safetensors`` / ``*.bin``) or a LoRA ``adapter/``
    subdir. Returns False for missing/incomplete/corrupted directories instead
    of raising, so callers can skip and fall back to an earlier checkpoint.
    """
    try:
        if not os.path.isdir(ckpt_dir):
            return False
        if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
            return False
        names = os.listdir(ckpt_dir)
        has_weights = any(n.endswith((".safetensors", ".bin")) for n in names)
        has_adapter = os.path.isdir(os.path.join(ckpt_dir, "adapter"))
        return bool(has_weights or has_adapter)
    except Exception:
        return False


def read_checkpoint_step(run_dir):
    """Return the last-saved RL step from ``<run_dir>/ckpt/metadata.json``.

    Returns None if the metadata is missing or corrupted (caller decides what
    to do — typically start from scratch).
    """
    meta_path = os.path.join(run_dir, "ckpt", "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            data = json.load(f)
        return int(data["current_epoch"])
    except Exception:
        return None


def find_latest_checkpoint(exp_base, run_prefix=None):
    """Scan ``exp_base`` for run directories that hold a valid checkpoint and
    return ``(run_dir, step)`` for the most recently saved one, else
    ``(None, None)``.

    ``run_prefix`` optionally restricts to run names starting with that prefix.
    """
    if not exp_base or not os.path.isdir(exp_base):
        return None, None

    candidates = []
    for name in os.listdir(exp_base):
        if run_prefix and not name.startswith(run_prefix):
            continue
        run_dir = os.path.join(exp_base, name)
        if not os.path.isdir(run_dir):
            continue
        step = read_checkpoint_step(run_dir)
        if step is None:
            continue
        meta_path = os.path.join(run_dir, "ckpt", "metadata.json")
        candidates.append((os.path.getmtime(meta_path), run_dir, step))

    if not candidates:
        return None, None
    candidates.sort(key=lambda c: c[0])
    _, run_dir, step = candidates[-1]
    return run_dir, step


def resolve_auto_resume(project_dir, optimized_name):
    """Decide whether to resume an existing run pinned to ``project_dir``.

    Returns ``(should_resume, last_step)``. ``should_resume`` is True only when
    the run already holds a *valid* checkpoint; ``last_step`` is the last
    completed RL step (the caller continues at ``last_step + 1``).
    """
    step = read_checkpoint_step(project_dir)
    ckpt_dir = os.path.join(project_dir, "ckpt", optimized_name)
    if step is not None and is_valid_checkpoint(ckpt_dir):
        return True, step
    return False, None


# ──────────────────────────────────────────────────────────────────────────
# CUDA memory observability
# ──────────────────────────────────────────────────────────────────────────
def log_cuda_memory(tag="", reset_peak=False, printer=print):
    """Log allocated / reserved / peak CUDA memory. Never raises.

    If torch or CUDA is unavailable, prints a clean one-line notice instead of
    crashing, so the same call site is safe on CPU-only nodes.
    """
    try:
        import torch
    except Exception:
        printer("CUDA unavailable; skipping memory logging.")
        return
    if not torch.cuda.is_available():
        printer("CUDA unavailable; skipping memory logging.")
        return

    dev = torch.cuda.current_device()
    gb = 1024 ** 3
    alloc = torch.cuda.memory_allocated(dev) / gb
    reserved = torch.cuda.memory_reserved(dev) / gb
    peak_alloc = torch.cuda.max_memory_allocated(dev) / gb
    peak_reserved = torch.cuda.max_memory_reserved(dev) / gb
    label = f" {tag}" if tag else ""
    printer(
        f"[mem]{label} alloc={alloc:.2f}G reserved={reserved:.2f}G "
        f"peak_alloc={peak_alloc:.2f}G peak_reserved={peak_reserved:.2f}G"
    )
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(dev)


# ──────────────────────────────────────────────────────────────────────────
# Reproducibility / run metadata
# ──────────────────────────────────────────────────────────────────────────
def collect_env_metadata(repo_dir=None):
    """Return a dict describing the runtime environment (best-effort)."""
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": " ".join(sys.argv),
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "git_commit": get_git_commit(repo_dir),
        "git_dirty": git_is_dirty(repo_dir),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = torch.version.cuda
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        meta["torch_version"] = None
        meta["cuda_available"] = None
    return meta


def write_run_metadata(project_dir, resolved_config=None, config_path=None,
                       seed=None, repo_dir=None):
    """Write ``<project_dir>/run_metadata.json`` capturing enough to reproduce
    the run. Returns the path written.
    """
    os.makedirs(project_dir, exist_ok=True)
    meta = collect_env_metadata(repo_dir=repo_dir)
    meta["config_path"] = config_path
    meta["seed"] = seed
    meta["resolved_config"] = resolved_config
    path = os.path.join(project_dir, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


# ──────────────────────────────────────────────────────────────────────────
# Failure tolerance — fail early with actionable messages
# ──────────────────────────────────────────────────────────────────────────
def validate_gpu_requested(require_cuda=True):
    """Fail early (with a clear message) when GPU training is requested but no
    CUDA device is visible, instead of a late, opaque CUDA error.
    """
    import torch
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training was requested, but torch.cuda.is_available() is False.\n"
            "Check that this job was submitted with --gres=gpu and that CUDA "
            "modules are loaded (e.g. `module add cuda`)."
        )


def oom_hint():
    """Return a multi-line, repo-relevant hint for CUDA OOM situations."""
    return (
        "CUDA out of memory. Things to try (cheapest first):\n"
        "  - lower rollout.max_active / rollout.num_task_per_step\n"
        "  - lower the max-token schedule (max_token_schedule.end / evaluation.max_token)\n"
        "  - reduce training.batch_size_lm and raise training.gradient_accumulation_steps\n"
        "  - keep DeepSpeed offload on (training.offload_optimizer_device=cpu, offload_param_device=cpu)\n"
        "  - use a smaller student model or a larger GPU / MIG slice"
    )
