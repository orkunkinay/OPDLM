"""CPU-only tests for the cluster helpers in cluster/cluster_utils.py.

Run with either:
    python -m pytest tests/test_cluster_utils.py
    python tests/test_cluster_utils.py        # no pytest needed

They use tiny dummy checkpoint directories (a config.json + a stub weight
file) — no GPU, no real model, no training.
"""

import json
import os
import sys

# Make the repo root importable when run directly (python tests/...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.cluster_utils import (
    is_valid_checkpoint,
    read_checkpoint_step,
    find_latest_checkpoint,
    resolve_auto_resume,
    log_cuda_memory,
    write_run_metadata,
)


def _make_run(base, name, step, optimized_name="optimized", valid=True,
              with_metadata=True):
    """Create a fake run dir <base>/<name> with ckpt/<optimized_name> and
    ckpt/metadata.json. If valid=False, omit the weight file (corrupt/incomplete)."""
    run_dir = os.path.join(base, name)
    ckpt_dir = os.path.join(run_dir, "ckpt", optimized_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump({"model_type": "dummy"}, f)
    if valid:
        with open(os.path.join(ckpt_dir, "model.safetensors"), "wb") as f:
            f.write(b"\x00\x01")
    if with_metadata:
        with open(os.path.join(run_dir, "ckpt", "metadata.json"), "w") as f:
            json.dump({"current_epoch": step, "last_save_name": optimized_name}, f)
    return run_dir, ckpt_dir


def test_is_valid_checkpoint(tmp_path):
    base = str(tmp_path)
    _, ckpt = _make_run(base, "run_ok", step=5, valid=True)
    assert is_valid_checkpoint(ckpt) is True

    # Missing weights -> incomplete -> invalid
    _, ckpt_bad = _make_run(base, "run_bad", step=3, valid=False)
    assert is_valid_checkpoint(ckpt_bad) is False

    # Nonexistent dir -> invalid (no crash)
    assert is_valid_checkpoint(os.path.join(base, "does_not_exist")) is False

    # LoRA-style adapter dir counts as valid
    adapter_run = os.path.join(base, "run_lora", "ckpt", "optimized")
    os.makedirs(os.path.join(adapter_run, "adapter"), exist_ok=True)
    with open(os.path.join(adapter_run, "config.json"), "w") as f:
        json.dump({}, f)
    assert is_valid_checkpoint(adapter_run) is True


def test_read_checkpoint_step(tmp_path):
    base = str(tmp_path)
    run_dir, _ = _make_run(base, "run", step=42)
    assert read_checkpoint_step(run_dir) == 42

    # No metadata -> None
    empty = os.path.join(base, "empty")
    os.makedirs(empty, exist_ok=True)
    assert read_checkpoint_step(empty) is None

    # Corrupted metadata -> None (no crash)
    bad = os.path.join(base, "corrupt")
    os.makedirs(os.path.join(bad, "ckpt"), exist_ok=True)
    with open(os.path.join(bad, "ckpt", "metadata.json"), "w") as f:
        f.write("{ not json")
    assert read_checkpoint_step(bad) is None


def test_find_latest_checkpoint(tmp_path):
    base = str(tmp_path)
    r1, _ = _make_run(base, "run_a", step=10)
    r2, _ = _make_run(base, "run_b", step=20)
    # Make run_b the most recently saved one.
    os.utime(os.path.join(r1, "ckpt", "metadata.json"), (1000, 1000))
    os.utime(os.path.join(r2, "ckpt", "metadata.json"), (2000, 2000))

    run_dir, step = find_latest_checkpoint(base)
    assert run_dir == r2 and step == 20

    # Prefix filter restricts the search.
    run_dir, step = find_latest_checkpoint(base, run_prefix="run_a")
    assert run_dir == r1 and step == 10

    # Empty / missing base -> (None, None)
    assert find_latest_checkpoint(os.path.join(base, "nope")) == (None, None)


def test_resolve_auto_resume(tmp_path):
    base = str(tmp_path)
    run_dir, _ = _make_run(base, "stable_run", step=7)
    should, last = resolve_auto_resume(run_dir, "optimized")
    assert should is True and last == 7

    # Fresh dir (no checkpoint yet) -> do not resume.
    fresh = os.path.join(base, "fresh")
    os.makedirs(fresh, exist_ok=True)
    assert resolve_auto_resume(fresh, "optimized") == (False, None)

    # Metadata present but weights missing (interrupted save) -> do not resume.
    bad_dir, _ = _make_run(base, "halfsaved", step=9, valid=False)
    assert resolve_auto_resume(bad_dir, "optimized") == (False, None)


def test_log_cuda_memory_no_crash():
    """Must never raise; on a CPU-only box it prints a clean notice."""
    lines = []
    log_cuda_memory("unit-test", printer=lines.append)
    assert len(lines) == 1
    try:
        import torch
        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    if not cuda:
        assert "CUDA unavailable" in lines[0]
    else:
        assert "[mem]" in lines[0]


def test_write_run_metadata(tmp_path):
    project = str(tmp_path / "run")
    path = write_run_metadata(
        project,
        resolved_config={"experiment": {"project": project}},
        config_path="configs/rl_bd3lm.yaml",
        seed=10086,
    )
    assert os.path.exists(path)
    with open(path) as f:
        meta = json.load(f)
    for key in ("timestamp", "command", "python_version", "hostname",
                "seed", "resolved_config", "config_path"):
        assert key in meta
    assert meta["seed"] == 10086
    assert meta["config_path"] == "configs/rl_bd3lm.yaml"


# ── Standalone runner (no pytest required) ──────────────────────────────────
if __name__ == "__main__":
    import tempfile

    class _P:
        """Minimal tmp_path stand-in so tests run without pytest."""
        def __init__(self, p):
            self._p = p
        def __str__(self):
            return self._p
        def __truediv__(self, other):
            return _P(os.path.join(self._p, str(other)))

    tests = [
        ("is_valid_checkpoint", lambda d: test_is_valid_checkpoint(_P(d))),
        ("read_checkpoint_step", lambda d: test_read_checkpoint_step(_P(d))),
        ("find_latest_checkpoint", lambda d: test_find_latest_checkpoint(_P(d))),
        ("resolve_auto_resume", lambda d: test_resolve_auto_resume(_P(d))),
        ("log_cuda_memory_no_crash", lambda d: test_log_cuda_memory_no_crash()),
        ("write_run_metadata", lambda d: test_write_run_metadata(_P(d))),
    ]
    failures = 0
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(d)
                print(f"PASS  {name}")
            except Exception as e:
                failures += 1
                print(f"FAIL  {name}: {e!r}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
