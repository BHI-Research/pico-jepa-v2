"""Adapter for the JEPA pre-training phase.

Wraps ``app/train.py`` as a subprocess so OOM/SIGKILL/timeouts don't take
down the search loop. Returns a normalized metrics dict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_SCRIPT = os.path.join(PROJECT_ROOT, "app", "train.py")


def materialize_config(config: Dict[str, Any], work_dir: str) -> str:
    os.makedirs(work_dir, exist_ok=True)
    config_path = os.path.join(work_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=True)
    return config_path


def run_pretrain(
    config: Dict[str, Any],
    work_dir: str,
    timeout_s: Optional[float] = None,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one pretrain experiment.

    Returns a dict with keys:
        status: 'completed' | 'failed' | 'timeout' | 'oom'
        metrics: {final_loss, wallclock_s, encoder_save_path, ...}
        stdout_tail: last ~4KB of stdout (debug)
        stderr_tail: last ~4KB of stderr (debug)
    """
    config_path = materialize_config(config, work_dir)
    metrics_path = os.path.join(work_dir, "metrics.json")
    log_path = os.path.join(work_dir, "stdout.log")

    cmd = [
        python_executable or sys.executable,
        TRAIN_SCRIPT,
        "--config_path", config_path,
        "--output_json", metrics_path,
    ]

    started = time.perf_counter()
    status = "completed"
    stdout_tail = ""
    stderr_tail = ""
    try:
        with open(log_path, "wb") as logf:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=logf,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
        with open(log_path, "rb") as logf:
            data = logf.read()
            stdout_tail = data[-4096:].decode("utf-8", errors="replace")
        stderr_tail = (proc.stderr or b"")[-4096:].decode("utf-8", errors="replace")

        if proc.returncode != 0:
            # Heuristic OOM detection for downstream feedback to the proposer.
            blob = (stdout_tail + stderr_tail).lower()
            if "out of memory" in blob or "cuda oom" in blob or "cuda error: out of memory" in blob:
                status = "oom"
            else:
                status = "failed"
    except subprocess.TimeoutExpired:
        status = "timeout"
    wallclock_s = time.perf_counter() - started

    metrics: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except Exception:
            pass
    metrics.setdefault("wallclock_s", wallclock_s)

    return {
        "status": status,
        "metrics": metrics,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "wallclock_s": wallclock_s,
        "config_path": config_path,
    }
