"""Ratchet: per-phase improvement criteria + git tag of best artifacts.

Karpathy's autoresearch ratchet operates on code (commit if val_bpb improves,
git reset if not). For pico-JEPA we ratchet on YAML configs and checkpoints:
each phase has a criterion; on improvement we promote the iteration's best
artifacts into a stable location (``configs/best/phase{N}.yaml`` and
``models_best/phase{N}.pth``) and tag the commit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Sequence

from autoresearch.ledger import Ledger


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEST_CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs", "best")
BEST_MODELS_DIR = os.path.join(PROJECT_ROOT, "models_best")


# Improvement thresholds per phase (in absolute units, e.g., 0.005 = 0.5pp).
THRESHOLDS = {
    "pretrain": 0.005,   # +0.5pp probe top1
    "classify": 0.003,   # +0.3pp mean val acc
    "ensemble": 0.002,   # +0.2pp val acc
    "phase4":   1e-9,    # any improvement in gap
}

# Wallclock penalty: pretrain improvement only counts if not >1.5x slower.
WALLCLOCK_RATIO_LIMIT = 1.5


def is_improvement(
    phase: str,
    new_score: float,
    new_wallclock_s: float,
    ledger: Ledger,
) -> bool:
    cur = ledger.get_ratchet(phase)
    if cur is None or cur["current_score"] is None:
        return True
    delta = new_score - float(cur["current_score"])
    threshold = THRESHOLDS.get(phase, 0.0)
    if delta < threshold:
        return False
    if phase == "pretrain":
        # Reject improvements that come at >1.5× the baseline wallclock.
        baseline_wc = _wallclock_of(ledger, cur["best_experiment_id"])
        if baseline_wc and new_wallclock_s > baseline_wc * WALLCLOCK_RATIO_LIMIT:
            return False
    return True


def _wallclock_of(ledger: Ledger, exp_id: Optional[int]) -> Optional[float]:
    if exp_id is None:
        return None
    rows = ledger.recent_history(limit=1000)
    for r in rows:
        if r["id"] == exp_id:
            return r.get("wallclock_s")
    return None


def promote_artifacts(
    phase: str,
    experiment_id: int,
    config_path: str,
    artifact_paths: Sequence[str],
    score: float,
    ledger: Ledger,
    git_commit: bool = True,
    per_artifact_configs: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Copy iteration's config/checkpoints to the stable best/ location and tag git.

    Args:
        config_path: the "main" config (used by single-artifact phases and by
            consumers that don't know about heterogeneous submodels).
        artifact_paths: list of checkpoint paths to promote.
        per_artifact_configs: optional list parallel to ``artifact_paths``. When
            provided, each entry is copied to ``configs/best/{phase}_{i}.yaml``
            so each submodel keeps its own architecture-defining config
            (head_type, freeze_encoder, lr, seed, ...). Required for
            heterogeneous ensembles where head_type varies by submodel — the
            ensemble loader uses these to reconstruct each VideoClassifier
            with the correct architecture before load_state_dict.
    """
    os.makedirs(BEST_CONFIGS_DIR, exist_ok=True)
    os.makedirs(BEST_MODELS_DIR, exist_ok=True)

    promoted_config = os.path.join(BEST_CONFIGS_DIR, f"{phase}.yaml")
    if os.path.exists(config_path):
        shutil.copy2(config_path, promoted_config)

    promoted_per_artifact_configs: List[str] = []
    if per_artifact_configs is not None:
        for i, cfg_src in enumerate(per_artifact_configs, start=1):
            if cfg_src is None or not os.path.exists(cfg_src):
                continue
            target_cfg = os.path.join(BEST_CONFIGS_DIR, f"{phase}_{i}.yaml")
            shutil.copy2(cfg_src, target_cfg)
            promoted_per_artifact_configs.append(target_cfg)

    promoted_models: List[str] = []
    for i, ap in enumerate(artifact_paths, start=1):
        if not os.path.exists(ap):
            continue
        # For classify there are N submodels; for pretrain a single encoder.
        suffix = "" if len(artifact_paths) == 1 else f"_{i}"
        target = os.path.join(BEST_MODELS_DIR, f"{phase}{suffix}.pth")
        shutil.copy2(ap, target)
        promoted_models.append(target)

    cur = ledger.get_ratchet(phase)
    baseline = cur["baseline_score"] if cur else score
    ledger.set_ratchet(phase, experiment_id, score=score, baseline=baseline, dirty=False)

    # Invalidate downstream phases — the baseline they trained against is now stale.
    if phase == "pretrain":
        ledger.mark_dirty(["classify", "ensemble"])
    elif phase == "classify":
        ledger.mark_dirty(["ensemble"])

    git_info = {"committed": False, "tag": None}
    if git_commit:
        git_info = _git_commit_and_tag(phase, score)

    return {
        "promoted_config": promoted_config,
        "promoted_models": promoted_models,
        "promoted_per_artifact_configs": promoted_per_artifact_configs,
        **git_info,
    }


def _git_commit_and_tag(phase: str, score: float) -> Dict[str, Any]:
    try:
        # Stage only the promoted files — never the ledger or experiment artifacts.
        subprocess.run(
            ["git", "add", "configs/best", "models_best"],
            cwd=PROJECT_ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Check if anything changed (avoid empty commits).
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=PROJECT_ROOT,
        )
        if diff.returncode == 0:
            return {"committed": False, "tag": None, "reason": "no_changes_staged"}
        msg = f"autoresearch: {phase} improvement score={score:.4f}"
        subprocess.run(
            ["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        tag = f"autoresearch/best/{phase}"
        subprocess.run(
            ["git", "tag", "-f", tag], cwd=PROJECT_ROOT, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return {"committed": True, "tag": tag}
    except Exception as e:
        return {"committed": False, "tag": None, "reason": str(e)}
