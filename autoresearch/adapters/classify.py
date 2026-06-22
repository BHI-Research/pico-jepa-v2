"""Adapter for the classification phase.

Supports N submodels with different data-partitioning strategies
(disjoint, bagging, stratified, random subset). Each submodel is trained as
its own subprocess invocation of ``app/classify_videos.py``.

The adapter:
1. Builds N per-submodel CSVs in ``work_dir/csvs/`` from the train_indices
   provided by ``prepare.Splits``.
2. Builds N per-submodel YAML configs.
3. Runs each one sequentially (GPU memory permits only one at a time on the
   1060), capturing metrics.
4. Aggregates ``mean_val_acc`` across submodels.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CLASSIFY_SCRIPT = os.path.join(PROJECT_ROOT, "app", "classify_videos.py")


def _stratified_disjoint_partition(
    df: pd.DataFrame, indices: Sequence[int], num_models: int, rng: np.random.Generator
) -> List[List[int]]:
    """Each submodel gets a disjoint stratified slice of train_indices."""
    sub = df.iloc[list(indices)]
    buckets: List[List[int]] = [[] for _ in range(num_models)]
    for _, group in sub.groupby("label", sort=True):
        positions = group.index.to_list()
        rng.shuffle(positions)
        for i, pos in enumerate(positions):
            buckets[i % num_models].append(int(pos))
    for b in buckets:
        b.sort()
    return buckets


def _stratified_bagging_partition(
    df: pd.DataFrame, indices: Sequence[int], num_models: int, rng: np.random.Generator
) -> List[List[int]]:
    """Each submodel gets a bootstrap (with replacement) of train_indices."""
    n = len(indices)
    arr = np.array(indices)
    return [sorted(arr[rng.integers(0, n, size=n)].tolist()) for _ in range(num_models)]


def _stratified_random_subset(
    df: pd.DataFrame, indices: Sequence[int], num_models: int, rng: np.random.Generator,
    frac: float = 0.6,
) -> List[List[int]]:
    """Each submodel gets a random fraction of train_indices (without replacement)."""
    arr = np.array(indices)
    k = max(1, int(round(frac * len(arr))))
    return [sorted(rng.choice(arr, size=k, replace=False).tolist()) for _ in range(num_models)]


def _stratified_by_class_partition(
    df: pd.DataFrame, indices: Sequence[int], num_models: int, rng: np.random.Generator
) -> List[List[int]]:
    """Each submodel gets a disjoint set of CLASSES (every video of that class).

    With 30 classes and num_models=4, each submodel sees ~7-8 classes. This is
    the most aggressive partition; useful for testing whether class-specialized
    submodels generalize via the encoder.
    """
    sub = df.iloc[list(indices)]
    classes = sorted(sub["label"].unique().tolist())
    rng.shuffle(classes)
    class_groups: List[List[int]] = [[] for _ in range(num_models)]
    for i, c in enumerate(classes):
        class_groups[i % num_models].append(c)
    buckets: List[List[int]] = []
    for grp in class_groups:
        mask = sub["label"].isin(grp)
        buckets.append(sorted(sub.index[mask].tolist()))
    return buckets


PARTITION_STRATEGIES = {
    "disjoint": _stratified_disjoint_partition,
    "bagging": _stratified_bagging_partition,
    "random-subset": _stratified_random_subset,
    "stratified-by-class": _stratified_by_class_partition,
}


def write_partition_csv(
    classify_csv_path: str, indices: Sequence[int], out_path: str
) -> str:
    df = pd.read_csv(classify_csv_path)
    df.iloc[list(indices)].to_csv(out_path, index=False)
    return out_path


def make_submodel_csvs(
    classify_csv_path: str,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    num_models: int,
    partition_strategy: str,
    work_dir: str,
    seed: int,
) -> Tuple[List[str], str]:
    """Returns (per-submodel train CSV paths, val CSV path)."""
    df = pd.read_csv(classify_csv_path)
    rng = np.random.default_rng(seed)

    if partition_strategy not in PARTITION_STRATEGIES:
        raise ValueError(
            f"Unknown partition strategy {partition_strategy!r}; "
            f"valid: {list(PARTITION_STRATEGIES)}"
        )
    partition_fn = PARTITION_STRATEGIES[partition_strategy]
    buckets = partition_fn(df, train_indices, num_models, rng)

    csv_dir = os.path.join(work_dir, "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    train_csvs: List[str] = []
    for i, bucket in enumerate(buckets, start=1):
        path = os.path.join(csv_dir, f"submodel_{i}_train.csv")
        df.iloc[bucket].to_csv(path, index=False)
        train_csvs.append(path)

    val_csv = os.path.join(csv_dir, "val.csv")
    df.iloc[list(val_indices)].to_csv(val_csv, index=False)

    return train_csvs, val_csv


# V-JEPA-style ensembles benefit from per-submodel diversity beyond just the
# data partition: different seeds, different LRs, sometimes different head
# architectures. Without this, frozen-encoder submodels tend to converge to
# nearly identical classifiers and hard/soft voting collapses to single-model
# behavior. This factory applies bounded jitter to a shared base_config so
# every submodel gets a distinct-but-still-valid configuration.

_HEAD_TYPE_ROTATION: Tuple[str, ...] = ("linear", "mlp_2layer", "attentive")


def diversify_submodel_config(
    base_config: Dict[str, Any], submodel_idx: int, master_seed: int = 1337,
) -> Dict[str, Any]:
    """Return a config overlay that diversifies submodel ``submodel_idx``.

    Knobs varied (all bounded by the declared search_space ranges):
      * ``seed`` — deterministic per submodel.
      * ``learning_rate_classifier`` — log-uniform jitter ~±30%.
      * ``head_type`` — rotates linear / mlp_2layer / attentive.
      * ``freeze_encoder`` — alternates True/False.
      * ``num_epochs_classify`` — ±2 epochs around the base value.

    The base config is preserved when ``submodel_idx == 1`` so the first
    submodel acts as a "control" with the same config as a single-model run.
    """
    cfg = dict(base_config)
    cfg["seed"] = int(master_seed) + 1000 * int(submodel_idx)

    if submodel_idx <= 1:
        # First submodel keeps the base config (acts as a "reference" run);
        # diversity starts from the second.
        return cfg

    rng = np.random.default_rng(cfg["seed"])

    base_lr = float(base_config.get("learning_rate_classifier", 1e-4))
    log_jitter = float(rng.uniform(-0.3, 0.3))
    cfg["learning_rate_classifier"] = float(base_lr * np.exp(log_jitter))
    cfg["learning_rate_classifier"] = max(1e-5, min(1e-3, cfg["learning_rate_classifier"]))

    cfg["head_type"] = _HEAD_TYPE_ROTATION[(submodel_idx - 1) % len(_HEAD_TYPE_ROTATION)]
    cfg["freeze_encoder"] = bool(base_config.get("freeze_encoder", True)) ^ (submodel_idx % 2 == 0)

    # Pick a head_attn_heads value compatible with vit_embed_dim. Only
    # consumed when head_type=="attentive"; benign otherwise.
    embed = int(base_config.get("vit_embed_dim", 192))
    valid_heads = [h for h in (2, 4, 8) if embed % h == 0]
    if valid_heads:
        cfg["head_attn_heads"] = int(rng.choice(valid_heads))

    base_epochs = int(base_config.get("num_epochs_classify", 4))
    delta_epochs = int(rng.integers(-2, 3))  # in [-2, +2]
    cfg["num_epochs_classify"] = max(2, min(8, base_epochs + delta_epochs))

    return cfg


def run_one_submodel(
    base_config: Dict[str, Any],
    train_csv: str,
    val_csv: str,
    classifier_save_path: str,
    work_dir: str,
    submodel_idx: int,
    timeout_s: Optional[float] = None,
    python_executable: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = dict(base_config)
    # The new CSV holds both train and val rows; classify_videos.py will do its
    # own 80/20 random_split. To use prepare.py's split we'd need to wire the
    # indices through, but for V1 we keep classify_videos.py untouched and
    # accept its internal split — the submodel sees only its assigned partition,
    # which is what matters for the hypothesis.
    cfg["classify_csv_path"] = train_csv
    cfg["classify_video_dir"] = base_config.get(
        "classify_video_dir", base_config.get("video_dir")
    )
    cfg["classifier_save_path"] = classifier_save_path

    config_path = os.path.join(work_dir, f"submodel_{submodel_idx}_config.yaml")
    metrics_path = os.path.join(work_dir, f"submodel_{submodel_idx}_metrics.json")
    log_path = os.path.join(work_dir, f"submodel_{submodel_idx}_stdout.log")

    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    cmd = [
        python_executable or sys.executable,
        CLASSIFY_SCRIPT,
        "--config_path", config_path,
        "--output_json", metrics_path,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])

    started = time.perf_counter()
    status = "completed"
    stdout_tail = ""
    stderr_tail = ""
    try:
        with open(log_path, "wb") as logf:
            proc = subprocess.run(
                cmd, cwd=PROJECT_ROOT, stdout=logf, stderr=subprocess.PIPE,
                timeout=timeout_s, check=False,
            )
        with open(log_path, "rb") as logf:
            stdout_tail = logf.read()[-4096:].decode("utf-8", errors="replace")
        stderr_tail = (proc.stderr or b"")[-4096:].decode("utf-8", errors="replace")
        if proc.returncode != 0:
            blob = (stdout_tail + stderr_tail).lower()
            if "out of memory" in blob:
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

    return {
        "status": status,
        "metrics": metrics,
        "wallclock_s": wallclock_s,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "config_path": config_path,
        "classifier_save_path": classifier_save_path,
    }


def run_classify(
    base_config: Dict[str, Any],
    classify_csv_path: str,
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    num_models: int,
    partition_strategy: str,
    work_dir: str,
    seed: int = 1337,
    per_submodel_timeout_s: Optional[float] = None,
    python_executable: Optional[str] = None,
) -> Dict[str, Any]:
    """Train num_models submodels and aggregate metrics.

    Returns:
        {
            "status": "completed" | "partial" | "failed",
            "submodels": [ {status, metrics, classifier_save_path}, ... ],
            "mean_val_acc": float,
            "wallclock_s": float,
        }
    """
    started = time.perf_counter()
    train_csvs, val_csv = make_submodel_csvs(
        classify_csv_path=classify_csv_path,
        train_indices=train_indices,
        val_indices=val_indices,
        num_models=num_models,
        partition_strategy=partition_strategy,
        work_dir=work_dir,
        seed=seed,
    )

    submodel_results = []
    val_accs: List[float] = []
    any_completed = False
    any_failed = False
    diversify = bool(base_config.get("diversify_submodels", False))

    for i, train_csv in enumerate(train_csvs, start=1):
        ckpt_path = os.path.join(work_dir, f"submodel_{i}.pth")
        # Per-submodel config: diversified jitter when the user opted in,
        # otherwise the shared base_config (legacy behavior).
        per_submodel_cfg = (
            diversify_submodel_config(base_config, submodel_idx=i, master_seed=seed)
            if diversify else dict(base_config)
        )
        # The seed for torch/numpy in the subprocess: always per-submodel so
        # that even non-diversified runs get reproducible-but-distinct random
        # initialization of the classifier head.
        submodel_seed = int(per_submodel_cfg.get("seed", seed + i * 1000))

        result = run_one_submodel(
            base_config=per_submodel_cfg,
            train_csv=train_csv,
            val_csv=val_csv,
            classifier_save_path=ckpt_path,
            work_dir=work_dir,
            submodel_idx=i,
            timeout_s=per_submodel_timeout_s,
            python_executable=python_executable,
            seed=submodel_seed,
        )
        submodel_results.append(result)
        if result["status"] == "completed":
            any_completed = True
            v = result["metrics"].get("best_val_acc") or result["metrics"].get("final_val_acc")
            if v is not None:
                val_accs.append(float(v))
        else:
            any_failed = True

    overall_status = (
        "completed" if any_completed and not any_failed
        else ("partial" if any_completed else "failed")
    )
    mean_val_acc = float(np.mean(val_accs)) if val_accs else 0.0

    return {
        "status": overall_status,
        "submodels": submodel_results,
        "mean_val_acc": mean_val_acc,
        "wallclock_s": time.perf_counter() - started,
        "train_csvs": train_csvs,
        "val_csv": val_csv,
    }
