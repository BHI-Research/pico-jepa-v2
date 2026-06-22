"""Ensemble aggregation methods.

Consumes a (num_models, num_videos, num_classes) probability tensor produced
by ``adapters/infer.py`` and returns ensemble predictions. Supports:

- ``hard_vote``: argmax per model, majority vote.
- ``soft_vote``: average probabilities, argmax.
- ``weighted_vote``: weighted average by per-model val accuracy.
- ``stacking``: sklearn LogisticRegression / LinearSVC trained on val
  probabilities (concatenated across models) and applied at test time.

Bagging is not a method here — it is a partition strategy in
``adapters/classify.py``. Stacking and voting can be applied on top of it.
Boosting (V2) needs ``classify_videos.py`` to accept sample_weights.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


VALID_AGGREGATIONS = ("hard_vote", "soft_vote", "weighted_vote", "stacking")


def _concat_features(probs: np.ndarray) -> np.ndarray:
    """(N, M, C) -> (M, N*C) feature matrix for stacking."""
    n_models, n_videos, n_classes = probs.shape
    return probs.transpose(1, 0, 2).reshape(n_videos, n_models * n_classes)


def hard_vote(probs: np.ndarray) -> np.ndarray:
    """(N, M, C) -> (M,) majority vote with confidence-weighted tiebreak."""
    n_models, n_videos, n_classes = probs.shape
    preds = np.argmax(probs, axis=2)  # (N, M)
    out = np.zeros(n_videos, dtype=np.int64)
    for j in range(n_videos):
        counts = np.bincount(preds[:, j], minlength=n_classes)
        max_count = counts.max()
        candidates = np.where(counts == max_count)[0]
        if len(candidates) == 1:
            out[j] = candidates[0]
        else:
            # Tiebreak by mean confidence on tied classes.
            mean_conf = probs[:, j, candidates].mean(axis=0)
            out[j] = candidates[int(np.argmax(mean_conf))]
    return out


def soft_vote(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """(N, M, C) -> (M,)."""
    if temperature != 1.0 and temperature > 0:
        scaled = probs ** (1.0 / temperature)
        scaled = scaled / scaled.sum(axis=2, keepdims=True)
    else:
        scaled = probs
    avg = scaled.mean(axis=0)  # (M, C)
    return np.argmax(avg, axis=1)


def weighted_vote(probs: np.ndarray, weights: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    """(N, M, C) -> (M,) with per-model weight."""
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (probs.shape[0],):
        raise ValueError(f"weights must have shape ({probs.shape[0]},), got {w.shape}.")
    w = np.clip(w, 1e-6, None)
    w = w / w.sum()
    if temperature != 1.0 and temperature > 0:
        scaled = probs ** (1.0 / temperature)
        scaled = scaled / scaled.sum(axis=2, keepdims=True)
    else:
        scaled = probs
    avg = (w[:, None, None] * scaled).sum(axis=0)  # (M, C)
    return np.argmax(avg, axis=1)


def stacking_fit(
    val_probs: np.ndarray, val_labels: np.ndarray, meta_learner: str = "logreg"
):
    """Fit a meta-learner on validation probabilities.

    Imports sklearn lazily so a heuristic-only run doesn't pay for it.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.svm import LinearSVC  # noqa: F401

    X = _concat_features(val_probs)
    y = np.asarray(val_labels, dtype=np.int64)

    if meta_learner == "logreg":
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=2000)
    elif meta_learner == "svm":
        from sklearn.svm import LinearSVC
        clf = LinearSVC(max_iter=5000)
    else:
        raise ValueError(f"Unknown meta_learner {meta_learner!r}; valid: logreg, svm.")
    clf.fit(X, y)
    return clf


def stacking_predict(clf, test_probs: np.ndarray) -> np.ndarray:
    X = _concat_features(test_probs)
    return clf.predict(X).astype(np.int64)


def aggregate(
    method: str,
    probs: np.ndarray,
    weights: Optional[Sequence[float]] = None,
    temperature: float = 1.0,
    val_probs: Optional[np.ndarray] = None,
    val_labels: Optional[Sequence[int]] = None,
    meta_learner: str = "logreg",
) -> Dict[str, Any]:
    """Top-level aggregation entry point.

    Returns dict with key 'preds' (M,) and optional 'classifier' for stacking
    so the same fit can be reused on a different test set (e.g., holdout).
    """
    if method == "hard_vote":
        return {"preds": hard_vote(probs)}
    if method == "soft_vote":
        return {"preds": soft_vote(probs, temperature=temperature)}
    if method == "weighted_vote":
        if weights is None:
            raise ValueError("weighted_vote requires weights.")
        return {"preds": weighted_vote(probs, weights, temperature=temperature)}
    if method == "stacking":
        if val_probs is None or val_labels is None:
            raise ValueError("stacking requires val_probs and val_labels.")
        clf = stacking_fit(val_probs, np.asarray(val_labels), meta_learner=meta_learner)
        return {"preds": stacking_predict(clf, probs), "classifier": clf}
    raise ValueError(f"Unknown aggregation {method!r}; valid: {VALID_AGGREGATIONS}.")


def accuracy(preds: np.ndarray, labels: Sequence[int]) -> float:
    labels_arr = np.asarray(labels, dtype=np.int64)
    return float((preds == labels_arr).mean())


def bootstrap_gap_ci(
    ensemble_preds: np.ndarray,
    general_preds: np.ndarray,
    labels: Sequence[int],
    n_boot: int = 1000,
    seed: int = 1337,
) -> Dict[str, float]:
    """Bootstrap CI for the gap (ensemble_acc - general_acc).

    Used in phase 4 to detect non-significant improvements.
    """
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels, dtype=np.int64)
    n = len(labels_arr)
    gaps = np.empty(n_boot, dtype=np.float32)
    ens_correct = (ensemble_preds == labels_arr).astype(np.int8)
    gen_correct = (general_preds == labels_arr).astype(np.int8)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        gaps[b] = ens_correct[idx].mean() - gen_correct[idx].mean()
    return {
        "gap_mean": float(gaps.mean()),
        "gap_ci_low": float(np.percentile(gaps, 2.5)),
        "gap_ci_high": float(np.percentile(gaps, 97.5)),
        "ensemble_top1": float(ens_correct.mean()),
        "general_top1": float(gen_correct.mean()),
    }
