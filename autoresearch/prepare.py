"""LOCKED: dataset preparation, triple split, and official fitness metric.

This module is the integrity boundary of the autoresearch system. The proposer
must NOT modify it. The hash of this file is verified by the search loop at
startup; if it changes between resumes, the loop refuses to continue without
explicit user confirmation.

The triple split is deterministic (seed fixed below). The holdout split is
guarded by ``AUTORESEARCH_PHASE`` and may only be loaded during phase 4
(final hypothesis evaluation). Any other phase that attempts to read it
raises ``HoldoutAccessError``.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


SEED = 1337
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
HOLDOUT_FRAC = 0.20
PROBE_VIDEOS = 500
PROBE_EPOCHS = 1


class HoldoutAccessError(RuntimeError):
    """Raised when code outside phase 4 attempts to read the holdout split."""


@dataclass(frozen=True)
class Splits:
    train_indices: Tuple[int, ...]
    val_indices: Tuple[int, ...]
    holdout_indices: Tuple[int, ...]
    csv_path: str
    num_classes: int


def _read_classify_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if not {"video_name", "label"}.issubset(df.columns):
        raise ValueError(
            f"Classification CSV at {csv_path} must have 'video_name' and 'label' columns."
        )
    df = df.dropna(subset=["video_name", "label"]).reset_index(drop=True)
    df["label"] = pd.to_numeric(df["label"]).astype(int)
    return df


def make_splits(classify_csv_path: str) -> Splits:
    """Compute the deterministic 60/20/20 stratified split.

    Stratification preserves class balance within each split so that a small
    holdout still represents every class.
    """
    df = _read_classify_csv(classify_csv_path)
    rng = np.random.default_rng(SEED)
    train_idx: List[int] = []
    val_idx: List[int] = []
    holdout_idx: List[int] = []

    for _, group in df.groupby("label", sort=True):
        positions = group.index.to_numpy()
        rng.shuffle(positions)
        n = len(positions)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # Whatever remains goes to holdout (covers small remainders).
        train_idx.extend(positions[:n_train].tolist())
        val_idx.extend(positions[n_train : n_train + n_val].tolist())
        holdout_idx.extend(positions[n_train + n_val :].tolist())

    train_idx.sort()
    val_idx.sort()
    holdout_idx.sort()

    return Splits(
        train_indices=tuple(train_idx),
        val_indices=tuple(val_idx),
        holdout_indices=tuple(holdout_idx),
        csv_path=classify_csv_path,
        num_classes=int(df["label"].max() + 1),
    )


def load_holdout_indices(splits: Splits) -> Tuple[int, ...]:
    """Return holdout indices. Raises unless ``AUTORESEARCH_PHASE`` == 4.

    This is the only sanctioned way to access the holdout. The search loop
    sets the env var only when it disposes of phase 4 evaluation.
    """
    if os.environ.get("AUTORESEARCH_PHASE") != "4":
        raise HoldoutAccessError(
            "Holdout split is only loadable in phase 4. Set AUTORESEARCH_PHASE=4 "
            "in the runner that performs the final ensemble-vs-general evaluation."
        )
    return splits.holdout_indices


def fitness_gap(ensemble_top1: float, general_top1: float) -> float:
    """Official metric: gap between ensemble and single general model on holdout.

    Positive gap supports the central hypothesis. The search loop reports a
    bootstrap CI for this gap to detect non-significant improvements.
    """
    return float(ensemble_top1) - float(general_top1)


def file_hash() -> str:
    """SHA256 of this very file. Used by the search loop integrity check."""
    here = os.path.abspath(__file__)
    with open(here, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
