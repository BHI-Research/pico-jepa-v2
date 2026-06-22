"""Hyperparameter search space per phase, tuned for GTX 1060 6GB.

Each phase exposes a ``sample`` function that takes a base config and an
``np.random.Generator`` and returns a config delta (just the keys that the
proposer chose to mutate). Validation also lives here: a config that exits
the declared range gets rejected by the runner before training starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class HParam:
    name: str
    values: Optional[List[Any]] = None  # categorical/discrete
    low: Optional[float] = None  # continuous
    high: Optional[float] = None
    log: bool = False
    is_int: bool = False

    def sample(self, rng: np.random.Generator) -> Any:
        if self.values is not None:
            return self.values[int(rng.integers(0, len(self.values)))]
        if self.low is None or self.high is None:
            raise ValueError(f"HParam {self.name} has no range or values.")
        if self.log:
            v = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        else:
            v = float(rng.uniform(self.low, self.high))
        return int(round(v)) if self.is_int else v

    def in_range(self, v: Any) -> bool:
        if self.values is not None:
            return v in self.values
        if self.low is None or self.high is None:
            return False
        return self.low <= float(v) <= self.high


# Phase 1: Pretrain JEPA. Conservative ranges that fit in 6GB VRAM.
PRETRAIN = {
    "learning_rate":            HParam("learning_rate", low=5e-5, high=5e-4, log=True),
    "predictor_lr_multiplier":  HParam("predictor_lr_multiplier", low=1.0, high=4.0),
    "mask_ratio":               HParam("mask_ratio", low=0.5, high=0.85),
    "ema_decay":                HParam("ema_decay", low=0.99, high=0.999),
    "weight_decay":             HParam("weight_decay", low=0.01, high=0.1),
    "vit_embed_dim":            HParam("vit_embed_dim", values=[96, 128, 192, 256]),
    "vit_depth":                HParam("vit_depth", values=[4, 6, 8, 10]),
    "vit_num_heads":            HParam("vit_num_heads", values=[4, 6, 8]),
    "predictor_depth":          HParam("predictor_depth", values=[1, 2, 3, 4]),
    "predictor_heads":          HParam("predictor_heads", values=[2, 4, 6, 8]),
    "vit_mlp_ratio":            HParam("vit_mlp_ratio", values=[2.0, 3.0, 4.0]),
    "batch_size":               HParam("batch_size", values=[6, 8, 10, 12]),
    # Pretrain epochs widened to accommodate larger pretrain datasets (14K+
    # videos): with the original [2,4,6,8] range, a base config of 20 would
    # be rejected by validate_config before training. 8/12 are cheap probes,
    # 16/20/24/32 are realistic full pretrains on K700-train.
    "num_epochs":               HParam("num_epochs", values=[8, 12, 16, 20, 24, 32]),
    "frames_per_clip":          HParam("frames_per_clip", values=[6, 8, 12]),
    # Phase B: V-JEPA-inspired pretrain knobs.
    "masking_strategy":         HParam("masking_strategy", values=["tubelet", "multiblock"]),
    "num_mask_blocks":          HParam("num_mask_blocks", values=[1, 2, 3, 4]),
    "ema_decay_end":            HParam("ema_decay_end", low=0.998, high=0.9999),
    "layerwise_lr_decay":       HParam("layerwise_lr_decay", low=0.7, high=1.0),
}

# Phase 2: Classify (per submodel + ensemble structure).
CLASSIFY = {
    "num_models":               HParam("num_models", values=[2, 3, 4, 5, 6, 8]),
    "partition_strategy":       HParam(
        "partition_strategy",
        values=["disjoint", "bagging", "stratified-by-class", "random-subset"],
    ),
    "freeze_encoder":           HParam("freeze_encoder", values=[True, False]),
    "learning_rate_classifier": HParam("learning_rate_classifier", low=1e-5, high=1e-3, log=True),
    "learning_rate_encoder_finetune": HParam("learning_rate_encoder_finetune", low=1e-6, high=1e-4, log=True),
    "num_epochs_classify":      HParam("num_epochs_classify", values=[2, 4, 6, 8]),
    "batch_size_classify":      HParam("batch_size_classify", values=[8, 16, 24]),
    "classify_weight_decay":    HParam("classify_weight_decay", low=0.01, high=0.1),
    # V-JEPA-inspired head architecture (Phase A):
    "head_type":                HParam("head_type", values=["linear", "mlp_2layer", "attentive"]),
    "head_attn_heads":          HParam("head_attn_heads", values=[2, 4, 8]),
    "head_mlp_ratio":           HParam("head_mlp_ratio", values=[1.0, 2.0, 4.0]),
}

# Phase 3: Ensemble aggregation.
ENSEMBLE = {
    "aggregation":  HParam("aggregation", values=["hard_vote", "soft_vote", "weighted_vote", "stacking"]),
    "meta_learner": HParam("meta_learner", values=["logreg", "svm"]),
    "temperature":  HParam("temperature", low=0.5, high=2.0),
    # V-JEPA-inspired multi-clip eval (Phase A): average softmax across N clips.
    "num_eval_clips": HParam("num_eval_clips", values=[1, 3, 5, 10]),
}


SPACES: Dict[str, Dict[str, HParam]] = {
    "pretrain": PRETRAIN,
    "classify": CLASSIFY,
    "ensemble": ENSEMBLE,
}


def validate_config(phase: str, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check that all values present for the phase are within declared ranges.

    Unknown keys (e.g., shared keys like resize_height) are ignored.
    """
    space = SPACES.get(phase, {})
    errors: List[str] = []
    for key, hp in space.items():
        if key not in config:
            continue
        if not hp.in_range(config[key]):
            errors.append(f"{key}={config[key]!r} out of declared range.")
    # Cross-field coherence checks.
    if phase == "pretrain":
        embed = config.get("vit_embed_dim")
        heads = config.get("vit_num_heads")
        if embed is not None and heads is not None and embed % heads != 0:
            errors.append(f"vit_embed_dim={embed} not divisible by vit_num_heads={heads}.")
    return (not errors, errors)


def estimated_oom(config: Dict[str, Any]) -> bool:
    """Cheap heuristic to short-circuit obviously-OOM configurations on a 6GB card.

    The runner does its own dry-run; this just lets the proposer skip
    proposals that almost certainly will fail.
    """
    embed = config.get("vit_embed_dim", 192)
    depth = config.get("vit_depth", 8)
    batch = config.get("batch_size", 10)
    frames = config.get("frames_per_clip", 8)
    score = embed * depth * batch * frames
    return score > 192 * 10 * 12 * 8 * 1.6  # ~1.6× the largest known-good config
