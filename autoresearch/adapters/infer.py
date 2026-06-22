"""Batch inference adapter.

Replaces ``infers_part.sh`` / ``infers_gral.sh`` (which spawn one Python
process per video × per model). This module loads each model once and
processes all videos in a tight loop, which is 10-50× faster.

Returns a 3D probability tensor of shape (num_models, num_videos, num_classes)
plus the parallel arrays of video paths and ground-truth labels (when
available). Downstream ensemble methods consume this directly.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.datasets import load_and_preprocess_single_video, load_multi_clip_uniform  # noqa: E402
from models.video_classifier import VideoClassifier  # noqa: E402


def _load_classifier(model_path: str, config: Dict[str, Any], num_classes: int, device: torch.device) -> VideoClassifier:
    model = VideoClassifier(
        encoder_config=config,
        num_classes=num_classes,
        pretrained_encoder_path=None,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def infer_with_models(
    model_paths: Sequence[str],
    configs: Sequence[Dict[str, Any]],
    num_classes: int,
    video_paths: Sequence[str],
    labels: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
    num_clips: int = 1,
) -> Dict[str, Any]:
    """Run inference for every (model, video) pair.

    Args:
        model_paths: list of N classifier .pth paths.
        configs: list of N config dicts (one per model — they may differ in
                 vit_embed_dim/depth/etc.).
        num_classes: number of classes (must match models' heads).
        video_paths: list of M absolute video paths.
        labels: optional ground truth labels (length M) for accuracy reporting.
        num_clips: V-JEPA-style multi-clip evaluation. ``1`` keeps the cheap
                   center-clip behavior; ``5-10`` averages softmax across
                   uniformly-spaced clips (≈+2-4pp top-1 at 10× cost).

    Returns:
        {
            "probs": np.ndarray (N, M, C) softmax probabilities,
            "preds": np.ndarray (N, M) argmax,
            "video_paths": list[str],
            "labels": np.ndarray (M,) or None,
            "per_model_acc": list[float] or None,
            "num_clips": int,
        }
    """
    if len(model_paths) != len(configs):
        raise ValueError("model_paths and configs must have same length.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_models = len(model_paths)
    n_videos = len(video_paths)
    probs = np.zeros((n_models, n_videos, num_classes), dtype=np.float32)
    preds = np.zeros((n_models, n_videos), dtype=np.int64)

    # Pre-decode videos once on CPU — they don't depend on the model. We
    # stream each tensor to GPU only inside the inner loop and free it
    # immediately afterwards, so peak GPU memory stays at 1 video × num_clips
    # regardless of dataset size. Keeping these on GPU instead would OOM on
    # any card with multi-clip eval over a real val/holdout set
    # (e.g. 600 videos × 10 clips × ~4.6MB = ~27GB).
    decoded_cpu: List[Optional[torch.Tensor]] = []
    for vp in video_paths:
        if num_clips <= 1:
            t = load_and_preprocess_single_video(vp, configs[0])
            decoded_cpu.append(t.unsqueeze(0) if t is not None else None)
        else:
            t = load_multi_clip_uniform(vp, configs[0], num_clips=num_clips)
            decoded_cpu.append(t if t is not None else None)

    for mi, (mp, cfg) in enumerate(zip(model_paths, configs)):
        model = _load_classifier(mp, cfg, num_classes, device)
        with torch.no_grad():
            for vi, cpu_tensor in enumerate(decoded_cpu):
                if cpu_tensor is None:
                    probs[mi, vi] = 1.0 / num_classes
                    preds[mi, vi] = -1
                    continue
                tensor = cpu_tensor.to(device, non_blocking=True)
                logits = model(tensor)  # (num_clips_or_1, num_classes)
                p = torch.softmax(logits, dim=1).mean(dim=0).detach().cpu().numpy()
                probs[mi, vi] = p
                preds[mi, vi] = int(np.argmax(p))
                del tensor, logits
        # Free GPU memory before loading the next model.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_model_acc: Optional[List[float]] = None
    labels_arr: Optional[np.ndarray] = None
    if labels is not None:
        labels_arr = np.asarray(labels, dtype=np.int64)
        per_model_acc = [
            float((preds[mi] == labels_arr).mean()) for mi in range(n_models)
        ]

    return {
        "probs": probs,
        "preds": preds,
        "video_paths": list(video_paths),
        "labels": labels_arr,
        "per_model_acc": per_model_acc,
        "num_clips": int(num_clips),
    }


def videos_from_csv(
    csv_path: str, video_dir: str, indices: Optional[Sequence[int]] = None
) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(csv_path)
    if indices is not None:
        df = df.iloc[list(indices)].reset_index(drop=True)
    if "video_name" not in df.columns:
        raise ValueError(f"CSV {csv_path} must have 'video_name' column.")
    has_labels = "label" in df.columns
    paths = [os.path.join(video_dir, str(n)) for n in df["video_name"].tolist()]
    labels = (
        [int(x) for x in df["label"].tolist()] if has_labels else [-1] * len(paths)
    )
    return paths, labels
