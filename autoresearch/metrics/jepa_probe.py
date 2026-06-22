"""Fast linear probe for evaluating a pre-trained JEPA encoder.

The JEPA training loss is not directly indicative of downstream
classification performance. After each pretrain experiment, we freeze the
encoder, train a linear head on a small labeled subset for one epoch, and
report top-1 accuracy. ~3 minutes on a GTX 1060.

This metric drives the ratchet for phase 1.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset.datasets import VideoDataset  # noqa: E402
from models.video_classifier import VideoClassifier  # noqa: E402


def linear_probe(
    encoder_path: str,
    config: Dict[str, Any],
    classify_csv_path: str,
    classify_video_dir: str,
    num_classes: int,
    num_videos: int = 500,
    epochs: int = 1,
    batch_size: int = 16,
    seed: int = 1337,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train a linear head on top of the frozen encoder; return top-1 acc.

    Uses a fixed deterministic subset of size ``num_videos`` so that probes
    across different encoders are comparable.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = VideoDataset(
        video_dir=classify_video_dir,
        csv_path=classify_csv_path,
        frames_per_clip=config["frames_per_clip"],
        target_height=config["resize_height"],
        target_width=config["resize_width"],
        channels=config["video_channels"],
        labeled=True,
        sampling_strategy="center",
    )
    n = len(full_dataset)
    if n == 0:
        return {"probe_top1": 0.0, "probe_wallclock_s": 0.0, "n_train": 0, "n_val": 0}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = min(num_videos, n)
    subset_idx = perm[:k].tolist()
    split = max(1, int(0.8 * k))
    train_idx = subset_idx[:split]
    val_idx = subset_idx[split:]

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
    ) if val_idx else None

    model = VideoClassifier(
        encoder_config=config,
        num_classes=num_classes,
        freeze_encoder=True,
        pretrained_encoder_path=encoder_path,
    ).to(device)

    optimizer = torch.optim.AdamW(model.classifier_head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            if batch is None:
                continue
            videos, labels = batch
            if videos is None or videos.nelement() == 0:
                continue
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(videos)
            loss = criterion(logits, labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    if val_loader is not None:
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                videos, labels = batch
                if videos is None or videos.nelement() == 0:
                    continue
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(videos)
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == labels).sum().item())
                total += int(labels.size(0))

    top1 = (correct / total) if total > 0 else 0.0
    return {
        "probe_top1": float(top1),
        "probe_wallclock_s": float(time.perf_counter() - started),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
