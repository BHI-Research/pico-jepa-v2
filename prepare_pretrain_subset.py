"""
prepare_pretrain_subset.py

Distills a small, concept-diverse and motion-rich subset of K700-2020 for JEPA
pre-training, writing a CSV that VideoDataset can consume directly.

Pipeline: cluster class names (TF-IDF + K-Means) and keep one representative
class per cluster, then score each class's clips by temporal variance (motion
proxy) and keep the most dynamic ones.

Usage:
    python prepare_pretrain_subset.py \\
        --k700_dir /dataset/K700-2020/train \\
        --output_csv pretrain_subset.csv \\
        --num_clusters 30 \\
        --videos_per_class 100 \\
        --diversity_sample 200
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Data curation for JEPA pre-training")
    parser.add_argument(
        "--k700_dir",
        type=str,
        required=True,
        help="Path to the K700-2020 dataset train directory (contains per-class subdirectories)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="pretrain_subset.csv",
        help="Output CSV name (saved inside k700_dir). Default: pretrain_subset.csv",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=30,
        help="Number of semantic clusters (≈ number of selected classes). Default: 30",
    )
    parser.add_argument(
        "--videos_per_class",
        type=int,
        default=100,
        help="Maximum number of videos to include per selected class. Default: 100",
    )
    parser.add_argument(
        "--diversity_sample",
        type=int,
        default=200,
        help="How many videos to sample per class for diversity scoring. Default: 200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Discover available classes in the directory
# ---------------------------------------------------------------------------

def discover_classes(k700_dir: str) -> list[str]:
    """Returns the list of classes (subdirectories with at least 1 .mp4)."""
    base = Path(k700_dir)
    classes = sorted(
        d.name
        for d in base.iterdir()
        if d.is_dir() and any(d.glob("*.mp4"))
    )
    return classes


# ---------------------------------------------------------------------------
# Step 2: Semantic clustering with TF-IDF + K-Means
# ---------------------------------------------------------------------------

def cluster_classes(class_names: list[str], num_clusters: int, seed: int) -> dict[int, list[str]]:
    """
    Groups the class names using word-level TF-IDF and K-Means.
    Returns dict {cluster_id: [class_name, ...]}.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    # Represent each class name as a TF-IDF bag of words
    # (the names are already in English with spaces/underscores separating words)
    normalized = [name.replace("_", " ").lower() for name in class_names]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(normalized)

    effective_clusters = min(num_clusters, len(class_names))
    km = KMeans(n_clusters=effective_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    clusters: dict[int, list[str]] = {}
    for cls_name, cluster_id in zip(class_names, labels):
        clusters.setdefault(int(cluster_id), []).append(cls_name)

    return clusters, km, vectorizer, X


def select_representative_per_cluster(
    clusters: dict[int, list[str]],
    class_names: list[str],
    km,
    X,
) -> list[str]:
    """
    For each cluster, picks the class closest to the centroid.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    selected = []

    for cluster_id, members in clusters.items():
        if not members:
            continue
        centroid = km.cluster_centers_[cluster_id].reshape(1, -1)
        member_indices = [class_to_idx[m] for m in members]
        member_vectors = X[member_indices]
        dists = euclidean_distances(member_vectors, centroid).flatten()
        closest_local_idx = int(np.argmin(dists))
        selected.append(members[closest_local_idx])

    return sorted(selected)


# ---------------------------------------------------------------------------
# Step 3: Visual diversity scoring of videos
# ---------------------------------------------------------------------------

def score_video_diversity(video_path: str) -> float:
    """
    Decodes 3 frames (start, middle, end) and computes the temporal variance
    as a proxy for motion/dynamism. Returns -1.0 on error.
    """
    try:
        import torchcodec.decoders as decoders

        decoder = decoders.VideoDecoder(video_path)
        total_frames = decoder.metadata.num_frames
        if total_frames is None or total_frames < 3:
            return 0.0

        indices = [
            0,
            total_frames // 2,
            total_frames - 1,
        ]
        # Remove duplicates if the video is very short
        indices = sorted(set(indices))

        frames_data = decoder.get_frames_at(indices).data  # (T, C, H, W) uint8
        frames_float = frames_data.float() / 255.0  # normalize to [0, 1]

        # Variance across frames: measures how much the image changes over time
        variance = float(torch.var(frames_float, dim=0).mean())
        return variance

    except Exception:
        return -1.0


def select_diverse_videos(
    class_name: str,
    k700_dir: str,
    videos_per_class: int,
    diversity_sample: int,
    seed: int,
) -> list[str]:
    """
    Lists the class's .mp4 files, scores up to `diversity_sample` of them by
    temporal variance and returns the best `videos_per_class` as relative paths
    (class/video.mp4).
    """
    class_dir = Path(k700_dir) / class_name
    all_videos = sorted(class_dir.glob("*.mp4"))

    if not all_videos:
        return []

    # Sample for scoring (efficiency)
    rng = random.Random(seed)
    candidates = all_videos if len(all_videos) <= diversity_sample else rng.sample(all_videos, diversity_sample)

    # Score
    scored = []
    for vp in candidates:
        score = score_video_diversity(str(vp))
        if score >= 0.0:
            scored.append((score, vp))

    if not scored:
        # Fallback: take the first videos without scoring
        fallback = all_videos[:videos_per_class]
        return [f"{class_name}/{v.name}" for v in fallback]

    # Sort descending by variance and take top-K
    scored.sort(key=lambda x: x[0], reverse=True)
    top_videos = scored[:videos_per_class]

    return [f"{class_name}/{vp.name}" for _, vp in top_videos]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    k700_dir = args.k700_dir
    if not os.path.isdir(k700_dir):
        print(f"ERROR: directory not found: {k700_dir}")
        sys.exit(1)

    # --- Step 1: Discover classes ---
    print(f"\n[1/4] Scanning classes in: {k700_dir}")
    class_names = discover_classes(k700_dir)
    print(f"      Classes found: {len(class_names)}")

    if len(class_names) == 0:
        print("ERROR: No subdirectories with .mp4 videos found.")
        sys.exit(1)

    # --- Step 2: Semantic clustering ---
    num_clusters = min(args.num_clusters, len(class_names))
    print(f"\n[2/4] Grouping {len(class_names)} classes into {num_clusters} semantic clusters (TF-IDF + K-Means)...")
    clusters, km, vectorizer, X = cluster_classes(class_names, num_clusters, args.seed)
    selected_classes = select_representative_per_cluster(clusters, class_names, km, X)
    print(f"      Selected classes ({len(selected_classes)}):")
    for i, cls in enumerate(selected_classes, 1):
        print(f"        {i:2d}. {cls}")

    # --- Step 3: Visual diversity scoring ---
    print(f"\n[3/4] Selecting the {args.videos_per_class} most dynamic videos per class")
    print(f"      (sampling up to {args.diversity_sample} videos per class for scoring)...")

    all_video_paths: list[str] = []
    for i, cls in enumerate(selected_classes, 1):
        print(f"      [{i:2d}/{len(selected_classes)}] {cls}...", end=" ", flush=True)
        videos = select_diverse_videos(
            cls,
            k700_dir,
            args.videos_per_class,
            args.diversity_sample,
            args.seed,
        )
        all_video_paths.extend(videos)
        print(f"{len(videos)} videos")

    # --- Step 4: Write CSV ---
    # csv.writer with QUOTE_ALL is used so that paths with spaces or
    # parentheses (e.g. "acting in play/abc.mp4", "backflip (human)/xyz.mp4")
    # are always quoted and pandas reads them unambiguously.
    output_path = os.path.join(k700_dir, args.output_csv)
    print(f"\n[4/4] Writing CSV: {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for vp in all_video_paths:
            writer.writerow([vp])

    print(f"\n✓ Subset ready:")
    print(f"  Classes: {len(selected_classes)}")
    print(f"  Videos: {len(all_video_paths)}")
    print(f"  CSV:    {output_path}")
    print(f"\nUpdate configs/config.yaml:")
    print(f'  csv_file: "{args.output_csv}"')


if __name__ == "__main__":
    main()
