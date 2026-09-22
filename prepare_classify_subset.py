"""
prepare_classify_subset.py

Builds the labeled CSV for the supervised classification phase, taking the
K700-2020 val set restricted to the same classes used in pre-training.

Pipeline: read the pre-training subset to recover its classes, collect the val
clips for each of them, and write a `video_name,label` CSV. Val (not train) is
used to avoid leakage from the JEPA encoder and because it is class-balanced.

Usage:
    python prepare_classify_subset.py
    python prepare_classify_subset.py \\
        --pretrain_csv /dataset/k700-2020/train/pretrain_subset.csv \\
        --val_dir     /dataset/k700-2020/val \\
        --output_csv  classify_subset.csv
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates labeled CSV for classification from the K700-2020 val set"
    )
    parser.add_argument(
        "--pretrain_csv",
        type=str,
        default="/dataset/k700-2020/train/pretrain_subset.csv",
        help="Pre-training subset CSV (used to extract the 30 classes).",
    )
    parser.add_argument(
        "--val_dir",
        type=str,
        default="/dataset/k700-2020/val",
        help="Root directory of the K700-2020 val set.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="classify_subset.csv",
        help="Output CSV name (saved inside val_dir).",
    )
    return parser.parse_args()


def extract_classes_from_pretrain_csv(pretrain_csv: str) -> list[str]:
    """Reads pretrain_subset.csv and extracts the unique classes (first part of the path)."""
    path = Path(pretrain_csv)
    if not path.exists():
        print(f"ERROR: {pretrain_csv} not found")
        sys.exit(1)

    classes = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().strip('"')
            if not line:
                continue
            # Format: "class/video.mp4"  ->  take the part before /
            parts = line.split("/")
            if len(parts) >= 2:
                classes.add(parts[0])

    return sorted(classes)


def main():
    args = parse_args()
    val_dir = Path(args.val_dir)

    if not val_dir.is_dir():
        print(f"ERROR: val directory not found: {val_dir}")
        sys.exit(1)

    # --- Step 1: Extract the 30 classes from pretrain_subset ---
    print(f"[1/3] Reading classes from: {args.pretrain_csv}")
    classes = extract_classes_from_pretrain_csv(args.pretrain_csv)
    print(f"      Classes found: {len(classes)}")
    for i, cls in enumerate(classes):
        print(f"        {i:2d}. {cls}")

    # --- Step 2: Collect videos from the val set for those classes ---
    print(f"\n[2/3] Collecting videos from: {val_dir}")
    label_map = {cls: idx for idx, cls in enumerate(classes)}
    rows: list[tuple[str, int]] = []
    missing_classes = []

    for cls in classes:
        class_dir = val_dir / cls
        if not class_dir.is_dir():
            print(f"  [!] Class not found in val: '{cls}'")
            missing_classes.append(cls)
            continue

        videos = sorted(class_dir.glob("*.mp4"))
        if not videos:
            print(f"  [!] No .mp4 videos in: {class_dir}")
            missing_classes.append(cls)
            continue

        label = label_map[cls]
        for vp in videos:
            rows.append((f"{cls}/{vp.name}", label))

        print(f"      {cls}: {len(videos)} videos (label={label})")

    if missing_classes:
        print(f"\n  WARNING: {len(missing_classes)} classes not found in val: {missing_classes}")

    # --- Step 3: Write CSV ---
    output_path = val_dir / args.output_csv
    print(f"\n[3/3] Writing CSV: {output_path}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["video_name", "label"])  # header required by VideoDataset
        for video_name, label in rows:
            writer.writerow([video_name, label])

    # --- Summary ---
    labels_used = sorted(set(r[1] for r in rows))
    print(f"\n✓ CSV ready:")
    print(f"  Classes: {len(classes) - len(missing_classes)}")
    print(f"  Videos:  {len(rows)}")
    print(f"  Labels:  {labels_used[0]}-{labels_used[-1]}")
    print(f"  File:    {output_path}")
    print(f"\nUpdate configs/config.yaml:")
    print(f'  video_dir: "{val_dir}"')
    print(f'  csv_file_labeled: "{args.output_csv}"')
    print(f"  num_classes: {len(classes) - len(missing_classes)}")


if __name__ == "__main__":
    main()
