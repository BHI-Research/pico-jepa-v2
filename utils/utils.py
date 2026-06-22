import os

import torch
import torchcodec


def print_system_info(force_cpu=True):
    """Prints information about the device and torchcodec version."""

    if force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage. This is useful for debugging or environments without GPU support.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

    torchcodec_version = "N/A"
    if hasattr(torchcodec, "__version__") and torchcodec.__version__:
        torchcodec_version = torchcodec.__version__
    elif hasattr(torchcodec, "version") and hasattr(torchcodec.version, "git_version"):
        torchcodec_version = torchcodec.version.git_version
    print(f"Using torchcodec version: {torchcodec_version}")
    return device


def create_dummy_dataset_if_needed(video_dir, csv_file, config, labeled=False):
    """
    Checks for the existence of a dataset directory and CSV file.
    If they are missing or the CSV is empty, it creates a dummy dataset
    and a corresponding CSV file for demonstration purposes.

    Args:
        video_dir (str): The path to the main video directory.
        csv_file (str): The filename of the CSV file.
        config (dict): The configuration dictionary. Must contain 'batch_size'.
                       If labeled is True, must also contain 'num_classes'.
        labeled (bool): If True, creates a CSV with 'video_name,label' columns.
                        If False, creates a CSV with just video names.
    """
    csv_file_path = os.path.join(video_dir, csv_file)

    # Check if the directory or CSV needs to be created
    needs_creation = False
    if not os.path.exists(video_dir):
        print(f"Dataset directory '{video_dir}' not found. Creating dummy dataset.")
        os.makedirs(video_dir, exist_ok=True)
        needs_creation = True
    elif not os.path.exists(csv_file_path) or os.path.getsize(csv_file_path) == 0:
        print(f"CSV file '{csv_file_path}' is missing or empty. Creating dummy CSV.")
        needs_creation = True

    if not needs_creation:
        return

    batch_size = config.get("batch_size")
    if batch_size is None:
        raise ValueError("Configuration must contain 'batch_size'.")

    num_dummy_videos = batch_size * 2
    dummy_video_names = [
        f"dummy_class/dummy_video_{i}.mp4" for i in range(num_dummy_videos)
    ]

    # Create a subdirectory for the dummy videos
    dummy_class_dir = os.path.join(video_dir, "dummy_class")
    if not os.path.exists(dummy_class_dir):
        os.makedirs(dummy_class_dir, exist_ok=True)

    # Create the dummy CSV file
    print(f"Creating dummy '{csv_file}' with {num_dummy_videos} entries.")
    with open(csv_file_path, "w") as f:
        if labeled:
            num_classes = config.get("num_classes")
            if num_classes is None:
                raise ValueError(
                    "Configuration must contain 'num_classes' for a labeled dummy dataset."
                )
            f.write("video_name,label\n")  # Header for classification CSV
            for i, name in enumerate(dummy_video_names):
                f.write(f"{name},{i % num_classes}\n")  # Assign dummy labels
        else:
            for name in dummy_video_names:
                f.write(name + "\n")  # Pre-training CSV just has video names

    print(f"Dummy CSV created at '{csv_file_path}'.")
    print(
        "Note: This is a placeholder. For real training, ensure actual .mp4 files exist and the CSV lists them correctly."
    )


def generate_spatiotemporal_masks(
    num_patches_t,
    num_patches_h,
    num_patches_w,
    mask_ratio,
    device,
    batch_size=1,
):
    """
    Generates spatio-temporal "tubelet" masks (single-block, uniform random).
    The same spatial patches are masked across all temporal steps.
    If batch_size > 1, returns a batched mask.
    """
    num_spatial_patches = num_patches_h * num_patches_w
    num_masked_spatial = int(mask_ratio * num_spatial_patches)

    # Generate spatial masks for each item in the batch
    spatial_mask = torch.zeros(
        batch_size, num_spatial_patches, device=device, dtype=torch.bool
    )
    for i in range(batch_size):
        shuffled_indices = torch.randperm(num_spatial_patches, device=device)
        spatial_mask[i, shuffled_indices[:num_masked_spatial]] = True

    # Expand the spatial mask to create spatio-temporal "tubelets"
    spatiotemporal_mask = spatial_mask.repeat_interleave(num_patches_t, dim=1)

    return spatiotemporal_mask.squeeze(0) if batch_size == 1 else spatiotemporal_mask


def generate_multiblock_masks(
    num_patches_t,
    num_patches_h,
    num_patches_w,
    mask_ratio,
    num_blocks,
    device,
    batch_size=1,
):
    """V-JEPA-style multi-block tubelet masking.

    Each sample gets ``num_blocks`` rectangular blocks placed randomly in the
    (H, W) grid. Blocks may overlap; the union is then capped (or padded with
    random patches) to exactly ``K = int(mask_ratio * H * W)`` patches per
    sample so that ``mask.sum(-1)`` is **uniform across the batch**. This is
    critical: with variable cardinality the predictor's gather/reshape paths
    in pico_jepa.py would crash.

    The same spatial mask is repeated across all ``num_patches_t`` temporal
    steps (tubelet), matching the V-JEPA "spatiotemporal" assumption that the
    target stays the same shape over time.

    For ``num_blocks <= 1`` this falls back to the single-block uniform
    sampling already implemented in ``generate_spatiotemporal_masks``.
    """
    if num_blocks <= 1:
        return generate_spatiotemporal_masks(
            num_patches_t=num_patches_t,
            num_patches_h=num_patches_h,
            num_patches_w=num_patches_w,
            mask_ratio=mask_ratio,
            device=device,
            batch_size=batch_size,
        )

    H, W = num_patches_h, num_patches_w
    HW = H * W
    K = int(mask_ratio * HW)
    K = max(num_blocks, min(K, HW))  # leave at least 1 patch per block

    # Per-block area target. Use a roughly-square aspect ratio +/- 50% jitter.
    base_area = max(1, K // num_blocks)

    spatial_mask = torch.zeros(batch_size, HW, dtype=torch.bool, device=device)

    for b in range(batch_size):
        for _ in range(num_blocks):
            # Sample block dimensions around sqrt(area), with jitter.
            side = max(1, int(round(base_area ** 0.5)))
            jitter_h = int(torch.randint(-side // 2, side // 2 + 1, (1,)).item())
            block_h = max(1, min(H, side + jitter_h))
            block_w = max(1, min(W, max(1, base_area // block_h)))

            top = int(torch.randint(0, max(1, H - block_h + 1), (1,)).item())
            left = int(torch.randint(0, max(1, W - block_w + 1), (1,)).item())

            for r in range(top, top + block_h):
                row_start = r * W
                spatial_mask[b, row_start + left : row_start + left + block_w] = True

        # Cap or pad to exactly K masked patches so the batch has uniform cardinality.
        current = int(spatial_mask[b].sum().item())
        if current < K:
            unmasked = (~spatial_mask[b]).nonzero(as_tuple=True)[0]
            n_add = K - current
            if n_add > 0 and len(unmasked) > 0:
                perm = torch.randperm(len(unmasked), device=device)[:n_add]
                spatial_mask[b, unmasked[perm]] = True
        elif current > K:
            masked = spatial_mask[b].nonzero(as_tuple=True)[0]
            n_rm = current - K
            perm = torch.randperm(len(masked), device=device)[:n_rm]
            spatial_mask[b, masked[perm]] = False

    spatiotemporal_mask = spatial_mask.repeat_interleave(num_patches_t, dim=1)
    return spatiotemporal_mask.squeeze(0) if batch_size == 1 else spatiotemporal_mask