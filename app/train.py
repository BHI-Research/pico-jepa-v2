import argparse
import hashlib
import json
import os
import sys
import time
import torch
from collections import Counter
from datetime import datetime
import yaml
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler



project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.datasets import VideoDataset
from models.pico_jepa import PicoJEPA_Pretrain
from utils.utils import create_dummy_dataset_if_needed, print_system_info


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Parse Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path",
    type=str,
    default=os.path.join(project_root, "configs", "config.yaml"),
    help="Path to the YAML configuration file.",
)
parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="If set, write final metrics (loss, wallclock_s, config_hash) to this JSON path.",
)
args = parser.parse_args()
CONFIG = load_config(args.config_path)

# Phase-aware path resolution: prefer pretrain_* keys, fallback to video_dir/csv_file.
_PRETRAIN_VIDEO_DIR = CONFIG.get("pretrain_video_dir", CONFIG.get("video_dir"))
_PRETRAIN_CSV_PATH = CONFIG.get("pretrain_csv_path")
_PRETRAIN_CSV_FILE = CONFIG.get("csv_file")


def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, "a") as f:
        f.write(full_message + "\n")


def save_model(pico_jepa_model, encoder_save_path):
    print(f"Saving pre-trained online_encoder weights to {encoder_save_path}...")
    torch.save(pico_jepa_model.online_encoder.state_dict(), encoder_save_path)
    print(f"Pre-trained encoder saved successfully to {encoder_save_path}")


def build_pretrain_param_groups(
    model,
    base_lr: float,
    predictor_lr_multiplier: float,
    weight_decay: float,
    layerwise_lr_decay: float = 1.0,
):
    """Build AdamW param_groups with optional V-JEPA-style layer-wise LR decay.

    With ``layerwise_lr_decay == 1.0`` this is numerically identical to the
    legacy 4-group setup (encoder weights, predictor weights, encoder bias/1D,
    predictor bias/1D). With ``decay < 1.0`` each encoder layer (patch_embed,
    each transformer block, final norm) gets its own group with
    ``lr = base_lr * decay ** (n_layers - 1 - i)`` so early layers train slower
    than late ones. Predictor stays on a single multiplier — its small depth
    doesn't justify its own decay.

    Bias and 1D parameters always get ``weight_decay=0`` regardless of layer.
    """
    encoder = model.online_encoder
    predictor = model.predictor

    encoder_layers = [encoder.patch_embed] + list(encoder.blocks) + [encoder.norm]
    n_layers = len(encoder_layers)

    groups = []
    for i, layer in enumerate(encoder_layers):
        scale = layerwise_lr_decay ** (n_layers - 1 - i) if layerwise_lr_decay < 1.0 else 1.0
        lr = base_lr * scale
        for n_p, p in layer.named_parameters():
            if not p.requires_grad:
                continue
            wd = 0.0 if (p.dim() == 1 or n_p.endswith("bias")) else weight_decay
            groups.append({"params": [p], "lr": lr, "initial_lr": lr, "weight_decay": wd})

    # Encoder positional embedding: treat as the earliest layer (lowest LR).
    if hasattr(encoder, "pos_embed") and encoder.pos_embed.requires_grad:
        scale = layerwise_lr_decay ** (n_layers - 1) if layerwise_lr_decay < 1.0 else 1.0
        lr = base_lr * scale
        groups.append({"params": [encoder.pos_embed], "lr": lr, "initial_lr": lr, "weight_decay": 0.0})

    # Predictor: single LR multiplier (no per-layer decay inside predictor).
    pred_lr = base_lr * predictor_lr_multiplier
    for n_p, p in predictor.named_parameters():
        if not p.requires_grad:
            continue
        wd = 0.0 if (p.dim() == 1 or n_p.endswith("bias")) else weight_decay
        groups.append({"params": [p], "lr": pred_lr, "initial_lr": pred_lr, "weight_decay": wd})

    if hasattr(model, "mask_token") and model.mask_token.requires_grad:
        groups.append({
            "params": [model.mask_token], "lr": pred_lr, "initial_lr": pred_lr, "weight_decay": 0.0,
        })

    return groups


def do_pretraining():
    log_file_name = "training_" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".log"
    device = print_system_info(force_cpu=CONFIG["force_cpu"])
    print("Initializing Video Dataset for Pre-training (with torchcodec 0.4.0 API)...")
    video_dataset = VideoDataset(
        video_dir=_PRETRAIN_VIDEO_DIR,
        csv_file=_PRETRAIN_CSV_FILE if _PRETRAIN_CSV_PATH is None else None,
        csv_path=_PRETRAIN_CSV_PATH,
        frames_per_clip=CONFIG["frames_per_clip"],
        target_height=CONFIG["resize_height"],
        target_width=CONFIG["resize_width"],
        channels=CONFIG["video_channels"],
        labeled=False,
        sampling_strategy="random",
    )

    if len(video_dataset) == 0:
        resolved_csv = _PRETRAIN_CSV_PATH or os.path.join(_PRETRAIN_VIDEO_DIR or ".", _PRETRAIN_CSV_FILE or "")
        print(
            f"Error: Dataset is empty. video_dir='{_PRETRAIN_VIDEO_DIR}', csv='{resolved_csv}'."
        )
        if _PRETRAIN_VIDEO_DIR is None or not os.path.exists(_PRETRAIN_VIDEO_DIR) or not os.path.exists(resolved_csv):
            print(
                "Hint: The dataset directory or CSV file might be missing."
            )
        exit()

    data_loader = DataLoader(
        video_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )
    print(
        f"Dataset initialized with {len(video_dataset)} videos. DataLoader ready with {len(data_loader)} batches."
    )

    encoder_save_path = CONFIG["encoder_save_path"]

    pico_jepa_model = PicoJEPA_Pretrain(CONFIG).to(device)
    params_count = sum(
        p.numel() for p in pico_jepa_model.parameters() if p.requires_grad
    )
    print(f"PicoJEPA_Pretrain model created. Trainable parameters: {params_count:,}")

    # --- Learning Rate Optimization: Parameter Groups and Scheduler ---
    base_lr = CONFIG["learning_rate"]
    predictor_lr_multiplier = CONFIG.get("predictor_lr_multiplier", 1.0)
    layerwise_lr_decay = float(CONFIG.get("layerwise_lr_decay", 1.0))
    param_groups = build_pretrain_param_groups(
        model=pico_jepa_model,
        base_lr=base_lr,
        predictor_lr_multiplier=predictor_lr_multiplier,
        weight_decay=CONFIG["weight_decay"],
        layerwise_lr_decay=layerwise_lr_decay,
    )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,

    )

    total_steps = len(data_loader) * CONFIG["num_epochs"]
    warmup_epochs = CONFIG.get("warmup_epochs", int(0.1 * CONFIG["num_epochs"]))
    warmup_steps = warmup_epochs * len(data_loader)
    # Inform the model of the schedule horizon so its EMA cosine decay aligns.
    pico_jepa_model.set_step(0, total_steps=total_steps)

    # final_lr, if it does not exist by default it will be 0.
    final_lr = CONFIG.get("final_lr", 0.0)
    # T_max is the number of steps for the scheduler after the warmup.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=final_lr,
    )
    print(
        f"\n--- Starting PicoJEPA_ViT Self-Supervised Pre-training for {CONFIG['num_epochs']} epochs ---"
    )
    log_message(
        f"Dataset initialized with {len(video_dataset)} videos. DataLoader ready with {len(data_loader)} batches.",
        log_file_name,
    )
    log_message(
        f"Optimizer: AdamW | Base LR: {base_lr} | Predictor LR Multiplier: {predictor_lr_multiplier} | Weight Decay: {CONFIG['weight_decay']}",
        log_file_name,
    )
    log_message(
        f"Scheduler: CosineAnnealingLR with {warmup_epochs} warmup epochs. Final LR: {final_lr}",
        log_file_name,
    )
    if CONFIG.get("clip_grad_norm") is not None:
        log_message(
            f"Gradient Clipping: Enabled with norm {CONFIG['clip_grad_norm']} (after warmup epochs).",
            log_file_name,
        )
    else:
        log_message(
            "Gradient Clipping: Disabled.",
            log_file_name,
        )

    pico_jepa_model.train()
    wallclock_start = time.perf_counter()
    last_avg_loss = float("nan")
    # Best-checkpoint tracking: the JEPA loss often bottoms out early and then
    # rises (overfitting tail). Saving only the final epoch can persist a worse
    # encoder than an intermediate one, so we keep the lowest-avg-loss checkpoint.
    best_avg_loss = float("inf")
    best_epoch = -1

    for epoch in range(CONFIG["num_epochs"]):
        total_epoch_loss = 0
        batches_processed = 0
        for i, videos_batch in enumerate(data_loader):
            if videos_batch is None or (
                isinstance(videos_batch, torch.Tensor) and videos_batch.nelement() == 0
            ):
                print(f"Warning: Received an empty or None batch {i + 1}. Skipping.")
                continue
            current_step = epoch * len(data_loader) + i
            # Drive the EMA cosine schedule alongside the LR schedule.
            pico_jepa_model.set_step(current_step)
            # ---  Learning Rate Management (Warmup and Scheduler)---
            if current_step < warmup_steps:
                # Linear Warmup Phase: Increase LR from 0 to base_lr
                warmup_factor = current_step / warmup_steps
                for param_group in optimizer.param_groups:
                    # Apply the warmup factor to the base LR of each group
                    # Make sure param_group['lr'] is set to base
                    param_group["lr"] = param_group["initial_lr"] * warmup_factor
            else:
                # After the warmup, the scheduler takes over
                scheduler.step()

            if (i + 1) % 50 == 0:  #
                lr_counts = Counter(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
                summary = ", ".join(f"{lr}×{n}" for lr, n in sorted(lr_counts.items()))
                print(f"Current LRs (lr×count): {summary}")

            videos_batch = videos_batch.to(device, non_blocking=True)
            expected_shape = (
                CONFIG["batch_size"],
                CONFIG["video_channels"],
                CONFIG["frames_per_clip"],
                CONFIG["resize_height"],
                CONFIG["resize_width"],
            )
            if videos_batch.shape != expected_shape:
                print(
                    f"Warning: Batch {i + 1} has unexpected shape {videos_batch.shape}. Expected {expected_shape}. Skipping."
                )
                continue

            optimizer.zero_grad()
            loss = pico_jepa_model(videos_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: NaN or Inf loss encountered at Epoch {epoch + 1}, Batch {i + 1}. Skipping update."
                )
                continue

            loss.backward()
            clip_grad_norm_value = CONFIG.get("clip_grad_norm")
            if (
                clip_grad_norm_value is not None and epoch >= warmup_epochs
            ):  # Apply clipping after the warmup
                # Applies to all model parameters to simplify
                # If mixed precision (scaler) were used, scaler.unscale_(optimizer) would go here before the clip
                torch.nn.utils.clip_grad_norm_(
                    pico_jepa_model.parameters(), clip_grad_norm_value
                )


            optimizer.step()

            total_epoch_loss += loss.item()
            batches_processed += 1

            if (i + 1) % 5 == 0 or i == len(data_loader) - 1:
                msg = f"Epoch {epoch + 1}/{CONFIG['num_epochs']} | Batch {i + 1}/{len(data_loader)} | Loss: {loss.item():.4f}"
                print(msg)
                log_message(msg, log_file_name)

        if batches_processed > 0:
            avg_epoch_loss = total_epoch_loss / batches_processed
            last_avg_loss = avg_epoch_loss
            print(
                f"--- Epoch {epoch + 1} Summary --- Avg. Loss: {avg_epoch_loss:.4f} ---"
            )
            log_message(
                f"Epoch {epoch + 1} Summary: Avg. Loss: {avg_epoch_loss:.4f}",
                log_file_name,
            )
        else:
            print(
                f"--- Epoch {epoch + 1} Summary --- No batches were processed. Check data loading and dataset contents. ---"
            )
            log_message(
                f"Epoch {epoch + 1} Summary: No batches were processed. Check data loading and dataset contents.",
                log_file_name,
            )

        # Save only when this epoch improves on the best avg loss so far.
        # The encoder_save_path always holds the best-of-run checkpoint.
        if batches_processed > 0 and avg_epoch_loss < best_avg_loss:
            best_avg_loss = avg_epoch_loss
            best_epoch = epoch + 1
            save_model(pico_jepa_model, encoder_save_path)
            log_message(
                f"Epoch {epoch + 1}: new best avg loss {best_avg_loss:.4f} -> saved encoder.",
                log_file_name,
            )

    print("\n--- PicoJEPA_ViT Self-Supervised Pre-training finished. ---")
    msg = (
        f"Best checkpoint: epoch {best_epoch} with avg loss {best_avg_loss:.4f} "
        f"(kept at {encoder_save_path}). Final epoch avg loss was {last_avg_loss:.4f}."
    )
    print(msg)
    log_message(msg, log_file_name)

    print("\nImportant Notes:")
    print(
        "1. This script performs self-supervised pre-training using the PicoJEPA method."
    )
    print(
        "2. Ensure 'torchcodec' (v0.4.0 in this case) and its dependencies (like a compatible FFmpeg) are correctly installed."
    )
    print(
        "3. For actual training, ensure your dataset path and CSV (e.g., nano-train.csv with video names) are correct."
    )
    print(
        "4. Monitor memory usage. If OOM errors occur, reduce batch_size, video dimensions, or ViT model size in CONFIG."
    )
    print(
        f"5. The pre-trained encoder is saved to '{encoder_save_path}' and can be used for downstream tasks like classification."
    )

    wallclock_s = time.perf_counter() - wallclock_start
    config_hash = hashlib.sha256(
        json.dumps(CONFIG, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    if args.output_json:
        metrics = {
            "phase": "pretrain",
            "final_loss": float(last_avg_loss) if last_avg_loss == last_avg_loss else None,
            "best_loss": float(best_avg_loss) if best_avg_loss != float("inf") else None,
            "best_epoch": best_epoch,
            "wallclock_s": wallclock_s,
            "config_hash": config_hash,
            "encoder_save_path": encoder_save_path,
            "num_videos": len(video_dataset),
            "num_epochs": CONFIG["num_epochs"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to {args.output_json}")


if __name__ == "__main__":
    do_pretraining()
