import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from datetime import datetime
import torch.optim.lr_scheduler


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataset.datasets import VideoDataset
from models.video_classifier import VideoClassifier
from utils.utils import create_dummy_dataset_if_needed, print_system_info


def load_config_from_yaml(config_path):

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The configuration file was not found at: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def log_message(message, log_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    with open(log_file, "a") as f:
        f.write(full_message + "\n")


# --- Main Classification Training Script ---
def train_video_classifier(args):
    try:
        CONFIG_CLASSIFY = load_config_from_yaml(args.config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Determinism: when the autoresearch classify adapter trains N submodels
    # with the same data but distinct seeds, each one converges to a slightly
    # different classifier — that diversity is what an ensemble exploits.
    seed = int(getattr(args, "seed", None) or CONFIG_CLASSIFY.get("seed", 1337))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[classify_videos] seed = {seed}")

    device = print_system_info(force_cpu=CONFIG_CLASSIFY["force_cpu"])
    log_file_name = (
        "classification" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + ".log"
    )
    classifier_save_path = CONFIG_CLASSIFY.get(
        "classifier_save_path", "./pico_jepa_classifier.pth"
    )

    # Phase-aware path resolution: prefer classify_* keys, fallback to video_dir/csv_file_labeled.
    classify_video_dir = CONFIG_CLASSIFY.get("classify_video_dir", CONFIG_CLASSIFY.get("video_dir"))
    classify_csv_path = CONFIG_CLASSIFY.get("classify_csv_path")
    classify_csv_file = CONFIG_CLASSIFY.get("csv_file_labeled")

    print("Initializing Labeled Video Dataset for Classification...")
    classify_dataset = VideoDataset(
        video_dir=classify_video_dir,
        csv_file=classify_csv_file if classify_csv_path is None else None,
        csv_path=classify_csv_path,
        frames_per_clip=CONFIG_CLASSIFY["frames_per_clip"],
        target_height=CONFIG_CLASSIFY["resize_height"],
        target_width=CONFIG_CLASSIFY["resize_width"],
        channels=CONFIG_CLASSIFY["video_channels"],
        labeled=True,
        sampling_strategy="random",
    )

    if len(classify_dataset) == 0:
        resolved_csv = classify_csv_path or os.path.join(classify_video_dir or ".", classify_csv_file or "")
        print(
            f"Error: Classification dataset is empty. video_dir='{classify_video_dir}', csv='{resolved_csv}'."
        )
        exit()

    # Simple 80/20 train/validation split
    train_size = int(0.8 * len(classify_dataset))
    val_size = len(classify_dataset) - train_size

    if train_size == 0 or val_size == 0:
        print(
            f"Dataset too small for train/val split (Total: {len(classify_dataset)}). Needs at least 2 samples for split."
        )
        print("Proceeding with the full dataset for training, no validation.")
        train_dataset = classify_dataset
        val_dataset = None  # No validation set
    else:
        train_dataset, val_dataset = torch.utils.data.random_split(
            classify_dataset, [train_size, val_size]
        )

    print(f"Training set size: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG_CLASSIFY["batch_size"],
        shuffle=True,
        num_workers=CONFIG_CLASSIFY["num_workers"],
        pin_memory=True if device.type == "cuda" else False,
        drop_last=(
            True if len(train_dataset) > CONFIG_CLASSIFY["batch_size"] else False
        ),  # Avoid error if dataset smaller than batch
    )
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG_CLASSIFY["batch_size"],
            shuffle=False,
            num_workers=CONFIG_CLASSIFY["num_workers"],
            pin_memory=True if device.type == "cuda" else False,
            drop_last=False,
        )
    else:
        val_loader = None

    print("DataLoader for classification ready.")

    classification_model = VideoClassifier(
        encoder_config=CONFIG_CLASSIFY,  # Pass the classification config
        num_classes=CONFIG_CLASSIFY["num_classes"],
        freeze_encoder=CONFIG_CLASSIFY[
            "freeze_encoder"
        ],  # This will be handled by the class
        pretrained_encoder_path=CONFIG_CLASSIFY["encoder_save_path"],
    ).to(device)

    # --- Optimizer Configuration and Parameter Groups ---
    params_to_optimize = []
    if not CONFIG_CLASSIFY["freeze_encoder"]:
        print("Setting up optimizer for fine-tuning encoder and classification head.")

        params_to_optimize.append(
            {
                "params": (
                    p
                    for n, p in classification_model.encoder.named_parameters()
                    if ("bias" not in n) and (len(p.shape) != 1)
                ),
                "lr": CONFIG_CLASSIFY["learning_rate_encoder_finetune"],
                "weight_decay": CONFIG_CLASSIFY["classify_weight_decay"],
            }
        )
        params_to_optimize.append(
            {
                "params": (
                    p
                    for n, p in classification_model.encoder.named_parameters()
                    if ("bias" in n) or (len(p.shape) == 1)
                ),
                "lr": CONFIG_CLASSIFY["learning_rate_encoder_finetune"],
                "weight_decay": 0,
            }
        )
    else:
        print("Encoder is frozen. Setting up optimizer for classification head only.")
        pass

    # Head includes both the pool module (which may have learnable params for
    # head_type in {mlp_2layer, attentive}) and the final classifier_head Linear.
    head_modules = [classification_model.pool, classification_model.classifier_head]
    head_params = [(n, p) for m in head_modules for n, p in m.named_parameters() if p.requires_grad]
    params_to_optimize.append(
        {
            "params": [p for n, p in head_params if ("bias" not in n) and (len(p.shape) != 1)],
            "lr": CONFIG_CLASSIFY["learning_rate_classifier"],
            "weight_decay": CONFIG_CLASSIFY["classify_weight_decay"],
        }
    )
    params_to_optimize.append(
        {
            "params": [p for n, p in head_params if ("bias" in n) or (len(p.shape) == 1)],
            "lr": CONFIG_CLASSIFY["learning_rate_classifier"],
            "weight_decay": 0,
        }
    )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    criterion = nn.CrossEntropyLoss()

    # ---  Configuring the Learning Rate Scheduler with Warmup ---
    total_steps_classify = len(train_loader) * CONFIG_CLASSIFY["num_epochs_classify"]
    warmup_epochs_classify = CONFIG_CLASSIFY.get("warmup_epochs_classify", 0)
    warmup_steps_classify = warmup_epochs_classify * len(train_loader)
    final_lr_classify = CONFIG_CLASSIFY.get("final_lr_classify", 0.0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps_classify - warmup_steps_classify,
        eta_min=final_lr_classify,
    )

    wallclock_start = time.perf_counter()
    last_train_loss = float("nan")
    last_train_acc = float("nan")
    last_val_loss = float("nan")
    last_val_acc = float("nan")
    best_val_acc = float("-inf")

    print(
        f"\n--- Starting Video Classification Training for {CONFIG_CLASSIFY['num_epochs_classify']} epochs ---"
    )
    log_message(
        f"Training set size: {len(train_dataset)}. DataLoader ready with {len(train_loader)} batches.",
        log_file_name,
    )
    log_message(
        f"Optimizer: AdamW | Encoder Fine-tune LR: {CONFIG_CLASSIFY['learning_rate_encoder_finetune']} | Classifier Head LR: {CONFIG_CLASSIFY['learning_rate_classifier']} | Weight Decay: {CONFIG_CLASSIFY['classify_weight_decay']}",
        log_file_name,
    )
    log_message(
        f"Scheduler: CosineAnnealingLR with {warmup_epochs_classify} warmup epochs. Final LR: {final_lr_classify}",
        log_file_name,
    )
    if CONFIG_CLASSIFY.get("clip_grad_norm_classify") is not None:
        log_message(
            f"Gradient Clipping: Enabled with norm {CONFIG_CLASSIFY['clip_grad_norm_classify']} (after warmup epochs).",
            log_file_name,
        )
    else:
        log_message(
            "Gradient Clipping: Disabled.",
            log_file_name,
        )

    for epoch in range(CONFIG_CLASSIFY["num_epochs_classify"]):
        classification_model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        total_train_samples = 0

        if not train_loader:
            print(
                "Error: Train loader is not initialized (dataset might be too small)."
            )
            break

        for i, batch_data in enumerate(train_loader):
            if batch_data is None:
                continue
            videos_batch, labels_batch = batch_data
            if videos_batch is None or videos_batch.nelement() == 0:
                continue

            current_step_classify = epoch * len(train_loader) + i

            # ---  Learning Rate Management (Warmup and Scheduler) ---
            if current_step_classify < warmup_steps_classify:
                warmup_factor = current_step_classify / warmup_steps_classify
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["initial_lr"] * warmup_factor
            else:
                scheduler.step()

            if (i + 1) % 50 == 0:
                current_lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
                print(f"Current LRs: {', '.join(current_lrs)}")

            videos_batch = videos_batch.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = classification_model(videos_batch)
            loss = criterion(logits, labels_batch)

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: NaN or Inf loss in training (Epoch {epoch + 1}, Batch {i + 1}). Skipping batch."
                )
                continue

            loss.backward()

            clip_grad_norm_value = CONFIG_CLASSIFY.get("clip_grad_norm_classify")

            if clip_grad_norm_value is not None and epoch >= warmup_epochs_classify:
                torch.nn.utils.clip_grad_norm_(
                    classification_model.parameters(), clip_grad_norm_value
                )

            optimizer.step()

            total_train_loss += loss.item()
            _, predicted_labels = torch.max(logits, 1)
            correct_train_predictions += (predicted_labels == labels_batch).sum().item()
            total_train_samples += labels_batch.size(0)

            if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
                print(
                    f"Epoch {epoch + 1} [Train] | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )
                log_message(
                    f"Epoch {epoch + 1} [Train] | Batch {i + 1}/{len(train_loader)} | Loss: {loss.item():.4f}",
                    log_file_name,
                )

        avg_train_loss = (
            total_train_loss / len(train_loader)
            if len(train_loader) > 0
            else float("inf")
        )
        train_accuracy = (
            (correct_train_predictions / total_train_samples) * 100
            if total_train_samples > 0
            else 0.0
        )
        last_train_loss = avg_train_loss
        last_train_acc = train_accuracy
        print(
            f"--- Epoch {epoch + 1} [Train] Summary --- Avg. Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2f}% ---"
        )
        log_message(
            f"--- Epoch {epoch + 1} [Train] Summary --- Avg. Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.2f}% ---",
            log_file_name,
        )

        if val_loader:
            classification_model.eval()
            total_val_loss = 0
            correct_val_predictions = 0
            total_val_samples = 0
            with torch.no_grad():
                for videos_batch, labels_batch in val_loader:
                    if videos_batch is None or videos_batch.nelement() == 0:
                        continue
                    videos_batch = videos_batch.to(device, non_blocking=True)
                    labels_batch = labels_batch.to(device, non_blocking=True)

                    logits = classification_model(videos_batch)
                    loss = criterion(logits, labels_batch)
                    total_val_loss += loss.item()
                    _, predicted_labels = torch.max(logits, 1)
                    correct_val_predictions += (
                        (predicted_labels == labels_batch).sum().item()
                    )
                    total_val_samples += labels_batch.size(0)

            avg_val_loss = (
                total_val_loss / len(val_loader)
                if len(val_loader) > 0
                else float("inf")
            )
            val_accuracy = (
                (correct_val_predictions / total_val_samples) * 100
                if total_val_samples > 0
                else 0.0
            )
            last_val_loss = avg_val_loss
            last_val_acc = val_accuracy
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
            print(
                f"--- Epoch {epoch + 1} [Validation] Summary --- Avg. Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}% ---"
            )
            log_message(
                f"--- Epoch {epoch + 1} [Validation] Summary --- Avg. Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.2f}% ---",
                log_file_name,
            )
        else:
            print(
                f"--- Epoch {epoch + 1} [Validation] --- No validation set to evaluate."
            )
        torch.save(classification_model.state_dict(), classifier_save_path)

    print("\n--- Video Classification Training finished. ---")

    print(f"Saving classification model state_dict to: {classifier_save_path}")
    torch.save(classification_model.state_dict(), classifier_save_path)
    print(f"Classification model saved successfully to {classifier_save_path}")

    wallclock_s = time.perf_counter() - wallclock_start
    config_hash = hashlib.sha256(
        json.dumps(CONFIG_CLASSIFY, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]

    if getattr(args, "output_json", None):
        metrics = {
            "phase": "classify",
            "final_train_loss": float(last_train_loss) if last_train_loss == last_train_loss else None,
            "final_train_acc": float(last_train_acc) if last_train_acc == last_train_acc else None,
            "final_val_loss": float(last_val_loss) if last_val_loss == last_val_loss else None,
            "final_val_acc": float(last_val_acc) if last_val_acc == last_val_acc else None,
            "best_val_acc": float(best_val_acc) if best_val_acc != float("-inf") else None,
            "wallclock_s": wallclock_s,
            "config_hash": config_hash,
            "classifier_save_path": classifier_save_path,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset) if val_dataset else 0,
            "num_classes": CONFIG_CLASSIFY["num_classes"],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PicoJEPA video classification."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the configuration file YAML (e.g., configs/config.yaml).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="If set, write final metrics (val_acc, wallclock_s, config_hash) to this JSON path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for torch/numpy/random. Overrides the 'seed' key in the YAML config. "
             "Used by autoresearch to diversify N submodels trained on the same data.",
    )
    args = parser.parse_args()
    train_video_classifier(args)
