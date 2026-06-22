# --- PicoJEPA_ViT Model (for Pre-training) ---
import copy
import math

import torch
import torch.nn as nn

from models.backbone import VitEncoder, Predictor

from utils.utils import generate_multiblock_masks, generate_spatiotemporal_masks

class PicoJEPA_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config["vit_embed_dim"]
        self.mask_ratio = config["mask_ratio"]
        # Phase B: EMA cosine schedule. ema_decay (legacy) is the constant
        # fallback. ema_decay_start/end define the cosine endpoints; if absent
        # we keep the constant behavior so old configs work unchanged.
        self.ema_decay = config["ema_decay"]
        self.ema_decay_start = config.get("ema_decay_start", config["ema_decay"])
        self.ema_decay_end = config.get("ema_decay_end", config["ema_decay"])
        self._step = 0
        self._total_steps = max(1, int(config.get("total_steps", 1)))

        # Phase B: multi-block tubelet masking. masking_strategy={tubelet,multiblock}.
        self.masking_strategy = config.get("masking_strategy", "tubelet")
        self.num_mask_blocks = int(config.get("num_mask_blocks", 1))

        self.num_patches_t = config["frames_per_clip"] // config["vit_patch_size_t"]
        self.num_patches_h = config["resize_height"] // config["vit_patch_size_h"]
        self.num_patches_w = config["resize_width"] // config["vit_patch_size_w"]
        self.num_total_patches = (
                self.num_patches_t * self.num_patches_h * self.num_patches_w
        )

        self.online_encoder = VitEncoder(
            C_in=config["video_channels"],
            T_video=config["frames_per_clip"],
            H_video=config["resize_height"],
            W_video=config["resize_width"],
            patch_t=config["vit_patch_size_t"],
            patch_h=config["vit_patch_size_h"],
            patch_w=config["vit_patch_size_w"],
            embed_dim=config["vit_embed_dim"],
            depth=config["vit_depth"],
            num_heads=config["vit_num_heads"],
            mlp_ratio=config["vit_mlp_ratio"],
            dropout=config["vit_dropout"],
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.predictor = Predictor(
            embed_dim=config["vit_embed_dim"],
            depth=config["predictor_depth"],
            num_heads=config["predictor_heads"],
            mlp_ratio=config[
                "vit_mlp_ratio"
            ],
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config["vit_embed_dim"]))
        nn.init.normal_(self.mask_token, std=0.02)

        self.loss_fn = nn.MSELoss()

    def set_step(self, step: int, total_steps: int = None) -> None:
        """Called by the training loop to drive the EMA cosine schedule.

        ``total_steps`` is optional; if absent, whatever was set in __init__
        from ``config.total_steps`` is used. The schedule is a no-op when
        ``ema_decay_start == ema_decay_end`` (legacy constant decay).
        """
        self._step = int(step)
        if total_steps is not None:
            self._total_steps = max(1, int(total_steps))

    def _current_ema_decay(self) -> float:
        """Cosine schedule from ``ema_decay_start`` to ``ema_decay_end``.

        If start == end the schedule is flat (legacy constant decay).
        """
        if self.ema_decay_start == self.ema_decay_end:
            return float(self.ema_decay)
        t = min(self._step, self._total_steps) / float(self._total_steps)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(self.ema_decay_end + (self.ema_decay_start - self.ema_decay_end) * cos)

    def _update_target_encoder(self):
        m = self._current_ema_decay()
        for online_param, target_param in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data = (
                    m * target_param.data
                    + (1 - m) * online_param.data
            )

    def _generate_masks(self, B, device):
        """
        Generates spatio-temporal "tubelet" masks using the shared utility.
        Switches between single-block (legacy) and multi-block strategies.
        Both guarantee uniform per-sample mask cardinality so the gather/
        reshape paths below work without padding logic.
        """
        if self.masking_strategy == "multiblock" and self.num_mask_blocks > 1:
            spatiotemporal_mask = generate_multiblock_masks(
                num_patches_t=self.num_patches_t,
                num_patches_h=self.num_patches_h,
                num_patches_w=self.num_patches_w,
                mask_ratio=self.mask_ratio,
                num_blocks=self.num_mask_blocks,
                device=device,
                batch_size=B,
            )
        else:
            spatiotemporal_mask = generate_spatiotemporal_masks(
                num_patches_t=self.num_patches_t,
                num_patches_h=self.num_patches_h,
                num_patches_w=self.num_patches_w,
                mask_ratio=self.mask_ratio,
                device=device,
                batch_size=B,
            )
        # The shared util squeezes the batch dim when B==1; restore 2D shape.
        if spatiotemporal_mask.dim() == 1:
            spatiotemporal_mask = spatiotemporal_mask.unsqueeze(0)

        # 2. Convert the boolean mask to context and target indices
        all_indices = torch.arange(self.num_total_patches, device=device).expand(B, -1)

        # Invert the mask to get context indices (visible patches)
        context_indices = all_indices[~spatiotemporal_mask].reshape(B, -1)

        # Use the mask directly for target indices (occluded patches)
        target_indices = all_indices[spatiotemporal_mask].reshape(B, -1)

        return context_indices, target_indices


    def forward(self, videos):
        B = videos.shape[0]
        device = videos.device

        # Generate masks using the new tubelet strategy
        context_indices, target_indices = self._generate_masks(
            B, device
        )

        # Process online encoder with all patches
        all_patch_embeddings_online = self.online_encoder(videos)
        
        # Gather context (visible) embeddings
        context_embeddings = torch.gather(
            all_patch_embeddings_online,
            dim=1,
            index=context_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )

        with torch.no_grad():
            self._update_target_encoder()
            all_patch_embeddings_target = self.target_encoder(videos)

        # Gather true target (occluded) embeddings
        true_target_embeddings = torch.gather(
            all_patch_embeddings_target,
            dim=1,
            index=target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )

        num_targets = target_indices.shape[1]
        
        # Prepare input for the predictor: mask tokens + positional embeddings of targets
        target_pos_embeddings = torch.gather(
            self.online_encoder.pos_embed.expand(
                B, -1, -1
            ),
            dim=1,
            index=target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        predictor_target_input_tokens = (
                self.mask_token.expand(B, num_targets, -1) + target_pos_embeddings
        )

        # Predict the embeddings of the target patches
        predicted_target_embeddings = self.predictor(
            predictor_target_input_tokens, context_embeddings
        )
        
        loss = self.loss_fn(predicted_target_embeddings, true_target_embeddings)
        return loss
