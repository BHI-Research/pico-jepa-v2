import os

import torch
import torch.nn as nn

from models.backbone import VitEncoder, Mlp


class AttentivePooler(nn.Module):
    """V-JEPA-style attentive probe.

    A learnable query token cross-attends over the (T*H*W) encoder tokens, then
    a 2-layer MLP refines the pooled vector. Designed to live on top of a
    *frozen* encoder; recovers temporal/spatial structure that average-pool
    discards. Adds ~D^2 * (1 + 2*mlp_ratio) params -- negligible vs the encoder.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            drop=dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, N, D) -> (B, D)
        B = tokens.shape[0]
        q = self.norm_q(self.query.expand(B, -1, -1))
        kv = self.norm_kv(tokens)
        attended, _ = self.cross_attn(q, kv, kv, need_weights=False)
        x = attended.squeeze(1)
        x = x + self.mlp(self.norm_mlp(x))
        return x


def build_head(head_type: str, embed_dim: int, num_classes: int, attn_heads: int = 4, mlp_ratio: float = 2.0):
    """Factory: returns (pool_module, classifier_linear).

    pool_module receives encoder tokens (B, N, D) and returns a vector (B, D).
    classifier_linear maps (B, D) -> (B, num_classes).
    """
    if head_type == "linear":
        pool = _AvgPool1dWrapper()
    elif head_type == "mlp_2layer":
        pool = nn.Sequential(_AvgPool1dWrapper(), Mlp(embed_dim, int(embed_dim * mlp_ratio), embed_dim))
    elif head_type == "attentive":
        pool = AttentivePooler(embed_dim, num_heads=attn_heads, mlp_ratio=mlp_ratio)
    else:
        raise ValueError(f"Unknown head_type {head_type!r}; valid: linear, mlp_2layer, attentive.")
    classifier = nn.Linear(embed_dim, num_classes)
    return pool, classifier


class _AvgPool1dWrapper(nn.Module):
    """tokens (B, N, D) -> (B, D) via mean over N. Replaces the AdaptiveAvgPool1d
    + transpose dance the original code did, with the same numerical result.
    """
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


class VideoClassifier(nn.Module):
    def __init__(
        self,
        encoder_config,
        num_classes,
        freeze_encoder=True,
        pretrained_encoder_path=None,
    ):
        super().__init__()
        self.encoder = VitEncoder(
            C_in=encoder_config["video_channels"],
            T_video=encoder_config["frames_per_clip"],
            H_video=encoder_config["resize_height"],
            W_video=encoder_config["resize_width"],
            patch_t=encoder_config["vit_patch_size_t"],
            patch_h=encoder_config["vit_patch_size_h"],
            patch_w=encoder_config["vit_patch_size_w"],
            embed_dim=encoder_config["vit_embed_dim"],
            depth=encoder_config["vit_depth"],
            num_heads=encoder_config["vit_num_heads"],
            mlp_ratio=encoder_config["vit_mlp_ratio"],
            dropout=encoder_config.get(
                "vit_dropout", 0.0
            ),
        )

        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            print(
                f"Loading pre-trained ENCODER weights for VideoClassifier from: {pretrained_encoder_path}"
            )
            state = torch.load(pretrained_encoder_path, map_location="cpu")
            # Fail loudly on shape mismatch. Previously this was caught and
            # silently fell back to random init -- the classifier would then
            # train on noise (~chance accuracy) without any visible error.
            self.encoder.load_state_dict(state, strict=True)
            print(
                "Pre-trained ENCODER weights loaded successfully into VideoClassifier's encoder."
            )
        elif pretrained_encoder_path:
            raise FileNotFoundError(
                f"Pretrained encoder not found at {pretrained_encoder_path}. "
                "Aborting: a classifier trained on a random encoder is meaningless."
            )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        head_type = encoder_config.get("head_type", "linear")
        attn_heads = encoder_config.get("head_attn_heads", 4)
        mlp_ratio = encoder_config.get("head_mlp_ratio", 2.0)
        self.head_type = head_type
        self.pool, self.classifier_head = build_head(
            head_type=head_type,
            embed_dim=encoder_config["vit_embed_dim"],
            num_classes=num_classes,
            attn_heads=attn_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x, return_features=False):
        features = self.encoder(x)  # (B, num_patches, embed_dim)
        pooled = self.pool(features)  # (B, embed_dim)
        logits = self.classifier_head(pooled)
        if return_features:
            return logits, features
        return logits
