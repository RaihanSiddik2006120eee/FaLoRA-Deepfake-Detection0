"""Complete FaLoRA deepfake detection model."""

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .falora_attention import FaLoRAAttention
from .frequency_extractor import FrequencyExtractor


class FaLoRADetector(nn.Module):
    """
    Frequency-aware parameter-efficient deepfake detector.

    The model contains:

    - A frozen CLIP-pretrained ViT-B/16 backbone
    - A 9-channel frequency extractor
    - Frequency-gated LoRA modules in all transformer blocks
    - A binary classification head
    - A supervised contrastive projection head
    """

    def __init__(
        self,
        number_of_classes: int = 2,
        backbone_name: str = "vit_base_patch16_clip_224.openai",
        lora_rank: int = 16,
        frequency_dimension: int = 128,
        projection_dimension: int = 128,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        os.environ.setdefault(
            "HF_HUB_DISABLE_SYMLINKS_WARNING",
            "1",
        )

        self.frequency_extractor = FrequencyExtractor(
            frequency_dimension=frequency_dimension,
            image_size=image_size,
        )

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
        )

        embedding_dimension = self.backbone.embed_dim

        # Freeze every original backbone parameter.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Replace the attention module in every transformer block.
        for block in self.backbone.blocks:
            original_attention = block.attn

            block.attn = FaLoRAAttention(
                original_attention=original_attention,
                embedding_dimension=embedding_dimension,
                number_of_heads=original_attention.num_heads,
                rank=lora_rank,
                frequency_dimension=frequency_dimension,
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dimension),
            nn.Linear(embedding_dimension, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, number_of_classes),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                embedding_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                embedding_dimension,
                projection_dimension,
            ),
        )

    def set_frequency_token(
        self,
        frequency_token: torch.Tensor | None,
    ) -> None:
        for block in self.backbone.blocks:
            block.attn.set_frequency_token(
                frequency_token
            )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frequency_token = self.frequency_extractor(images)

        self.set_frequency_token(frequency_token)

        try:
            features = self.backbone.forward_features(images)
        finally:
            # Prevent a token from one batch being reused later.
            self.set_frequency_token(None)

        cls_token = features[:, 0]

        logits = self.classifier(cls_token)

        projected_features = self.projection_head(
            cls_token
        )

        projected_features = F.normalize(
            projected_features,
            dim=-1,
        )

        return logits, projected_features

    def count_parameters(self) -> dict[str, int]:
        """Return trainable and frozen parameter counts."""

        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        frozen = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not parameter.requires_grad
        )

        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": trainable + frozen,
        }
