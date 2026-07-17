"""Frequency-gated LoRA modules for Vision Transformer attention."""

import math

import torch
import torch.nn as nn


class FaLoRALayer(nn.Module):
    """
    Apply a frequency-conditioned low-rank update.

    The frequency token is fused with the stop-gradient CLS token.
    A sigmoid gate controls each LoRA rank direction independently.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        rank: int = 16,
        fusion_dimension: int = 896,
    ) -> None:
        super().__init__()

        self.lora_a = nn.Linear(
            input_dimension,
            rank,
            bias=False,
        )

        self.lora_b = nn.Linear(
            rank,
            output_dimension,
            bias=False,
        )

        self.frequency_gate = nn.Sequential(
            nn.Linear(fusion_dimension, rank),
            nn.LayerNorm(rank),
            nn.Sigmoid(),
        )

        nn.init.kaiming_uniform_(
            self.lora_a.weight,
            a=math.sqrt(5),
        )

        # Starting from zero preserves the frozen backbone initially.
        nn.init.zeros_(self.lora_b.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        fusion_token: torch.Tensor,
    ) -> torch.Tensor:
        gate = self.frequency_gate(fusion_token)

        if gate.ndim == 2 and tokens.ndim == 3:
            gate = gate.unsqueeze(1)

        low_rank_features = self.lora_a(tokens)
        gated_features = low_rank_features * gate

        return self.lora_b(gated_features)


class FaLoRAAttention(nn.Module):
    """
    Replace standard ViT attention with frequency-gated Q/V adaptation.

    The original query, key, value, and output projections remain frozen.
    LoRA updates are applied only to query and value projections.
    """

    def __init__(
        self,
        original_attention: nn.Module,
        embedding_dimension: int,
        number_of_heads: int,
        rank: int = 16,
        frequency_dimension: int = 128,
    ) -> None:
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self.scale = self.head_dimension**-0.5

        self.qkv = original_attention.qkv
        self.projection = original_attention.proj
        self.attention_dropout = original_attention.attn_drop
        self.projection_dropout = original_attention.proj_drop

        self.query_normalization = getattr(
            original_attention,
            "q_norm",
            nn.Identity(),
        )

        self.key_normalization = getattr(
            original_attention,
            "k_norm",
            nn.Identity(),
        )

        fusion_dimension = (
            frequency_dimension + embedding_dimension
        )

        self.query_lora = FaLoRALayer(
            input_dimension=embedding_dimension,
            output_dimension=embedding_dimension,
            rank=rank,
            fusion_dimension=fusion_dimension,
        )

        self.value_lora = FaLoRALayer(
            input_dimension=embedding_dimension,
            output_dimension=embedding_dimension,
            rank=rank,
            fusion_dimension=fusion_dimension,
        )

        self._frequency_token = None

    def set_frequency_token(
        self,
        frequency_token: torch.Tensor | None,
    ) -> None:
        self._frequency_token = frequency_token

    def forward(
        self,
        tokens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, token_count, channel_count = tokens.shape

        qkv = self.qkv(tokens).reshape(
            batch_size,
            token_count,
            3,
            self.number_of_heads,
            self.head_dimension,
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        query = self.query_normalization(query)
        key = self.key_normalization(key)

        if self._frequency_token is not None:
            cls_token = tokens[:, 0, :].detach()

            fusion_token = torch.cat(
                [
                    self._frequency_token,
                    cls_token,
                ],
                dim=-1,
            )

            query_update = self.query_lora(
                tokens,
                fusion_token,
            )

            value_update = self.value_lora(
                tokens,
                fusion_token,
            )

            query_update = query_update.reshape(
                batch_size,
                token_count,
                self.number_of_heads,
                self.head_dimension,
            ).permute(0, 2, 1, 3)

            value_update = value_update.reshape(
                batch_size,
                token_count,
                self.number_of_heads,
                self.head_dimension,
            ).permute(0, 2, 1, 3)

            query = query + query_update
            value = value + value_update

        attention = (
            query @ key.transpose(-2, -1)
        ) * self.scale

        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        output = attention @ value

        output = output.transpose(1, 2).reshape(
            batch_size,
            token_count,
            channel_count,
        )

        output = self.projection(output)

        return self.projection_dropout(output)
