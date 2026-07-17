"""Loss functions and training utilities used by FaLoRA."""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for emphasizing difficult samples."""

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()

        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        correct_class_probability = torch.exp(
            -cross_entropy
        )

        focal_weight = (
            1.0 - correct_class_probability
        ) ** self.gamma

        return (
            focal_weight * cross_entropy
        ).mean()


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss for class separation."""

    def __init__(
        self,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = features.shape[0]

        if batch_size < 4:
            return features.sum() * 0.0

        device = features.device

        features = F.normalize(
            features,
            dim=-1,
        )

        labels = labels.view(-1, 1)

        positive_mask = torch.eq(
            labels,
            labels.T,
        ).float()

        self_mask = torch.eye(
            batch_size,
            device=device,
        )

        similarity = (
            features @ features.T
        ) / self.temperature

        similarity = similarity - similarity.max(
            dim=1,
            keepdim=True,
        ).values.detach()

        exponential_similarity = torch.exp(
            similarity
        ) * (1.0 - self_mask)

        log_probability = similarity - torch.log(
            exponential_similarity.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )

        positive_mask = positive_mask * (
            1.0 - self_mask
        )

        positive_count = positive_mask.sum(dim=1)
        valid_samples = positive_count > 0

        if not valid_samples.any():
            return features.sum() * 0.0

        mean_log_probability = (
            positive_mask * log_probability
        ).sum(dim=1)

        mean_log_probability = (
            mean_log_probability[valid_samples]
            / positive_count[valid_samples]
        )

        return -mean_log_probability.mean()


class ModelEMA:
    """Maintain an exponential moving average model."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9995,
    ) -> None:
        self.decay = decay

        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        for parameter in self.ema_model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
    ) -> None:
        model_parameters = dict(
            model.named_parameters()
        )

        for name, ema_parameter in (
            self.ema_model.named_parameters()
        ):
            source_parameter = model_parameters[name]

            ema_parameter.data.mul_(self.decay)
            ema_parameter.data.add_(
                source_parameter.data,
                alpha=1.0 - self.decay,
            )

        model_buffers = dict(model.named_buffers())

        for name, ema_buffer in (
            self.ema_model.named_buffers()
        ):
            if name in model_buffers:
                ema_buffer.data.copy_(
                    model_buffers[name].data
                )


def cosine_warmup_multiplier(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int = 3,
) -> float:
    """Return the learning-rate multiplier for one epoch."""

    if epoch < warmup_epochs:
        return float(epoch + 1) / float(
            max(1, warmup_epochs)
        )

    progress = (
        epoch - warmup_epochs
    ) / max(
        1,
        total_epochs - warmup_epochs,
    )

    return 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )
