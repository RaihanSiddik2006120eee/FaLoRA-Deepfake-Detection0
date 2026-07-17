"""Frame-level evaluation utilities for FaLoRA."""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm


@torch.no_grad()
def evaluate_frame_level(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = True,
    description: str = "Evaluation",
) -> dict[str, Any]:
    """
    Evaluate frame-level AUC, accuracy, F1, and AP.

    Horizontal flipping is used for test-time augmentation.
    """

    model.eval()

    all_probabilities: list[float] = []
    all_labels: list[int] = []

    for batch in tqdm(
        data_loader,
        desc=description,
        leave=False,
    ):
        images = batch[0].to(
            device,
            non_blocking=True,
        )

        labels = batch[1].to(
            device,
            non_blocking=True,
        )

        autocast_enabled = device.type == "cuda"

        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
        ):
            logits, _ = model(images)

            probabilities = F.softmax(
                logits,
                dim=1,
            )[:, 1]

            if use_tta:
                flipped_images = torch.flip(
                    images,
                    dims=[3],
                )

                flipped_logits, _ = model(
                    flipped_images
                )

                flipped_probabilities = F.softmax(
                    flipped_logits,
                    dim=1,
                )[:, 1]

                probabilities = (
                    probabilities
                    + flipped_probabilities
                ) / 2.0

        all_probabilities.extend(
            probabilities.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

    labels_array = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    probabilities_array = np.asarray(
        all_probabilities,
        dtype=np.float32,
    )

    predictions_array = (
        probabilities_array >= 0.5
    ).astype(np.int64)

    has_both_classes = (
        np.unique(labels_array).size > 1
    )

    auc = (
        roc_auc_score(
            labels_array,
            probabilities_array,
        )
        if has_both_classes
        else float("nan")
    )

    average_precision = (
        average_precision_score(
            labels_array,
            probabilities_array,
        )
        if has_both_classes
        else float("nan")
    )

    return {
        "auc": float(auc),
        "accuracy": float(
            accuracy_score(
                labels_array,
                predictions_array,
            )
        ),
        "f1": float(
            f1_score(
                labels_array,
                predictions_array,
                zero_division=0,
            )
        ),
        "average_precision": float(
            average_precision
        ),
        "labels": labels_array,
        "probabilities": probabilities_array,
    }
