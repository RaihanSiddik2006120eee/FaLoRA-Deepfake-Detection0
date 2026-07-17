"""Frequency-domain feature extractor used by FaLoRA."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct


class SRMFilters(nn.Module):
    """Extract three high-frequency noise residual maps."""

    def __init__(self) -> None:
        super().__init__()

        kernels = np.zeros((3, 1, 5, 5), dtype=np.float32)

        kernels[0, 0] = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, -4, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        kernels[1, 0] = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]

        kernels[2, 0, 1:4, 1:4] = [
            [-1, 2, -1],
            [2, -4, 2],
            [-1, 2, -1],
        ]

        self.register_buffer("weight", torch.tensor(kernels))

    def forward(self, grayscale: torch.Tensor) -> torch.Tensor:
        return F.conv2d(grayscale, self.weight, padding=2)


class FrequencyExtractor(nn.Module):
    """
    Produce a compact frequency token from an RGB image.

    The extractor combines:

    - Three DCT amplitude bands
    - Three FFT phase bands
    - Three SRM residual maps
    """

    def __init__(
        self,
        frequency_dimension: int = 128,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        self.srm_filters = SRMFilters()

        height = width = image_size

        # DCT radial masks.
        y_coordinates = torch.arange(height).float().view(-1, 1)
        y_coordinates = y_coordinates.expand(height, width)

        x_coordinates = torch.arange(width).float().view(1, -1)
        x_coordinates = x_coordinates.expand(height, width)

        dct_distance = torch.sqrt(
            y_coordinates**2 + x_coordinates**2
        )

        maximum_dct_distance = float(
            (height**2 + width**2) ** 0.5
        )

        self.register_buffer(
            "dct_low_mask",
            (dct_distance < maximum_dct_distance * 0.25).float(),
        )

        self.register_buffer(
            "dct_mid_mask",
            (
                (dct_distance >= maximum_dct_distance * 0.25)
                & (dct_distance < maximum_dct_distance * 0.55)
            ).float(),
        )

        self.register_buffer(
            "dct_high_mask",
            (dct_distance >= maximum_dct_distance * 0.55).float(),
        )

        # FFT radial masks.
        center_y = height // 2
        center_x = width // 2

        fft_y = (
            torch.arange(height).float() - center_y
        ).view(-1, 1).expand(height, width)

        fft_x = (
            torch.arange(width).float() - center_x
        ).view(1, -1).expand(height, width)

        fft_distance = torch.sqrt(fft_y**2 + fft_x**2)
        maximum_fft_distance = float(max(center_y, center_x))

        self.register_buffer(
            "fft_low_mask",
            (fft_distance < maximum_fft_distance * 0.25).float(),
        )

        self.register_buffer(
            "fft_mid_mask",
            (
                (fft_distance >= maximum_fft_distance * 0.25)
                & (fft_distance < maximum_fft_distance * 0.55)
            ).float(),
        )

        self.register_buffer(
            "fft_high_mask",
            (fft_distance >= maximum_fft_distance * 0.55).float(),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=9,
                out_channels=48,
                kernel_size=7,
                stride=4,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.projection = nn.Linear(
            96,
            frequency_dimension,
        )

        self.frequency_dropout = nn.Dropout(0.3)

    @staticmethod
    def convert_to_grayscale(images: torch.Tensor) -> torch.Tensor:
        """Convert normalized RGB images to grayscale."""

        return (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        grayscale = self.convert_to_grayscale(images)
        grayscale_2d = grayscale.squeeze(1)

        # DCT log-amplitude spectrum.
        dct_amplitude = torch.log(
            torch.abs(dct.dct_2d(grayscale_2d)) + 1e-8
        )

        dct_bands = torch.stack(
            [
                dct_amplitude * self.dct_low_mask,
                dct_amplitude * self.dct_mid_mask,
                dct_amplitude * self.dct_high_mask,
            ],
            dim=1,
        )

        # FFT phase spectrum.
        fft_spectrum = torch.fft.fft2(grayscale_2d)
        shifted_fft = torch.fft.fftshift(
            fft_spectrum,
            dim=(-2, -1),
        )
        fft_phase = torch.angle(shifted_fft)

        fft_bands = torch.stack(
            [
                fft_phase * self.fft_low_mask,
                fft_phase * self.fft_mid_mask,
                fft_phase * self.fft_high_mask,
            ],
            dim=1,
        )

        # SRM residual channels.
        srm_residuals = self.srm_filters(grayscale)

        frequency_channels = torch.cat(
            [
                dct_bands,
                fft_bands,
                srm_residuals,
            ],
            dim=1,
        )

        # Normalize each frequency channel independently.
        channel_mean = frequency_channels.mean(
            dim=(2, 3),
            keepdim=True,
        )
        channel_standard_deviation = frequency_channels.std(
            dim=(2, 3),
            keepdim=True,
        )

        frequency_channels = (
            frequency_channels - channel_mean
        ) / (channel_standard_deviation + 1e-8)

        encoded_features = self.encoder(frequency_channels)
        frequency_token = self.projection(encoded_features)

        return self.frequency_dropout(frequency_token)
