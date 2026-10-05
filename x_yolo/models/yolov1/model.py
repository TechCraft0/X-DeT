"""Paper-shaped YOLOv1 network for a fixed 448x448 input."""
from __future__ import annotations

import torch
from torch import nn


class YoloV1(nn.Module):
    """24 convolutional layers and two fully connected layers from YOLOv1.

    The raw output is [batch, 7, 7, 5 * boxes_per_cell + num_classes].
    Each cell emits B tuples (x_offset, y_offset, sqrt_w, sqrt_h, confidence)
    followed by one set of conditional class scores.
    """

    input_size = 448
    grid_size = 7
    boxes_per_cell = 2

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)

        def conv(in_channels: int, out_channels: int, kernel: int, stride: int = 1) -> list[nn.Module]:
            return [
                nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
            ]

        layers: list[nn.Module] = []
        channels = 3

        def add_conv(out_channels: int, kernel: int, stride: int = 1) -> None:
            nonlocal channels
            layers.extend(conv(channels, out_channels, kernel, stride))
            channels = out_channels

        # The first five convolutions downsample 448x448 to 56x56.
        add_conv(64, 7, 2)
        layers.append(nn.MaxPool2d(2, 2))
        add_conv(192, 3)
        layers.append(nn.MaxPool2d(2, 2))
        add_conv(128, 1)
        add_conv(256, 3)
        add_conv(256, 1)
        add_conv(512, 3)
        layers.append(nn.MaxPool2d(2, 2))

        # Four 1x1 reduction + 3x3 feature blocks.
        for _ in range(4):
            add_conv(256, 1)
            add_conv(512, 3)

        add_conv(512, 1)
        add_conv(1024, 3)
        layers.append(nn.MaxPool2d(2, 2))

        # Two deeper reduction blocks.
        for _ in range(2):
            add_conv(512, 1)
            add_conv(1024, 3)

        add_conv(1024, 3)
        add_conv(1024, 3, stride=2)
        add_conv(1024, 3)
        add_conv(1024, 3)

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.grid_size * self.grid_size, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.grid_size * self.grid_size * (5 * self.boxes_per_cell + self.num_classes)),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB input, got shape {tuple(images.shape)}")
        if images.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f"YOLOv1 expects {self.input_size}x{self.input_size}, got {tuple(images.shape[-2:])}")
        features = self.features(images)
        if features.shape[-2:] != (self.grid_size, self.grid_size):
            raise RuntimeError(f"Feature map must be 7x7, got {tuple(features.shape[-2:])}")
        predictions = self.classifier(features)
        return predictions.view(
            images.shape[0], self.grid_size, self.grid_size, 5 * self.boxes_per_cell + self.num_classes
        )
