"""YOLOv1-style head over an ImageNet-pretrained ResNet-50 backbone."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class YoloV1ResNet50(nn.Module):
    """A transfer-learning variant inspired by tanjeffreyz/yolo-v1.

    This is an engineering variant, not the paper's Darknet backbone. It keeps
    the project's raw `[N,7,7,5B+C]` output contract so the same YOLOv1 loss,
    decoder, and evaluator can be used. The checkpoint path is always explicit;
    this class never downloads weights.
    """

    input_size = 448
    grid_size = 7
    boxes_per_cell = 2

    def __init__(
        self,
        num_classes: int,
        backbone_checkpoint: str | Path,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        checkpoint_path = Path(backbone_checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"ResNet-50 backbone checkpoint does not exist: {checkpoint_path}")
        try:
            from torchvision.models import resnet50
        except ImportError as exc:
            raise RuntimeError("The ResNet-50 variant requires the optional torchvision dependency") from exc

        backbone = resnet50(weights=None)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Expected a torchvision ResNet-50 state dict in {checkpoint_path}")
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        backbone.load_state_dict(state, strict=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.freeze_backbone = bool(freeze_backbone)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self.freeze_backbone)

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.grid_size * self.grid_size, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, self.grid_size * self.grid_size * (5 * self.boxes_per_cell + num_classes)),
        )
        self.num_classes = int(num_classes)
        if self.freeze_backbone:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "YoloV1ResNet50":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Enable or disable gradients for the complete ResNet feature extractor."""
        self.freeze_backbone = not bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2:] != (448, 448):
            raise ValueError(f"Expected RGB [N,3,448,448] input, got {tuple(images.shape)}")
        normalized = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(normalized)
        if features.shape[-2:] != (14, 14):
            raise RuntimeError(f"ResNet-50 feature map must be 14x14, got {tuple(features.shape[-2:])}")
        grid_features = self.head(features)
        if grid_features.shape[-2:] != (self.grid_size, self.grid_size):
            raise RuntimeError(f"Detection head must produce 7x7, got {tuple(grid_features.shape[-2:])}")
        output = self.detector(grid_features)
        return output.view(
            images.shape[0], self.grid_size, self.grid_size, 5 * self.boxes_per_cell + self.num_classes
        )
