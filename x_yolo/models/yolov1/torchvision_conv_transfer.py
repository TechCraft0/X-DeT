"""YOLOv1 grid predictions with a compact head on TorchVision backbones.

This is an explicit transfer-learning experiment. The YOLOv1 target, loss and
decoder are unchanged; replacing the paper's fully connected head trades global
cell-to-cell mixing for far fewer parameters and spatial weight sharing.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class YoloV1TorchVisionConvHead(nn.Module):
    """A selected ImageNet feature extractor -> configurable YOLO grid head.

    Supported families have a 14x14 output at 448 input. Other TorchVision
    models need an explicit feature contract before they can be added here.
    """

    input_size = 448
    grid_size = 7
    boxes_per_cell = 2

    def __init__(
        self,
        num_classes: int,
        backbone_name: str,
        backbone_checkpoint: str | Path | None = None,
        backbone_weights: str | None = None,
        freeze_backbone: bool = True,
        grid_size: int = 7,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if bool(backbone_checkpoint) == bool(backbone_weights):
            raise ValueError("Specify exactly one of backbone_checkpoint or backbone_weights")
        if grid_size not in (7, 14):
            raise ValueError("The ConvNeXt convolutional head supports grid_size 7 or 14")
        supported = (
            "resnet18", "resnet34", "resnet50",
            "resnext50_32x4d", "regnet_y_400mf", "regnet_y_800mf",
            "efficientnet_b0", "efficientnet_v2_s", "convnext_tiny", "convnext_small",
            "mobilenet_v3_large",
        )
        if backbone_name not in supported:
            raise ValueError(f"Unsupported backbone {backbone_name!r}; use one of {supported}")
        try:
            from torchvision.models import get_model, get_model_weights
        except ImportError as exc:
            raise RuntimeError("The TorchVision transfer variant requires torchvision") from exc

        if backbone_weights:
            # This named TorchVision enum is explicit in the recipe. If missing
            # from the cache, TorchVision downloads it when this run starts.
            weight_enum = get_model_weights(backbone_name)
            try:
                weights = weight_enum[backbone_weights]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown {backbone_name} weights {backbone_weights!r}; "
                    f"available: {list(weight_enum.__members__)}"
                ) from exc
            source_model = get_model(backbone_name, weights=weights)
            transform = weights.transforms()
            mean, std = transform.mean, transform.std
        else:
            checkpoint_path = Path(backbone_checkpoint).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Backbone checkpoint does not exist: {checkpoint_path}")
            source_model = get_model(backbone_name, weights=None)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError(f"Expected a TorchVision {backbone_name} state dict in {checkpoint_path}")
            if state and all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            source_model.load_state_dict(state, strict=True)
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if backbone_name.startswith(("resnet", "resnext")):
            feature_channels = source_model.fc.in_features  # 512 for 18/34, 2048 for 50.
            self.backbone = nn.Sequential(*list(source_model.children())[:-2])
        elif backbone_name.startswith("regnet"):
            # RegNet names its feature path stem -> trunk_output; avgpool/fc are
            # classification-only and must not enter the 7x7 detector.
            feature_channels = source_model.fc.in_features
            self.backbone = nn.Sequential(source_model.stem, source_model.trunk_output)
        else:
            # These families expose a feature-only path directly at stride 32.
            feature_channels = {
                "efficientnet_b0": 1280,
                "efficientnet_v2_s": 1280,
                "convnext_tiny": 768,
                "convnext_small": 768,
                "mobilenet_v3_large": 960,
            }[backbone_name]
            self.backbone = source_model.features
        self.freeze_backbone = bool(freeze_backbone)
        self.grid_size = grid_size
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(not self.freeze_backbone)
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(std).view(1, 3, 1, 1))

        # The 7x7 paper grid downsamples the 14x14 feature map; the 14x14
        # experiment keeps one prediction cell per backbone feature location.
        # GroupNorm is stable with the small physical micro-batch.
        output_channels = 5 * self.boxes_per_cell + num_classes
        self.head = nn.Sequential(
            nn.Conv2d(feature_channels, 512, kernel_size=3, stride=2 if grid_size == 7 else 1, padding=1),
            nn.GroupNorm(32, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, output_channels, kernel_size=1),
        )
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        if self.freeze_backbone:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "YoloV1TorchVisionConvHead":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(bool(trainable))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, self.input_size, self.input_size):
            raise ValueError(f"Expected RGB [N,3,448,448] input, got {tuple(images.shape)}")
        normalized = (images - self.pixel_mean) / self.pixel_std
        features = self.backbone(normalized)
        if features.shape[-2:] != (14, 14):
            raise RuntimeError(f"Backbone features must be 14x14, got {tuple(features.shape[-2:])}")
        logits = self.head(features)
        if logits.shape[-2:] != (self.grid_size, self.grid_size):
            raise RuntimeError(
                f"Detection grid must be {self.grid_size}x{self.grid_size}, got {tuple(logits.shape[-2:])}"
            )
        # Preserve raw [N,S,S,5B+C] predictions. NMS stays outside the model.
        return logits.permute(0, 2, 3, 1).contiguous()
