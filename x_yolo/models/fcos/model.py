"""A compact FCOS implementation using TorchVision ResNet-FPN features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.models.resnet import ResNet50_Weights


class _Tower(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend((
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, channels),
                nn.ReLU(inplace=True),
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.layers(feature)


class FCOS(nn.Module):
    """FCOS with P3–P7 predictions and explicit ImageNet backbone loading.

    Images enter as RGB float tensors in [0, 1]. Feature strides are 8, 16,
    32, 64, and 128. A checkpoint path loads only the ResNet-50 body; weights
    are never downloaded by model construction.
    """

    strides = (8, 16, 32, 64, 128)

    def __init__(
        self,
        num_classes: int,
        *,
        backbone_checkpoint: str | Path | None = None,
        trainable_layers: int = 3,
        tower_depth: int = 4,
        channels: int = 256,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("FCOS needs at least one foreground class")
        if channels % 32:
            raise ValueError("FCOS head channels must be divisible by 32 for GroupNorm")
        self.num_classes = num_classes
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=trainable_layers,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(256, 256),
        )
        if backbone_checkpoint is not None:
            self.load_backbone_checkpoint(backbone_checkpoint)

        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.classification_tower = _Tower(channels, tower_depth)
        self.regression_tower = _Tower(channels, tower_depth)
        self.classifier = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.box_regressor = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(channels, 1, 3, padding=1)
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(())) for _ in self.strides])
        self._initialize_head()

    def _initialize_head(self) -> None:
        for module in (self.classifier, self.box_regressor, self.centerness):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)
        nn.init.constant_(self.classifier.bias, -4.59512)  # sigmoid prior 0.01

    def load_backbone_checkpoint(self, checkpoint_path: str | Path) -> int:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FCOS backbone checkpoint does not exist: {path}")
        checkpoint: Any = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(state, dict):
            raise ValueError(f"No state dictionary found in {path}")
        cleaned = {}
        for key, value in state.items():
            name = key.removeprefix("module.")
            for prefix in ("backbone.body.", "backbone.", "body."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            cleaned[name] = value
        own = self.backbone.body.state_dict()
        compatible = {key: value for key, value in cleaned.items() if key in own and own[key].shape == value.shape}
        coverage = sum(own[key].numel() for key in compatible) / sum(value.numel() for value in own.values())
        if coverage < 0.9:
            raise ValueError(f"Only {coverage:.1%} of ResNet-50 backbone matched {path}; expected >=90%")
        self.backbone.body.load_state_dict(compatible, strict=False)
        return len(compatible)

    def set_backbone_trainable(self, trainable: bool) -> None:
        # The FPN starts from random weights and must learn during the warmup;
        # only freeze/unfreeze the ImageNet-pretrained ResNet body.
        for parameter in self.backbone.body.parameters():
            parameter.requires_grad = trainable

    def forward(self, images: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        features = self.backbone((images - self.pixel_mean) / self.pixel_std)
        keys = ("0", "1", "2", "p6", "p7")
        class_logits: list[torch.Tensor] = []
        box_distances: list[torch.Tensor] = []
        centerness_logits: list[torch.Tensor] = []
        for index, key in enumerate(keys):
            feature = features[key]
            cls_feature = self.classification_tower(feature)
            reg_feature = self.regression_tower(feature)
            class_logits.append(self.classifier(cls_feature))
            # FCOS predicts positive distances in feature-grid units, then
            # multiplies by stride in both assignment and box decoding.
            box_distances.append(torch.exp(self.scales[index] * self.box_regressor(reg_feature).float()))
            centerness_logits.append(self.centerness(reg_feature))
        return {
            "class_logits": class_logits,
            "box_distances": box_distances,
            "centerness_logits": centerness_logits,
        }
