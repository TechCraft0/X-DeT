"""ResNet-18 CenterNet: center heatmap, box size, and sub-cell offset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class _ConvModule(nn.Module):
    """Conv + BatchNorm with state names compatible with the reference neck."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 *, stride: int = 1, padding: int = 0, transpose: bool = False) -> None:
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
            )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(value)))


class _CTResNetNeck(nn.Module):
    """Three deconvolution stages that restore stride-32 ResNet features to stride 4."""

    def __init__(self) -> None:
        super().__init__()
        self.deconv_layers = nn.Sequential(
            _ConvModule(512, 256, 3, padding=1),
            _ConvModule(256, 256, 4, stride=2, padding=1, transpose=True),
            _ConvModule(256, 128, 3, padding=1),
            _ConvModule(128, 128, 4, stride=2, padding=1, transpose=True),
            _ConvModule(128, 64, 3, padding=1),
            _ConvModule(64, 64, 4, stride=2, padding=1, transpose=True),
        )
        self._initialize_upsampling()

    def _initialize_upsampling(self) -> None:
        # Bilinear initialization makes each transposed convolution start as a
        # smooth 2x upsampler; later training can still learn its weights.
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                kernel = module.kernel_size[0]
                factor = (kernel + 1) // 2
                center = factor - 1 if kernel % 2 == 1 else factor - 0.5
                ramp = 1 - (torch.arange(kernel).float() / factor - center / factor).abs()
                bilinear = ramp[:, None] * ramp[None, :]
                with torch.no_grad():
                    module.weight.zero_()
                    for channel in range(module.out_channels):
                        module.weight[channel, channel].copy_(bilinear)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.deconv_layers(feature)


class _CenterNetHead(nn.Module):
    def __init__(self, num_classes: int, channels: int = 64) -> None:
        super().__init__()
        self.heatmap_head = self._branch(channels, num_classes)
        self.wh_head = self._branch(channels, 2)
        self.offset_head = self._branch(channels, 2)
        nn.init.constant_(self.heatmap_head[-1].bias, -2.197224577)
        for branch in (self.wh_head, self.offset_head):
            for layer in branch.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.001)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    @staticmethod
    def _branch(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.heatmap_head(feature), self.wh_head(feature), self.offset_head(feature)


class CenterNet(nn.Module):
    """CenterNet Objects-as-Points with a TorchVision ResNet-18 backbone.

    The input is RGB float `[0,1]`. The backbone receives ImageNet-normalized
    RGB values. For a 512 square input the head produces 128x128 maps (stride 4).
    """

    output_stride = 4

    def __init__(self, num_classes: int, *, backbone_weights: str | None = None) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("CenterNet needs at least one foreground class")
        if backbone_weights not in (None, "IMAGENET1K_V1"):
            raise ValueError("CenterNet ResNet-18 supports backbone_weights=null or IMAGENET1K_V1")
        self.num_classes = num_classes
        # No network access occurs for the default. Selecting the enum explicitly
        # opts into TorchVision's ImageNet weight cache/download behavior.
        weights = ResNet18_Weights.IMAGENET1K_V1 if backbone_weights == "IMAGENET1K_V1" else None
        self.backbone = resnet18(weights=weights)
        # This detector uses layer4 as features and has no image-classification head.
        self.backbone.fc = nn.Identity()
        self.neck = _CTResNetNeck()
        self.bbox_head = _CenterNetHead(num_classes)
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        self._coco_channels_remapped = 0

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        value = (images.float() * 255.0 - self.pixel_mean) / self.pixel_std
        value = self.backbone.conv1(value)
        value = self.backbone.bn1(value)
        value = self.backbone.relu(value)
        value = self.backbone.maxpool(value)
        value = self.backbone.layer1(value)
        value = self.backbone.layer2(value)
        value = self.backbone.layer3(value)
        value = self.backbone.layer4(value)
        feature = self.neck(value)
        heatmap_logits, box_sizes, offsets = self.bbox_head(feature)
        return {
            "center_heatmap_logits": heatmap_logits,
            "box_sizes": box_sizes,
            "center_offsets": offsets,
            "input_shape": (int(images.shape[-2]), int(images.shape[-1])),
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def load_pretrained_checkpoint(self, checkpoint_path: str | Path,
                                   class_names: list[str]) -> dict[str, int | str]:
        """Load matching MMDetection CenterNet COCO weights and remap heatmaps."""
        from x_yolo.models.keypoint_utils import coco_class_indices

        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CenterNet checkpoint does not exist: {path}")
        checkpoint: Any = torch.load(path, map_location="cpu", weights_only=True)
        source = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(source, dict):
            raise ValueError(f"No state dictionary found in {path}")
        source = {key.removeprefix("module."): value for key, value in source.items()}
        target = self.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        remapped = 0
        class_indices = coco_class_indices(class_names)
        for key, value in source.items():
            if key not in target:
                continue
            if value.shape == target[key].shape:
                loaded[key] = value
            elif key in ("bbox_head.heatmap_head.2.weight", "bbox_head.heatmap_head.2.bias"):
                if value.shape[0] == 80 and len(class_indices) == self.num_classes:
                    loaded[key] = value[class_indices]
                    remapped = self.num_classes
        if not loaded:
            raise ValueError(f"No CenterNet tensors matched the model in {path}")
        self.load_state_dict(loaded, strict=False)
        matched_numel = sum(target[key].numel() for key in loaded)
        total_numel = sum(value.numel() for value in target.values())
        if matched_numel / total_numel < 0.90:
            raise ValueError(
                f"Only {matched_numel / total_numel:.1%} of CenterNet parameters matched {path}"
            )
        self._coco_channels_remapped = remapped
        return {"matched_tensors": len(loaded), "matched_parameter_fraction": matched_numel / total_numel,
                "remapped_class_channels": remapped,
                "source": str(path.resolve())}
