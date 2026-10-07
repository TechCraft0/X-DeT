"""Darknet-53 and the three-scale YOLOv3 detection head."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_VOC_ANCHORS = (
    (10.0, 13.0), (16.0, 30.0), (33.0, 23.0),
    (30.0, 61.0), (62.0, 45.0), (59.0, 119.0),
    (116.0, 90.0), (156.0, 198.0), (373.0, 326.0),
)
DEFAULT_ANCHOR_MASKS = ((6, 7, 8), (3, 4, 5), (0, 1, 2))


class DarknetConv(nn.Module):
    """Darknet's bias-free Conv-BN-LeakyReLU block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(inputs)))


class DarknetResidual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.reduce = DarknetConv(channels, channels // 2, 1)
        self.expand = DarknetConv(channels // 2, channels, 3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.expand(self.reduce(inputs))


class Darknet53(nn.Module):
    """Return stride-8, stride-16 and stride-32 feature maps."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = DarknetConv(3, 32, 3)
        self.stage1 = self._stage(32, 64, 1)
        self.stage2 = self._stage(64, 128, 2)
        self.stage3 = self._stage(128, 256, 8)
        self.stage4 = self._stage(256, 512, 8)
        self.stage5 = self._stage(512, 1024, 4)

    @staticmethod
    def _stage(in_channels: int, out_channels: int, residual_blocks: int) -> nn.Sequential:
        layers: list[nn.Module] = [DarknetConv(in_channels, out_channels, 3, stride=2)]
        layers.extend(DarknetResidual(out_channels) for _ in range(residual_blocks))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.stem(images)
        features = self.stage1(features)
        features = self.stage2(features)
        stride8 = self.stage3(features)
        stride16 = self.stage4(stride8)
        stride32 = self.stage5(stride16)
        return stride8, stride16, stride32


def _detection_conv_set(in_channels: int, channels: int) -> nn.Sequential:
    """Five alternating 1x1/3x3 convolutions used before each YOLO head."""
    return nn.Sequential(
        DarknetConv(in_channels, channels, 1),
        DarknetConv(channels, channels * 2, 3),
        DarknetConv(channels * 2, channels, 1),
        DarknetConv(channels, channels * 2, 3),
        DarknetConv(channels * 2, channels, 1),
    )


class ConvNeXtSmallBackbone(nn.Module):
    """Expose TorchVision ConvNeXt-Small features at strides 8, 16, and 32."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from torchvision.models import convnext_small
        except ImportError as exc:
            raise RuntimeError("The ConvNeXt-Small YOLOv3 variant requires torchvision") from exc

        self.features = convnext_small(weights=None).features
        self.register_buffer("pixel_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = (images - self.pixel_mean) / self.pixel_std
        features = self.features[0](features)
        features = self.features[1](features)
        stride8 = self.features[3](self.features[2](features))
        stride16 = self.features[5](self.features[4](stride8))
        stride32 = self.features[7](self.features[6](stride16))
        return stride8, stride16, stride32


class YoloV3(nn.Module):
    """Return three raw heads in stride order 32, 16, 8.

    Each head has shape `[N,H/stride,W/stride,3,5+C]` and contains raw
    `tx, ty, tw, th, objectness, class_logits` without post-processing.
    Anchors are measured in pixels at the network input resolution.
    """

    stride = 32

    def __init__(
        self,
        num_classes: int,
        anchors: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
        anchor_masks: tuple[tuple[int, ...], ...] | list[list[int]] | None = None,
        backbone_name: str = "darknet53",
        backbone_weights: str | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        anchor_tensor = torch.as_tensor(
            DEFAULT_VOC_ANCHORS if anchors is None else anchors,
            dtype=torch.float32,
        )
        if anchor_tensor.shape != (9, 2) or not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
            raise ValueError("YOLOv3 requires nine finite, positive width-height anchors")
        masks = DEFAULT_ANCHOR_MASKS if anchor_masks is None else tuple(tuple(int(i) for i in row) for row in anchor_masks)
        if len(masks) != 3 or any(len(mask) != 3 for mask in masks):
            raise ValueError("anchor_masks must contain three groups of three indices")
        flat_masks = [index for mask in masks for index in mask]
        if sorted(flat_masks) != list(range(9)):
            raise ValueError("anchor_masks must use every anchor index exactly once")
        if backbone_name not in ("darknet53", "convnext_small"):
            raise ValueError("YOLOv3 supports backbone_name='darknet53' or 'convnext_small'")
        if backbone_name == "darknet53" and backbone_weights:
            raise ValueError("Named TorchVision weights require backbone_name='convnext_small'")

        self.num_classes = int(num_classes)
        self.anchor_masks = masks
        self.backbone_name = backbone_name
        self.backbone_weights = backbone_weights
        self.freeze_backbone = False
        self.register_buffer("anchors", anchor_tensor)
        if backbone_name == "darknet53":
            self.backbone = Darknet53()
            feature_channels = (256, 512, 1024)
        else:
            self.backbone = ConvNeXtSmallBackbone()
            feature_channels = (192, 384, 768)

        self.coarse_convs = _detection_conv_set(feature_channels[2], 512)
        self.coarse_pred = nn.Sequential(
            DarknetConv(512, 1024, 3),
            nn.Conv2d(1024, 3 * (5 + self.num_classes), 1),
        )
        self.coarse_to_middle = DarknetConv(512, 256, 1)

        self.middle_convs = _detection_conv_set(256 + feature_channels[1], 256)
        self.middle_pred = nn.Sequential(
            DarknetConv(256, 512, 3),
            nn.Conv2d(512, 3 * (5 + self.num_classes), 1),
        )
        self.middle_to_fine = DarknetConv(256, 128, 1)

        self.fine_convs = _detection_conv_set(128 + feature_channels[0], 128)
        self.fine_pred = nn.Sequential(
            DarknetConv(128, 256, 3),
            nn.Conv2d(256, 3 * (5 + self.num_classes), 1),
        )
        self._initialize_darknet_weights()

    def train(self, mode: bool = True) -> "YoloV3":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not bool(trainable)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(bool(trainable))

    def _initialize_darknet_weights(self) -> None:
        roots = [self.coarse_convs, self.coarse_pred, self.coarse_to_middle,
                 self.middle_convs, self.middle_pred, self.middle_to_fine,
                 self.fine_convs, self.fine_pred]
        if self.backbone_name == "darknet53":
            roots.insert(0, self.backbone)
        for root in roots:
            for module in root.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    @torch.no_grad()
    def load_torchvision_backbone_weights(self, weights_name: str) -> int:
        """Load an explicitly named TorchVision checkpoint into ConvNeXt-Small."""
        if self.backbone_name != "convnext_small":
            raise ValueError("TorchVision weights are supported only by the convnext_small backbone")
        try:
            from torchvision.models import get_model, get_model_weights
        except ImportError as exc:
            raise RuntimeError("Loading TorchVision weights requires torchvision") from exc

        weight_enum = get_model_weights(self.backbone_name)
        try:
            weights = weight_enum[weights_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown {self.backbone_name} weights {weights_name!r}; "
                f"available: {list(weight_enum.__members__)}"
            ) from exc
        source_model = get_model(self.backbone_name, weights=weights)
        state = source_model.features.state_dict()
        self.backbone.features.load_state_dict(state, strict=True)
        transform = weights.transforms()
        self.backbone.pixel_mean.copy_(torch.tensor(transform.mean).view(1, 3, 1, 1))
        self.backbone.pixel_std.copy_(torch.tensor(transform.std).view(1, 3, 1, 1))
        self.backbone_weights = weights_name
        return len(state)

    @torch.no_grad()
    def load_darknet53_weights(self, path: str | Path) -> int:
        """Load the official `darknet53.conv.74` backbone weights only."""
        if self.backbone_name != "darknet53":
            raise ValueError("Darknet-53 weights require backbone_name='darknet53'")
        weight_path = Path(path).expanduser()
        data = weight_path.read_bytes()
        if len(data) < 16:
            raise ValueError(f"Darknet weight file is truncated: {weight_path}")
        major, minor, _revision = struct.unpack("<3i", data[:12])
        seen_bytes = 8 if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000 else 4
        header_bytes = 12 + seen_bytes
        if len(data) < header_bytes:
            raise ValueError(f"Darknet weight header is truncated: {weight_path}")

        blocks = [module for module in self.backbone.modules() if isinstance(module, DarknetConv)]
        if len(blocks) != 52:
            raise RuntimeError(f"Expected 52 Darknet-53 convolution blocks, found {len(blocks)}")
        expected_floats = sum(
            4 * block.bn.num_features + block.conv.weight.numel() for block in blocks
        )
        values = np.frombuffer(data, dtype="<f4", offset=header_bytes)
        if values.size != expected_floats:
            raise ValueError(
                "Expected a Darknet-53 convolution-only checkpoint with 52 feature blocks; "
                f"found {values.size} floats in {weight_path} (expected {expected_floats})"
            )

        offset = 0

        def copy_values(target: torch.Tensor) -> None:
            nonlocal offset
            count = target.numel()
            source = torch.from_numpy(values[offset : offset + count].copy()).view_as(target)
            target.copy_(source.to(device=target.device, dtype=target.dtype))
            offset += count

        for block in blocks:
            # Darknet order: BN beta, gamma, running mean, running variance, kernels.
            copy_values(block.bn.bias)
            copy_values(block.bn.weight)
            copy_values(block.bn.running_mean)
            copy_values(block.bn.running_var)
            copy_values(block.conv.weight)
        return len(blocks)

    def _format_head(self, logits: torch.Tensor) -> torch.Tensor:
        batch, _, grid_height, grid_width = logits.shape
        values_per_anchor = 5 + self.num_classes
        return logits.view(batch, 3, values_per_anchor, grid_height, grid_width).permute(
            0, 3, 4, 1, 2
        ).contiguous()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB input, got shape {tuple(images.shape)}")
        height, width = images.shape[-2:]
        if height % self.stride or width % self.stride:
            raise ValueError(f"YOLOv3 input height and width must be divisible by {self.stride}")

        fine_backbone, middle_backbone, coarse_backbone = self.backbone(images)
        coarse = self.coarse_convs(coarse_backbone)
        coarse_logits = self._format_head(self.coarse_pred(coarse))

        middle_route = self.coarse_to_middle(coarse)
        middle_input = torch.cat(
            (F.interpolate(middle_route, size=middle_backbone.shape[-2:], mode="nearest"), middle_backbone),
            dim=1,
        )
        middle = self.middle_convs(middle_input)
        middle_logits = self._format_head(self.middle_pred(middle))

        fine_route = self.middle_to_fine(middle)
        fine_input = torch.cat(
            (F.interpolate(fine_route, size=fine_backbone.shape[-2:], mode="nearest"), fine_backbone),
            dim=1,
        )
        fine = self.fine_convs(fine_input)
        fine_logits = self._format_head(self.fine_pred(fine))
        return coarse_logits, middle_logits, fine_logits
