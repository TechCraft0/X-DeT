"""CornerNet with Hourglass-104, corner pooling, and associative embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class _ConvModule(nn.Module):
    """Small configurable ConvModule block with compatible state names."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int,
                 *, stride: int = 1, padding: int | None = None,
                 normalize: bool = True, activate: bool = True) -> None:
        super().__init__()
        if padding is None:
            padding = kernel // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding,
                              bias=not normalize)
        self.bn = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()
        self.activate = activate

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.bn(self.conv(value))
        return F.relu(value, inplace=True) if self.activate else value


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1,
                 downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        identity = value
        value = self.relu(self.bn1(self.conv1(value)))
        value = self.bn2(self.conv2(value))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(value + identity)


def _res_layer(in_channels: int, channels: int, blocks: int,
               stride: int = 1, downsample_first: bool = True) -> nn.Sequential:
    modules: list[nn.Module] = []
    if downsample_first:
        downsample = None
        if stride != 1 or in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        modules.append(_BasicBlock(in_channels, channels, stride, downsample))
        modules.extend(_BasicBlock(channels, channels) for _ in range(blocks - 1))
    else:
        modules.extend(_BasicBlock(in_channels, in_channels) for _ in range(blocks - 1))
        downsample = None
        if stride != 1 or in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        modules.append(_BasicBlock(in_channels, channels, stride, downsample))
    return nn.Sequential(*modules)


class _HourglassModule(nn.Module):
    def __init__(self, depth: int, stage_channels: Sequence[int],
                 stage_blocks: Sequence[int]) -> None:
        super().__init__()
        current, next_channels = stage_channels[:2]
        current_blocks, next_blocks = stage_blocks[:2]
        self.up1 = _res_layer(current, current, current_blocks)
        self.low1 = _res_layer(current, next_channels, current_blocks, stride=2)
        self.low2 = (
            _HourglassModule(depth - 1, stage_channels[1:], stage_blocks[1:])
            if depth > 1 else _res_layer(next_channels, next_channels, next_blocks)
        )
        self.low3 = _res_layer(next_channels, current, current_blocks,
                               downsample_first=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        upper = self.up1(value)
        lower = self.low3(self.low2(self.low1(value)))
        return upper + F.interpolate(lower, size=upper.shape[-2:], mode="nearest")


class _Hourglass104(nn.Module):
    """Two stacked depth-five hourglasses, yielding two stride-4 maps."""

    def __init__(self) -> None:
        super().__init__()
        stage_channels = (256, 256, 384, 384, 384, 512)
        stage_blocks = (2, 2, 2, 2, 2, 4)
        self.stem = nn.Sequential(
            _ConvModule(3, 128, 7, stride=2, padding=3),
            _res_layer(128, 256, 1, stride=2),
        )
        self.hourglass_modules = nn.ModuleList(
            [_HourglassModule(5, stage_channels, stage_blocks) for _ in range(2)]
        )
        self.inters = _res_layer(256, 256, 1)
        self.conv1x1s = nn.ModuleList([_ConvModule(256, 256, 1, activate=False)])
        self.out_convs = nn.ModuleList([_ConvModule(256, 256, 3) for _ in range(2)])
        self.remap_convs = nn.ModuleList([_ConvModule(256, 256, 1, activate=False)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> list[torch.Tensor]:
        inter = self.stem(value)
        features: list[torch.Tensor] = []
        for index, hourglass in enumerate(self.hourglass_modules):
            output = self.out_convs[index](hourglass(inter))
            features.append(output)
            if index == 0:
                inter = self.inters[0](self.relu(self.conv1x1s[0](inter) + self.remap_convs[0](output)))
        return features


class _CornerPool(nn.Module):
    def __init__(self, direction: str) -> None:
        super().__init__()
        self.direction = direction

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.direction == "top":
            return value.flip(2).cummax(2).values.flip(2)
        if self.direction == "bottom":
            return value.cummax(2).values
        if self.direction == "left":
            return value.flip(3).cummax(3).values.flip(3)
        if self.direction == "right":
            return value.cummax(3).values
        raise ValueError(f"Unsupported corner-pooling direction: {self.direction}")


class _BiCornerPool(nn.Module):
    def __init__(self, in_channels: int, directions: tuple[str, str]) -> None:
        super().__init__()
        self.direction1_conv = _ConvModule(in_channels, 128, 3)
        self.direction2_conv = _ConvModule(in_channels, 128, 3)
        self.aftpool_conv = _ConvModule(128, in_channels, 3, activate=False)
        self.conv1 = _ConvModule(in_channels, in_channels, 1, activate=False)
        self.conv2 = _ConvModule(in_channels, in_channels, 3)
        self.direction1_pool = _CornerPool(directions[0])
        self.direction2_pool = _CornerPool(directions[1])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.direction1_pool(self.direction1_conv(value))
        second = self.direction2_pool(self.direction2_conv(value))
        pooled = self.aftpool_conv(first + second)
        return self.conv2(self.relu(pooled + self.conv1(value)))


def _corner_branch(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        _ConvModule(in_channels, 256, 3, normalize=False),
        _ConvModule(256, out_channels, 1, normalize=False, activate=False),
    )


class _CornerHead(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.tl_pool = nn.ModuleList([_BiCornerPool(256, ("top", "left")) for _ in range(2)])
        self.br_pool = nn.ModuleList([_BiCornerPool(256, ("bottom", "right")) for _ in range(2)])
        self.tl_heat = nn.ModuleList([_corner_branch(256, num_classes) for _ in range(2)])
        self.br_heat = nn.ModuleList([_corner_branch(256, num_classes) for _ in range(2)])
        self.tl_emb = nn.ModuleList([_corner_branch(256, 1) for _ in range(2)])
        self.br_emb = nn.ModuleList([_corner_branch(256, 1) for _ in range(2)])
        self.tl_off = nn.ModuleList([_corner_branch(256, 2) for _ in range(2)])
        self.br_off = nn.ModuleList([_corner_branch(256, 2) for _ in range(2)])
        for level in range(2):
            self.tl_heat[level][-1].conv.bias.data.fill_(-2.197224577)
            self.br_heat[level][-1].conv.bias.data.fill_(-2.197224577)

    def forward(self, features: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        outputs = []
        for level, feature in enumerate(features):
            top_left = self.tl_pool[level](feature)
            bottom_right = self.br_pool[level](feature)
            outputs.append({
                "top_left_heatmap_logits": self.tl_heat[level](top_left),
                "bottom_right_heatmap_logits": self.br_heat[level](bottom_right),
                "top_left_embedding": self.tl_emb[level](top_left),
                "bottom_right_embedding": self.br_emb[level](bottom_right),
                "top_left_offsets": self.tl_off[level](top_left),
                "bottom_right_offsets": self.br_off[level](bottom_right),
            })
        return outputs


class CornerNet(nn.Module):
    """CornerNet Hourglass-104 detector.

    Two hourglass stacks produce stride-4 features. Each head predicts paired
    corner heatmaps, sub-cell offsets, and scalar associative embeddings.
    Inputs are RGB float tensors in [0,1].
    """

    output_stride = 4

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("CornerNet needs at least one foreground class")
        self.num_classes = num_classes
        self.backbone = _Hourglass104()
        self.bbox_head = _CornerHead(num_classes)
        self.register_buffer("pixel_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        value = (images.float() * 255.0 - self.pixel_mean) / self.pixel_std
        return {"stacks": self.bbox_head(self.backbone(value)),
                "input_shape": (int(images.shape[-2]), int(images.shape[-1]))}

    def load_pretrained_checkpoint(self, checkpoint_path: str | Path,
                                   class_names: list[str]) -> dict[str, int | str]:
        """Load a public MMDetection CornerNet COCO checkpoint, remapping classes."""
        from x_yolo.models.keypoint_utils import coco_class_indices

        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CornerNet checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        source = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(source, dict):
            raise ValueError(f"No state dictionary found in {path}")
        source = {key.removeprefix("module."): value for key, value in source.items()}
        target = self.state_dict()
        class_indices = coco_class_indices(class_names)
        loaded: dict[str, torch.Tensor] = {}
        remapped = 0
        class_weight_keys = {
            f"bbox_head.{prefix}_heat.{level}.1.conv.{suffix}"
            for prefix in ("tl", "br") for level in range(2)
            for suffix in ("weight", "bias")
        }
        for key, value in source.items():
            if key in target and value.shape == target[key].shape:
                loaded[key] = value
            elif key in class_weight_keys and value.shape[0] == 80:
                loaded[key] = value[class_indices]
                remapped += len(class_indices)
        if not loaded:
            raise ValueError(f"No CornerNet tensors matched the model in {path}")
        self.load_state_dict(loaded, strict=False)
        matched = sum(target[key].numel() for key in loaded)
        total = sum(value.numel() for value in target.values())
        if matched / total < 0.90:
            raise ValueError(f"Only {matched / total:.1%} of CornerNet parameters matched {path}")
        return {"matched_tensors": len(loaded), "matched_parameter_fraction": matched / total,
                "remapped_class_channels": remapped,
                "source": str(path.resolve())}
