"""Standalone RTMDet-Tiny modules, following the public MMDetection layout."""
from __future__ import annotations

from pathlib import Path
import inspect
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """The Conv-BN-SiLU block used throughout CSPNeXt and its neck."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int | None = None, groups: int = 1) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.depthwise_conv = ConvModule(channels, channels, kernel_size,
                                         padding=kernel_size // 2, groups=channels)
        self.pointwise_conv = ConvModule(channels, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    """A 3x3 Conv followed by depthwise 5x5 Conv and a residual connection."""

    def __init__(self, channels: int, add_identity: bool = True) -> None:
        super().__init__()
        hidden_channels = channels
        self.conv1 = ConvModule(channels, hidden_channels, 3)
        self.conv2 = DepthwiseSeparableConv(hidden_channels, 5)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.conv2(self.conv1(x))
        return result + x if self.add_identity else result


class CSPLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int,
                 *, add_identity: bool = True, channel_attention: bool = False) -> None:
        super().__init__()
        mid_channels = out_channels // 2
        self.main_conv = ConvModule(in_channels, mid_channels, 1, padding=0)
        self.short_conv = ConvModule(in_channels, mid_channels, 1, padding=0)
        self.final_conv = ConvModule(mid_channels * 2, out_channels, 1, padding=0)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid_channels, add_identity=add_identity) for _ in range(num_blocks)
        ])
        self.channel_attention = channel_attention
        if channel_attention:
            self.attention = ChannelAttention(mid_channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.blocks(self.main_conv(x))
        short = self.short_conv(x)
        joined = torch.cat((main, short), dim=1)
        if self.channel_attention:
            joined = self.attention(joined)
        return self.final_conv(joined)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.global_avgpool(x)))


class SPPBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        mid_channels = channels // 2
        self.conv1 = ConvModule(channels, mid_channels, 1, padding=0)
        self.poolings = nn.ModuleList([nn.MaxPool2d(size, stride=1, padding=size // 2) for size in (5, 9, 13)])
        self.conv2 = ConvModule(mid_channels * 4, channels, 1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        return self.conv2(torch.cat((x, *(pool(x) for pool in self.poolings)), dim=1))


class CSPNeXt(nn.Module):
    """CSPNeXt-P5 Tiny: outputs stride-8/16/32 features (96/192/384 ch)."""

    def __init__(self, deepen_factor: float = 0.167, widen_factor: float = 0.375) -> None:
        super().__init__()
        base_channels = 64
        stem_channels = int(base_channels * widen_factor // 2)
        self.stem = nn.Sequential(
            ConvModule(3, stem_channels, 3, stride=2),
            ConvModule(stem_channels, stem_channels, 3),
            ConvModule(stem_channels, int(base_channels * widen_factor), 3),
        )
        stage_settings = ((64, 128, 3, True, False), (128, 256, 6, True, False),
                          (256, 512, 6, True, False), (512, 1024, 3, False, True))
        in_channels = int(64 * widen_factor)
        self.out_channels = []
        for index, (_, base_out, base_blocks, add_identity, use_spp) in enumerate(stage_settings, 1):
            out_channels = int(base_out * widen_factor)
            stage: list[nn.Module] = [ConvModule(in_channels, out_channels, 3, stride=2)]
            if use_spp:
                stage.append(SPPBottleneck(out_channels))
            stage.append(CSPLayer(
                out_channels, out_channels, max(round(base_blocks * deepen_factor), 1),
                add_identity=add_identity, channel_attention=True,
            ))
            self.add_module(f"stage{index}", nn.Sequential(*stage))
            in_channels = out_channels
            if index >= 2:
                self.out_channels.append(out_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        outputs = []
        for index in range(1, 5):
            x = getattr(self, f"stage{index}")(x)
            if index >= 2:
                outputs.append(x)
        return tuple(outputs)


class CSPNeXtPAFPN(nn.Module):
    """Top-down and bottom-up path aggregation with CSPNeXt fusion blocks."""

    def __init__(self, in_channels: Sequence[int] = (96, 192, 384), out_channels: int = 96,
                 num_csp_blocks: int = 1) -> None:
        super().__init__()
        self.in_channels = tuple(in_channels)
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for index in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(ConvModule(in_channels[index], in_channels[index - 1], 1, padding=0))
            self.top_down_blocks.append(CSPLayer(
                in_channels[index - 1] * 2, in_channels[index - 1], num_csp_blocks, add_identity=False
            ))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for index in range(len(in_channels) - 1):
            self.downsamples.append(ConvModule(in_channels[index], in_channels[index], 3, stride=2))
            self.bottom_up_blocks.append(CSPLayer(
                in_channels[index] * 2, in_channels[index + 1], num_csp_blocks, add_identity=False
            ))
        self.out_convs = nn.ModuleList([ConvModule(c, out_channels, 3) for c in in_channels])

    def forward(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        inner = [inputs[-1]]
        for index in range(len(self.in_channels) - 1, 0, -1):
            high = self.reduce_layers[len(self.in_channels) - 1 - index](inner[0])
            inner[0] = high
            low = inputs[index - 1]
            inner.insert(0, self.top_down_blocks[len(self.in_channels) - 1 - index](
                torch.cat((self.upsample(high), low), dim=1)
            ))
        outputs = [inner[0]]
        for index in range(len(self.in_channels) - 1):
            downsampled = self.downsamples[index](outputs[-1])
            outputs.append(self.bottom_up_blocks[index](torch.cat((downsampled, inner[index + 1]), dim=1)))
        return tuple(conv(feature) for conv, feature in zip(self.out_convs, outputs))


class _SepBNHeadConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class RTMDetSepBNHead(nn.Module):
    """RTMDet's shared-convolution, per-level-BN classification/regression head."""

    strides = (8, 16, 32)

    def __init__(self, num_classes: int, channels: int = 96, stacked_convs: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        shared_cls: list[nn.Conv2d] = []
        shared_reg: list[nn.Conv2d] = []
        for level in range(3):
            cls_level, reg_level = nn.ModuleList(), nn.ModuleList()
            for layer in range(stacked_convs):
                incoming = channels
                cls_conv = _SepBNHeadConv(incoming, channels)
                reg_conv = _SepBNHeadConv(incoming, channels)
                if level == 0:
                    shared_cls.append(cls_conv.conv)
                    shared_reg.append(reg_conv.conv)
                else:
                    cls_conv.conv = shared_cls[layer]
                    reg_conv.conv = shared_reg[layer]
                cls_level.append(cls_conv)
                reg_level.append(reg_conv)
            self.cls_convs.append(cls_level)
            self.reg_convs.append(reg_level)
        self.rtm_cls = nn.ModuleList([nn.Conv2d(channels, num_classes, 1) for _ in self.strides])
        self.rtm_reg = nn.ModuleList([nn.Conv2d(channels, 4, 1) for _ in self.strides])
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        prior_bias = -torch.log(torch.tensor((1 - 0.01) / 0.01)).item()
        for classifier, regressor in zip(self.rtm_cls, self.rtm_reg):
            nn.init.constant_(classifier.bias, prior_bias)
            nn.init.normal_(regressor.weight, std=0.01)

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        class_logits, box_distances = [], []
        for level, feature in enumerate(features):
            cls_feature, reg_feature = feature, feature
            for layer in self.cls_convs[level]:
                cls_feature = layer(cls_feature)
            for layer in self.reg_convs[level]:
                reg_feature = layer(reg_feature)
            class_logits.append(self.rtm_cls[level](cls_feature))
            # RTMDet-S/Tiny uses linear, stride-scaled LTRB distances.
            box_distances.append(self.rtm_reg[level](reg_feature) * self.strides[level])
        return class_logits, box_distances


_COCO_INDEX = {
    "person": 0, "bicycle": 1, "car": 2, "motorbike": 3, "aeroplane": 4,
    "bus": 5, "train": 6, "boat": 8, "bird": 14, "cat": 15, "dog": 16,
    "horse": 17, "sheep": 18, "cow": 19, "bottle": 39, "chair": 56,
    "sofa": 57, "pottedplant": 58, "diningtable": 60, "tvmonitor": 62,
}


class RTMDet(nn.Module):
    """A standalone RTMDet-Tiny configuration for horizontal object boxes.

    Input is RGB float [0, 1]. It is converted to BGR 0-255 and normalized
    with the MMDetection RTMDet recipe before entering CSPNeXt.
    """

    strides = (8, 16, 32)

    def __init__(self, num_classes: int, *, channels: int = 96) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("RTMDet needs at least one foreground class")
        self.num_classes = num_classes
        self.backbone = CSPNeXt()
        self.neck = CSPNeXtPAFPN(self.backbone.out_channels, channels, num_csp_blocks=1)
        self.bbox_head = RTMDetSepBNHead(num_classes, channels, stacked_convs=2)
        self.register_buffer("pixel_mean", torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([57.375, 57.12, 58.395]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        bgr = images[:, [2, 1, 0]] * 255.0
        features = self.neck(self.backbone((bgr - self.pixel_mean) / self.pixel_std))
        class_logits, box_distances = self.bbox_head(features)
        return {"class_logits": class_logits, "box_distances": box_distances}

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def load_pretrained_checkpoint(self, checkpoint_path: str | Path, class_names: list[str]) -> dict[str, Any]:
        """Load matching RTMDet-Tiny COCO weights and remap VOC class channels."""
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"RTMDet checkpoint does not exist: {path}")
        # MMDetection checkpoints also pickle runner metadata. Allowlist the
        # inert metadata types, then let Torch's restricted unpickler read only
        # the tensors; loading the public checkpoint does not execute pickle code.
        class _HistoryBufferMetadata:
            pass

        safe_types: list[Any] = [
            (_HistoryBufferMetadata, "mmengine.logging.history_buffer.HistoryBuffer"),
            np.core.multiarray._reconstruct,
            np.core.multiarray.scalar,
            np.ndarray,
            np.dtype,
        ]
        numpy_dtypes = getattr(np, "dtypes", None)
        if numpy_dtypes is not None:
            safe_types.extend(value for value in vars(numpy_dtypes).values() if inspect.isclass(value))
        with torch.serialization.safe_globals(safe_types):
            checkpoint: Any = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("ema_state_dict", checkpoint.get("state_dict", checkpoint.get("model", checkpoint)))
        if not isinstance(state, dict):
            raise ValueError(f"No state dictionary found in {path}")
        source = {key.removeprefix("module."): value for key, value in state.items()}
        target = self.state_dict()
        loaded: dict[str, torch.Tensor] = {}
        remapped_classes = 0
        for key, value in source.items():
            if key not in target:
                continue
            if value.shape == target[key].shape:
                loaded[key] = value
            elif key.startswith("bbox_head.rtm_cls.") and value.ndim in (1, 4):
                indices = [_COCO_INDEX.get(name) for name in class_names]
                if value.shape[0] == 80 and all(index is not None for index in indices):
                    loaded[key] = value[indices]
                    remapped_classes += len(indices)
        if not loaded:
            raise ValueError(f"No RTMDet weights matched model state in {path}")
        self.load_state_dict(loaded, strict=False)
        return {"matched_tensors": len(loaded), "remapped_class_channels": remapped_classes,
                "source": str(path.resolve())}
