"""Darknet-19 based YOLOv2 detector with its passthrough feature route."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_VOC_ANCHORS = (
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
)


def _conv_bn_leaky(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1, inplace=True),
    )


class YoloV2(nn.Module):
    """Full YOLOv2 detection path, returning raw `[N,S,S,A,5+C]` logits.

    The route tensor is taken at stride 16 (26x26 for 416 input), reduced from
    512 to 64 channels and rearranged into 256 channels at stride 32. It is
    concatenated with the stride-32 detection features before prediction.
    """

    stride = 32

    def __init__(
        self,
        num_classes: int,
        anchors: tuple[tuple[float, float], ...] | list[list[float]] | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        anchor_values = DEFAULT_VOC_ANCHORS if anchors is None else anchors
        anchor_tensor = torch.as_tensor(anchor_values, dtype=torch.float32)
        if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2:
            raise ValueError("anchors must have shape [number_of_anchors, 2]")
        if anchor_tensor.shape[0] == 0 or not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
            raise ValueError("anchors must contain finite, positive width-height pairs")
        self.num_classes = int(num_classes)
        self.num_anchors = int(anchor_tensor.shape[0])
        self.register_buffer("anchors", anchor_tensor)

        pool = lambda: nn.MaxPool2d(kernel_size=2, stride=2)
        self.early = nn.Sequential(
            _conv_bn_leaky(3, 32, 3), pool(),
            _conv_bn_leaky(32, 64, 3), pool(),
            _conv_bn_leaky(64, 128, 3),
            _conv_bn_leaky(128, 64, 1),
            _conv_bn_leaky(64, 128, 3), pool(),
            _conv_bn_leaky(128, 256, 3),
            _conv_bn_leaky(256, 128, 1),
            _conv_bn_leaky(128, 256, 3), pool(),
        )
        self.route = nn.Sequential(
            _conv_bn_leaky(256, 512, 3),
            _conv_bn_leaky(512, 256, 1),
            _conv_bn_leaky(256, 512, 3),
            _conv_bn_leaky(512, 256, 1),
            _conv_bn_leaky(256, 512, 3),
        )
        self.downsample = pool()
        self.deep = nn.Sequential(
            _conv_bn_leaky(512, 1024, 3),
            _conv_bn_leaky(1024, 512, 1),
            _conv_bn_leaky(512, 1024, 3),
            _conv_bn_leaky(1024, 512, 1),
            _conv_bn_leaky(512, 1024, 3),
            _conv_bn_leaky(1024, 1024, 3),
            _conv_bn_leaky(1024, 1024, 3),
        )
        self.route_reduce = _conv_bn_leaky(512, 64, 1)
        self.fusion = _conv_bn_leaky(1024 + 64 * 4, 1024, 3)
        self.prediction = nn.Conv2d(1024, self.num_anchors * (5 + self.num_classes), 1)
        self._initialize_darknet_weights()

    def _initialize_darknet_weights(self) -> None:
        # Darknet initializes convolution filters with small zero-mean values.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected NCHW RGB input, got shape {tuple(images.shape)}")
        height, width = images.shape[-2:]
        if height % self.stride or width % self.stride:
            raise ValueError(f"YOLOv2 input height and width must be divisible by {self.stride}")

        route = self.route(self.early(images))
        deep = self.deep(self.downsample(route))
        passthrough = F.pixel_unshuffle(self.route_reduce(route), downscale_factor=2)
        if deep.shape[-2:] != passthrough.shape[-2:]:
            raise RuntimeError("Darknet-19 and passthrough feature maps must have matching spatial sizes")
        fused = self.fusion(torch.cat((passthrough, deep), dim=1))
        raw = self.prediction(fused)
        batch, _, grid_height, grid_width = raw.shape
        values_per_anchor = 5 + self.num_classes
        return raw.view(batch, self.num_anchors, values_per_anchor, grid_height, grid_width).permute(
            0, 3, 4, 1, 2
        ).contiguous()
