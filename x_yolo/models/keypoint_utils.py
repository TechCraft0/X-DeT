"""Small target and checkpoint helpers shared by keypoint detectors."""
from __future__ import annotations

from math import sqrt
from typing import Sequence

import torch
import torch.nn.functional as F


_COCO_INDEX = {
    "person": 0, "bicycle": 1, "car": 2, "motorbike": 3, "aeroplane": 4,
    "bus": 5, "train": 6, "boat": 8, "bird": 14, "cat": 15, "dog": 16,
    "horse": 17, "sheep": 18, "cow": 19, "bottle": 39, "chair": 56,
    "sofa": 57, "pottedplant": 58, "diningtable": 60, "tvmonitor": 62,
}


def coco_class_indices(class_names: Sequence[str]) -> list[int]:
    """Return the COCO detector channel for each supported VOC class."""
    missing = [name for name in class_names if name not in _COCO_INDEX]
    if missing:
        raise ValueError(f"No COCO classifier channel is known for: {missing}")
    return [_COCO_INDEX[name] for name in class_names]


def gaussian_radius(height: float, width: float, min_overlap: float = 0.3) -> float:
    """CornerNet/CenterNet radius that bounds overlap for nearby peaks."""
    if height <= 0 or width <= 0 or not 0 < min_overlap < 1:
        return 0.0
    a1, b1 = 1.0, height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - sqrt(max(b1 * b1 - 4 * a1 * c1, 0.0))) / (2 * a1)

    a2, b2 = 4.0, 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    r2 = (b2 - sqrt(max(b2 * b2 - 4 * a2 * c2, 0.0))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    r3 = (b3 + sqrt(max(b3 * b3 - 4 * a3 * c3, 0.0))) / (2 * a3)
    return min(r1, r2, r3)


def draw_gaussian(heatmap: torch.Tensor, x: int, y: int, radius: int) -> None:
    """Max-compose a CenterNet Gaussian at integer feature-map position."""
    radius = max(0, int(radius))
    diameter = 2 * radius + 1
    sigma = diameter / 6.0
    coords = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=heatmap.dtype)
    gaussian_1d = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    gaussian = gaussian_1d[:, None] * gaussian_1d[None, :]

    height, width = heatmap.shape
    if not (0 <= x < width and 0 <= y < height):
        return
    left, right = min(x, radius), min(width - x - 1, radius)
    top, bottom = min(y, radius), min(height - y - 1, radius)
    target = heatmap[y - top:y + bottom + 1, x - left:x + right + 1]
    patch = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    torch.maximum(target, patch, out=target)


def gaussian_focal_loss(logits: torch.Tensor, target: torch.Tensor, avg_factor: int | torch.Tensor) -> torch.Tensor:
    """Gaussian focal loss used for center and corner heatmaps."""
    probability = logits.float().sigmoid().clamp(min=1e-4, max=1 - 1e-4)
    target = target.float()
    positive = target.eq(1)
    negative = target.lt(1)
    positive_loss = torch.log(probability) * (1 - probability).square() * positive
    negative_loss = (
        torch.log1p(-probability)
        * probability.square()
        * (1 - target).pow(4)
        * negative
    )
    normalizer = torch.as_tensor(avg_factor, dtype=logits.dtype, device=logits.device).clamp_min(1)
    return -(positive_loss.sum() + negative_loss.sum()) / normalizer


def local_maximum(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only spatial local maxima, as in the original CenterNet decoder."""
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("local-maximum kernel must be a positive odd number")
    pooled = F.max_pool2d(heatmap, kernel, stride=1, padding=(kernel - 1) // 2)
    return heatmap * pooled.eq(heatmap)


def gather_feature_map(feature: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    """Gather `[B,C,H,W]` predictions at flattened `[B,K]` positions."""
    batch, channels, height, width = feature.shape
    flattened = feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
    return flattened.gather(1, flat_indices[..., None].expand(-1, -1, channels))
