"""CenterNet Gaussian heatmap, width/height, and center-offset targets/losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from x_yolo.models.keypoint_utils import draw_gaussian, gaussian_focal_loss, gaussian_radius


class CenterNetLoss:
    def __init__(self, num_classes: int, size_loss_weight: float = 0.1,
                 offset_loss_weight: float = 1.0) -> None:
        self.num_classes = num_classes
        self.size_loss_weight = size_loss_weight
        self.offset_loss_weight = offset_loss_weight

    def build_targets(self, box_lists: list[torch.Tensor], feature_shape: tuple[int, int],
                      image_shape: tuple[int, int], device: torch.device) -> dict[str, torch.Tensor]:
        height, width = feature_shape
        del image_shape  # Boxes are normalized to the square input by the dataset.
        # Build sparse object targets on CPU from the loader's CPU labels. Doing
        # per-box coordinate math on CUDA would synchronize once per scalar.
        heatmap = torch.zeros((len(box_lists), self.num_classes, height, width))
        sizes = torch.zeros((len(box_lists), 2, height, width))
        offsets = torch.zeros_like(sizes)
        weights = torch.zeros_like(sizes)
        for batch_index, boxes in enumerate(box_lists):
            if boxes.numel() == 0:
                continue
            boxes = boxes.detach().cpu()
            class_ids = boxes[:, 0].long()
            center_x = boxes[:, 1] * width
            center_y = boxes[:, 2] * height
            box_w = boxes[:, 3] * width
            box_h = boxes[:, 4] * height
            for item in range(len(boxes)):
                x = min(int(center_x[item]), width - 1)
                y = min(int(center_y[item]), height - 1)
                radius = int(gaussian_radius(float(box_h[item]), float(box_w[item])))
                draw_gaussian(heatmap[batch_index, class_ids[item]], x, y, radius)
                sizes[batch_index, 0, y, x] = box_w[item]
                sizes[batch_index, 1, y, x] = box_h[item]
                offsets[batch_index, 0, y, x] = center_x[item] - x
                offsets[batch_index, 1, y, x] = center_y[item] - y
                weights[batch_index, :, y, x] = 1
        return {name: value.to(device, non_blocking=True) for name, value in {
            "heatmap": heatmap, "sizes": sizes, "offsets": offsets, "weights": weights,
        }.items()}

    def __call__(self, outputs: dict[str, torch.Tensor | tuple[int, int]],
                 box_lists: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = outputs["center_heatmap_logits"]
        predicted_sizes = outputs["box_sizes"]
        predicted_offsets = outputs["center_offsets"]
        if not isinstance(logits, torch.Tensor) or not isinstance(predicted_sizes, torch.Tensor):
            raise TypeError("CenterNet outputs must contain heatmap and size tensors")
        if not isinstance(predicted_offsets, torch.Tensor):
            raise TypeError("CenterNet outputs must contain center offset tensors")
        targets = self.build_targets(
            box_lists, tuple(logits.shape[-2:]), tuple(outputs["input_shape"]), logits.device
        )
        positive_count = targets["heatmap"].eq(1).sum().clamp_min(1)
        heatmap_loss = gaussian_focal_loss(logits, targets["heatmap"], positive_count)
        normalizer = positive_count * 2
        size_loss = (
            F.l1_loss(predicted_sizes.float(), targets["sizes"], reduction="none")
            * targets["weights"]
        ).sum() / normalizer
        offset_loss = (
            F.l1_loss(predicted_offsets.float(), targets["offsets"], reduction="none")
            * targets["weights"]
        ).sum() / normalizer
        total = heatmap_loss + self.size_loss_weight * size_loss + self.offset_loss_weight * offset_loss
        return {"total": total, "heatmap": heatmap_loss, "size": size_loss,
                "offset": offset_loss, "positives": positive_count.float()}
