"""Small post-processing operations shared by the YOLO versions."""
from __future__ import annotations

import torch


def box_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    area_box = (box[2:] - box[:2]).clamp_min(0).prod()
    area_boxes = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (area_box + area_boxes - intersection).clamp_min(1e-9)


def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Use TorchVision NMS when available and retain a pure-PyTorch fallback."""
    try:
        from torchvision.ops import batched_nms

        return batched_nms(boxes, scores, labels, iou_threshold)
    except (ImportError, OSError, RuntimeError):
        pass

    kept: list[torch.Tensor] = []
    for class_id in labels.unique():
        class_indices = torch.where(labels == class_id)[0]
        order = class_indices[scores[class_indices].argsort(descending=True)]
        while order.numel() > 0:
            current = order[0]
            kept.append(current)
            if order.numel() == 1:
                break
            remaining = order[1:]
            overlaps = box_iou_xyxy(boxes[current], boxes[remaining])
            order = remaining[overlaps <= iou_threshold]
    if not kept:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    kept_tensor = torch.stack(kept)
    return kept_tensor[scores[kept_tensor].argsort(descending=True)]
