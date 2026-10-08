"""CenterNet heatmap peak selection and center/size box decoding."""
from __future__ import annotations

import torch

from x_yolo.models.keypoint_utils import gather_feature_map, local_maximum


@torch.inference_mode()
def decode_predictions(outputs: dict[str, torch.Tensor | tuple[int, int]],
                       original_sizes: list[tuple[int, int]], *,
                       confidence_threshold: float = 0.001,
                       nms_iou_threshold: float = 0.5,
                       max_detections: int = 100,
                       topk: int = 100,
                       local_maximum_kernel: int = 3) -> list[dict[str, torch.Tensor]]:
    """Decode top center peaks. Local maxima are CenterNet's primary duplicate filter."""
    del nms_iou_threshold  # The original CenterNet decoder uses heatmap NMS and top-k, not box NMS.
    logits = outputs["center_heatmap_logits"]
    sizes = outputs["box_sizes"]
    offsets = outputs["center_offsets"]
    input_shape = outputs["input_shape"]
    if not all(isinstance(value, torch.Tensor) for value in (logits, sizes, offsets)):
        raise TypeError("CenterNet outputs are missing prediction tensors")
    input_h, input_w = input_shape
    batch, classes, feat_h, feat_w = logits.shape
    heatmap = local_maximum(logits.sigmoid(), local_maximum_kernel).reshape(batch, -1)
    count = min(topk, max_detections, heatmap.shape[1])
    scores, flattened = torch.topk(heatmap, count, dim=1)
    labels = flattened // (feat_h * feat_w)
    spatial = flattened % (feat_h * feat_w)
    centers = gather_feature_map(offsets, spatial)
    box_sizes = gather_feature_map(sizes, spatial).clamp_min(0)
    x = (spatial % feat_w).to(logits.dtype) + centers[..., 0]
    y = (spatial // feat_w).to(logits.dtype) + centers[..., 1]
    x1 = (x - box_sizes[..., 0] / 2) * (input_w / feat_w)
    y1 = (y - box_sizes[..., 1] / 2) * (input_h / feat_h)
    x2 = (x + box_sizes[..., 0] / 2) * (input_w / feat_w)
    y2 = (y + box_sizes[..., 1] / 2) * (input_h / feat_h)
    detections = []
    for index, (original_w, original_h) in enumerate(original_sizes):
        boxes = torch.stack((x1[index], y1[index], x2[index], y2[index]), dim=1)
        boxes[:, 0] = boxes[:, 0].clamp(0, input_w) * (original_w / input_w)
        boxes[:, 2] = boxes[:, 2].clamp(0, input_w) * (original_w / input_w)
        boxes[:, 1] = boxes[:, 1].clamp(0, input_h) * (original_h / input_h)
        boxes[:, 3] = boxes[:, 3].clamp(0, input_h) * (original_h / input_h)
        keep = scores[index] >= confidence_threshold
        detections.append({"boxes": boxes[keep], "scores": scores[index][keep], "labels": labels[index][keep]})
    return detections
