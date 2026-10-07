"""Decode the three YOLOv3 heads into xyxy pixel boxes and apply class-aware NMS."""
from __future__ import annotations

from collections.abc import Sequence

import torch

from x_yolo.models.detection_utils import class_aware_nms
from .model import DEFAULT_ANCHOR_MASKS, DEFAULT_VOC_ANCHORS


def decode_predictions(
    outputs: Sequence[torch.Tensor],
    original_sizes: list[tuple[int, int]],
    anchors: torch.Tensor | Sequence[Sequence[float]] = DEFAULT_VOC_ANCHORS,
    anchor_masks: Sequence[Sequence[int]] = DEFAULT_ANCHOR_MASKS,
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Decode raw heads ordered coarse-to-fine (strides 32, 16, 8)."""
    if len(outputs) != 3 or any(output.ndim != 5 for output in outputs):
        raise ValueError("Expected three [N,gh,gw,3,5+C] prediction tensors")
    batch_size = outputs[0].shape[0]
    if len(original_sizes) != batch_size:
        raise ValueError("Expected one original image size for each batch item")
    if max_detections < 1:
        raise ValueError("max_detections must be at least 1")
    if not 0 <= confidence_threshold <= 1 or not 0 <= nms_iou_threshold <= 1:
        raise ValueError("confidence and NMS thresholds must be in [0,1]")
    anchor_tensor = torch.as_tensor(anchors, dtype=outputs[0].dtype, device=outputs[0].device)
    masks = tuple(tuple(int(index) for index in mask) for mask in anchor_masks)
    if anchor_tensor.shape != (9, 2) or not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
        raise ValueError("anchors must be nine finite, positive width-height pairs")
    if len(masks) != 3 or any(len(mask) != 3 for mask in masks):
        raise ValueError("anchor_masks must contain three groups of three indices")
    if sorted(index for mask in masks for index in mask) != list(range(9)):
        raise ValueError("anchor_masks must use every anchor index exactly once")

    input_height, input_width = outputs[0].shape[1] * 32, outputs[0].shape[2] * 32
    decoded_boxes, decoded_scores = [], []
    for output, stride, anchor_mask in zip(outputs, (32, 16, 8), masks):
        batch, grid_height, grid_width, num_anchors, channels = output.shape
        if (
            batch != batch_size
            or num_anchors != 3
            or channels < 6
            or grid_height * stride != input_height
            or grid_width * stride != input_width
        ):
            raise ValueError(f"Invalid YOLOv3 prediction shape at stride {stride}: {tuple(output.shape)}")
        anchors_for_scale = anchor_tensor[list(anchor_mask)].view(1, 1, 1, 3, 2)
        rows = torch.arange(grid_height, device=output.device, dtype=output.dtype).view(1, grid_height, 1, 1)
        columns = torch.arange(grid_width, device=output.device, dtype=output.dtype).view(1, 1, grid_width, 1)
        centers = torch.stack(
            ((columns + output[..., 0].sigmoid()) / grid_width,
             (rows + output[..., 1].sigmoid()) / grid_height),
            dim=-1,
        )
        sizes = output[..., 2:4].clamp(-10, 10).exp() * anchors_for_scale / output.new_tensor(
            [input_width, input_height]
        )
        center_size = torch.cat((centers, sizes), dim=-1)
        half = center_size[..., 2:4] / 2
        boxes = torch.cat((center_size[..., :2] - half, center_size[..., :2] + half), dim=-1)
        boxes[..., 0::2].clamp_(0, 1)
        boxes[..., 1::2].clamp_(0, 1)
        class_scores = output[..., 5:].sigmoid() * output[..., 4].sigmoid().unsqueeze(-1)
        decoded_boxes.append(boxes.reshape(batch_size, -1, 4))
        decoded_scores.append(class_scores.reshape(batch_size, -1, channels - 5))

    boxes = torch.cat(decoded_boxes, dim=1)
    scores = torch.cat(decoded_scores, dim=1)
    results: list[dict[str, torch.Tensor]] = []
    for batch_index, (image_width, image_height) in enumerate(original_sizes):
        pixel_boxes = boxes[batch_index] * boxes.new_tensor(
            [image_width, image_height, image_width, image_height]
        )
        class_scores = scores[batch_index]
        candidate_boxes, candidate_scores, candidate_labels = [], [], []
        for class_id in range(class_scores.shape[-1]):
            selected = torch.where(class_scores[:, class_id] >= confidence_threshold)[0]
            if selected.numel() == 0:
                continue
            candidate_boxes.append(pixel_boxes[selected])
            candidate_scores.append(class_scores[selected, class_id])
            candidate_labels.append(torch.full_like(selected, class_id))
        if not candidate_boxes:
            results.append({
                "boxes": boxes.new_zeros((0, 4)),
                "scores": boxes.new_zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long, device=boxes.device),
            })
            continue
        all_boxes = torch.cat(candidate_boxes)
        all_scores = torch.cat(candidate_scores)
        all_labels = torch.cat(candidate_labels)
        keep = class_aware_nms(all_boxes, all_scores, all_labels, nms_iou_threshold)[:max_detections]
        results.append({"boxes": all_boxes[keep], "scores": all_scores[keep], "labels": all_labels[keep]})
    return results
