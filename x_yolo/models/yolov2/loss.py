"""YOLOv2 anchor assignment and the Darknet-style squared-error objective."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class YoloV2Targets:
    """One anchor target per occupied cell, plus all GT boxes for ignore masking."""

    boxes: torch.Tensor  # [N,S,S,A,4]: cell offsets x/y and log width/height ratios
    classes: torch.Tensor  # [N,S,S,A], -1 for negative anchors
    object_mask: torch.Tensor  # [N,S,S,A]
    all_boxes: list[torch.Tensor]  # per-image normalized [cx,cy,w,h], including collisions
    ignored_ground_truths: int


def _dimension_iou(box_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    intersection = torch.minimum(box_wh.unsqueeze(0), anchors).prod(dim=-1)
    box_area = box_wh.prod()
    anchor_area = anchors.prod(dim=-1)
    return intersection / (box_area + anchor_area - intersection).clamp_min(1e-9)


def build_targets(
    batch_boxes: Sequence[torch.Tensor],
    num_classes: int,
    anchors: torch.Tensor | Sequence[Sequence[float]],
    grid_size: int,
) -> YoloV2Targets:
    """Assign each GT to its best-IoU anchor in the cell containing its center.

    If multiple GT boxes collide on the same cell and anchor, keep the one with
    the better anchor-shape IoU and count the others. All GTs remain available
    to the loss's ignore mask, so they do not become false background targets.
    """
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32).detach().cpu()
    if num_classes <= 0 or grid_size <= 0:
        raise ValueError("num_classes and grid_size must be positive")
    if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2 or anchor_tensor.shape[0] == 0:
        raise ValueError("anchors must have shape [number_of_anchors, 2]")
    if not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
        raise ValueError("anchors must contain finite, positive width-height pairs")

    batch_size, num_anchors = len(batch_boxes), anchor_tensor.shape[0]
    target_boxes = torch.zeros((batch_size, grid_size, grid_size, num_anchors, 4), dtype=torch.float32)
    target_classes = torch.full((batch_size, grid_size, grid_size, num_anchors), -1, dtype=torch.long)
    object_mask = torch.zeros((batch_size, grid_size, grid_size, num_anchors), dtype=torch.bool)
    anchor_quality = torch.full_like(object_mask, -1, dtype=torch.float32)
    all_boxes: list[torch.Tensor] = []
    ignored = 0

    for batch_index, image_boxes in enumerate(batch_boxes):
        if image_boxes.ndim != 2 or image_boxes.shape[-1] != 5:
            raise ValueError(f"Expected [objects,5] labels, got {tuple(image_boxes.shape)}")
        all_boxes.append(image_boxes[:, 1:5].detach().float().cpu().clone())
        for row in image_boxes:
            class_id = int(row[0].item())
            x_center, y_center, width, height = (float(value) for value in row[1:])
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"class_id {class_id} outside [0, {num_classes - 1}]")
            if not (0 <= x_center < 1 and 0 <= y_center < 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("YOLO labels need centers in [0,1) and positive normalized sizes")

            column = min(int(x_center * grid_size), grid_size - 1)
            grid_row = min(int(y_center * grid_size), grid_size - 1)
            box_wh_in_cells = torch.tensor([width * grid_size, height * grid_size])
            qualities = _dimension_iou(box_wh_in_cells, anchor_tensor)
            anchor_index = int(qualities.argmax().item())
            quality = float(qualities[anchor_index])
            if object_mask[batch_index, grid_row, column, anchor_index]:
                ignored += 1
                if quality <= float(anchor_quality[batch_index, grid_row, column, anchor_index]):
                    continue

            target_boxes[batch_index, grid_row, column, anchor_index] = torch.tensor(
                [
                    x_center * grid_size - column,
                    y_center * grid_size - grid_row,
                    torch.log(box_wh_in_cells[0] / anchor_tensor[anchor_index, 0]),
                    torch.log(box_wh_in_cells[1] / anchor_tensor[anchor_index, 1]),
                ]
            )
            target_classes[batch_index, grid_row, column, anchor_index] = class_id
            object_mask[batch_index, grid_row, column, anchor_index] = True
            anchor_quality[batch_index, grid_row, column, anchor_index] = quality

    return YoloV2Targets(target_boxes, target_classes, object_mask, all_boxes, ignored)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half = boxes[..., 2:4] / 2
    return torch.cat((boxes[..., :2] - half, boxes[..., :2] + half), dim=-1)


def _pairwise_iou_xywh(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for normalized xywh rows [M,4] and [K,4]."""
    if boxes.numel() == 0 or targets.numel() == 0:
        return boxes.new_zeros((boxes.shape[0], targets.shape[0]))
    box_xyxy, target_xyxy = _xywh_to_xyxy(boxes), _xywh_to_xyxy(targets)
    top_left = torch.maximum(box_xyxy[:, None, :2], target_xyxy[None, :, :2])
    bottom_right = torch.minimum(box_xyxy[:, None, 2:], target_xyxy[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    box_area = boxes[:, 2:].clamp_min(0).prod(dim=-1, keepdim=True)
    target_area = targets[:, 2:].clamp_min(0).prod(dim=-1).unsqueeze(0)
    return intersection / (box_area + target_area - intersection).clamp_min(1e-9)


def _aligned_iou_xywh(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """IoU for corresponding xywh boxes with identical leading dimensions."""
    box_xyxy, target_xyxy = _xywh_to_xyxy(boxes), _xywh_to_xyxy(targets)
    top_left = torch.maximum(box_xyxy[..., :2], target_xyxy[..., :2])
    bottom_right = torch.minimum(box_xyxy[..., 2:], target_xyxy[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    box_area = boxes[..., 2:].clamp_min(0).prod(dim=-1)
    target_area = targets[..., 2:].clamp_min(0).prod(dim=-1)
    return intersection / (box_area + target_area - intersection).clamp_min(1e-9)


class YoloV2Loss(nn.Module):
    """Anchor-box losses with IoU-rescored objectness and an IoU ignore rule."""

    def __init__(
        self,
        num_classes: int,
        anchors: torch.Tensor | Sequence[Sequence[float]],
        coord_scale: float = 1.0,
        object_scale: float = 5.0,
        noobject_scale: float = 1.0,
        class_scale: float = 1.0,
        ignore_iou_threshold: float = 0.6,
    ) -> None:
        super().__init__()
        anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if anchor_tensor.ndim != 2 or anchor_tensor.shape[1] != 2 or anchor_tensor.shape[0] == 0:
            raise ValueError("anchors must have shape [number_of_anchors, 2]")
        if not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
            raise ValueError("anchors must contain finite, positive width-height pairs")
        if not 0 <= ignore_iou_threshold <= 1:
            raise ValueError("ignore_iou_threshold must be in [0,1]")
        if min(coord_scale, object_scale, noobject_scale, class_scale) < 0:
            raise ValueError("loss scales must be non-negative")
        self.num_classes = int(num_classes)
        self.register_buffer("anchors", anchor_tensor)
        self.coord_scale = float(coord_scale)
        self.object_scale = float(object_scale)
        self.noobject_scale = float(noobject_scale)
        self.class_scale = float(class_scale)
        self.ignore_iou_threshold = float(ignore_iou_threshold)

    def forward(self, predictions: torch.Tensor, targets: YoloV2Targets) -> dict[str, torch.Tensor]:
        if predictions.ndim != 5:
            raise ValueError("YOLOv2 predictions must have shape [N,S,S,A,5+C]")
        batch, grid_h, grid_w, num_anchors, channels = predictions.shape
        expected_channels = 5 + self.num_classes
        if grid_h != grid_w or num_anchors != self.anchors.shape[0] or channels != expected_channels:
            raise ValueError(
                f"Expected [N,S,S,{self.anchors.shape[0]},{expected_channels}], got {tuple(predictions.shape)}"
            )
        expected_target_shape = (batch, grid_h, grid_w, num_anchors)
        if targets.object_mask.shape != expected_target_shape or targets.boxes.shape != (*expected_target_shape, 4):
            raise ValueError("Prediction and target batch/grid/anchor shapes do not match")

        device, dtype = predictions.device, predictions.dtype
        target_boxes = targets.boxes.to(device=device, dtype=dtype)
        object_mask = targets.object_mask.to(device=device)
        classes = targets.classes.to(device=device)
        anchors = self.anchors.to(device=device, dtype=dtype).view(1, 1, 1, num_anchors, 2)
        rows = torch.arange(grid_h, device=device, dtype=dtype).view(1, grid_h, 1, 1)
        columns = torch.arange(grid_w, device=device, dtype=dtype).view(1, 1, grid_w, 1)

        xy_prediction = predictions[..., :2].sigmoid()
        wh_logits = predictions[..., 2:4]
        objectness = predictions[..., 4].sigmoid()
        predicted_center = torch.stack(
            ((columns + xy_prediction[..., 0]) / grid_w, (rows + xy_prediction[..., 1]) / grid_h), dim=-1
        )
        predicted_wh = wh_logits.clamp(-10, 10).exp() * anchors / grid_w
        predicted_xywh = torch.cat((predicted_center, predicted_wh), dim=-1)

        target_center = torch.stack(
            ((columns + target_boxes[..., 0]) / grid_w, (rows + target_boxes[..., 1]) / grid_h), dim=-1
        )
        target_wh = target_boxes[..., 2:4].exp() * anchors / grid_w
        target_xywh = torch.cat((target_center, target_wh), dim=-1)
        ious = _aligned_iou_xywh(predicted_xywh, target_xywh)

        coordinate_values = (xy_prediction - target_boxes[..., :2]).square().sum(dim=-1)
        coordinate_values += (wh_logits - target_boxes[..., 2:4]).square().sum(dim=-1)
        coordinate = self.coord_scale * (coordinate_values * object_mask).sum()
        object_confidence = self.object_scale * (
            (objectness - ious.detach()).square() * object_mask
        ).sum()

        # Darknet ignores a negative anchor if it already overlaps any GT well.
        negative_mask = ~object_mask
        for batch_index, ground_truth in enumerate(targets.all_boxes):
            if ground_truth.numel() == 0:
                continue
            overlaps = _pairwise_iou_xywh(
                predicted_xywh[batch_index].detach().reshape(-1, 4),
                ground_truth.to(device=device, dtype=dtype),
            )
            best_overlap = overlaps.max(dim=1).values.view(grid_h, grid_w, num_anchors)
            negative_mask[batch_index] &= best_overlap <= self.ignore_iou_threshold
        no_object_confidence = self.noobject_scale * (objectness.square() * negative_mask).sum()

        class_probabilities = F.softmax(predictions[..., 5:], dim=-1)
        class_targets = F.one_hot(classes.clamp_min(0), self.num_classes).to(dtype=dtype)
        class_values = (class_probabilities - class_targets).square().sum(dim=-1)
        classification = self.class_scale * (class_values * object_mask).sum()

        scale = max(batch, 1)
        components = {
            "coordinate": coordinate / scale,
            "object_confidence": object_confidence / scale,
            "no_object_confidence": no_object_confidence / scale,
            "classification": classification / scale,
        }
        components["total"] = sum(components.values())
        return components
