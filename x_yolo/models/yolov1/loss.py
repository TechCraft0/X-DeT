"""YOLOv1 grid targets and the original objective with optional GIoU loss."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class YoloTargets:
    """Grid targets for one object per cell: normalized cell offsets and image-relative size."""

    boxes: torch.Tensor  # [N, S, S, 4] -> x_cell, y_cell, width, height
    classes: torch.Tensor  # [N, S, S], -1 for empty cells
    object_mask: torch.Tensor  # [N, S, S]
    ignored_ground_truths: int


def build_targets(
    batch_boxes: Sequence[torch.Tensor],
    num_classes: int,
    grid_size: int = 7,
) -> YoloTargets:
    """Assign each cell its largest-area ground truth and count collisions.

    YOLOv1 has one conditional class distribution per cell. A deterministic
    largest-area policy makes the limitation explicit while keeping the
    original 7x7/B=2 prediction contract. Evaluation still uses every label.
    """
    if num_classes <= 0 or grid_size <= 0:
        raise ValueError("num_classes and grid_size must be positive")
    batch_size = len(batch_boxes)
    boxes = torch.zeros((batch_size, grid_size, grid_size, 4), dtype=torch.float32)
    classes = torch.full((batch_size, grid_size, grid_size), -1, dtype=torch.long)
    object_mask = torch.zeros((batch_size, grid_size, grid_size), dtype=torch.bool)
    ignored = 0

    for batch_index, image_boxes in enumerate(batch_boxes):
        if image_boxes.ndim != 2 or image_boxes.shape[-1] != 5:
            raise ValueError(f"Expected [objects,5] labels, got {tuple(image_boxes.shape)}")
        candidates: dict[tuple[int, int], list[torch.Tensor]] = {}
        for row in image_boxes:
            class_id = int(row[0].item())
            x_center, y_center, width, height = (float(value) for value in row[1:])
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"class_id {class_id} outside [0, {num_classes - 1}]")
            if not (0 <= x_center < 1 and 0 <= y_center < 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("YOLO labels must use normalized centers in [0,1) and positive normalized sizes")
            column = min(int(x_center * grid_size), grid_size - 1)
            grid_row = min(int(y_center * grid_size), grid_size - 1)
            candidates.setdefault((grid_row, column), []).append(row)

        for (grid_row, column), objects in candidates.items():
            if len(objects) > 1:
                ignored += len(objects) - 1
            # max() preserves the first row when areas tie, making source order the tie-breaker.
            selected = max(objects, key=lambda item: float(item[3] * item[4]))
            class_id = int(selected[0].item())
            x_center, y_center, width, height = (float(value) for value in selected[1:])
            boxes[batch_index, grid_row, column] = torch.tensor(
                [x_center * grid_size - column, y_center * grid_size - grid_row, width, height]
            )
            classes[batch_index, grid_row, column] = class_id
            object_mask[batch_index, grid_row, column] = True

    return YoloTargets(boxes, classes, object_mask, ignored)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half = boxes[..., 2:4] / 2
    return torch.cat((boxes[..., :2] - half, boxes[..., :2] + half), dim=-1)


def _pairwise_iou_and_giou(predicted: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return IoU and GIoU for corresponding normalized xywh boxes."""
    pred_xyxy = _xywh_to_xyxy(predicted)
    target_xyxy = _xywh_to_xyxy(target)
    top_left = torch.maximum(pred_xyxy[..., :2], target_xyxy[..., :2])
    bottom_right = torch.minimum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    pred_area = predicted[..., 2:].clamp_min(0).prod(dim=-1)
    target_area = target[..., 2:].clamp_min(0).prod(dim=-1)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp_min(1e-9)

    enclosing_top_left = torch.minimum(pred_xyxy[..., :2], target_xyxy[..., :2])
    enclosing_bottom_right = torch.maximum(pred_xyxy[..., 2:], target_xyxy[..., 2:])
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).clamp_min(0).prod(dim=-1)
    giou = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-9)
    return iou, giou


class YoloV1Loss(nn.Module):
    """Paper loss, optionally augmented with GIoU for tighter box regression."""

    def __init__(
        self,
        num_classes: int,
        grid_size: int = 7,
        boxes_per_cell: int = 2,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_giou: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_giou = lambda_giou

    def forward(self, predictions: torch.Tensor, targets: YoloTargets) -> dict[str, torch.Tensor]:
        batch_size, grid_h, grid_w, channels = predictions.shape
        expected_channels = self.boxes_per_cell * 5 + self.num_classes
        if (grid_h, grid_w, channels) != (self.grid_size, self.grid_size, expected_channels):
            raise ValueError(
                f"Expected [N,{self.grid_size},{self.grid_size},{expected_channels}], got {tuple(predictions.shape)}"
            )
        if targets.boxes.shape[:3] != (batch_size, self.grid_size, self.grid_size):
            raise ValueError("Prediction and target batch/grid shapes do not match")
        device = predictions.device
        target_boxes = targets.boxes.to(device)
        object_mask = targets.object_mask.to(device)
        classes = targets.classes.to(device)

        predicted_boxes = predictions[..., : self.boxes_per_cell * 5].view(
            batch_size, self.grid_size, self.grid_size, self.boxes_per_cell, 5
        )
        pred_xy = predicted_boxes[..., :2]
        pred_sqrt_wh = predicted_boxes[..., 2:4]
        pred_confidence = predicted_boxes[..., 4]

        rows = torch.arange(self.grid_size, device=device, dtype=predictions.dtype).view(1, self.grid_size, 1, 1)
        cols = torch.arange(self.grid_size, device=device, dtype=predictions.dtype).view(1, 1, self.grid_size, 1)
        pred_center = torch.stack(
            ((cols + pred_xy[..., 0]) / self.grid_size, (rows + pred_xy[..., 1]) / self.grid_size), dim=-1
        )
        pred_wh = pred_sqrt_wh.clamp_min(0).square()
        pred_xywh = torch.cat((pred_center, pred_wh), dim=-1)
        target_xy = target_boxes[..., :2]
        target_center = torch.stack(
            ((cols[..., 0] + target_xy[..., 0]) / self.grid_size, (rows[..., 0] + target_xy[..., 1]) / self.grid_size), dim=-1
        )
        target_xywh = torch.cat((target_center, target_boxes[..., 2:4]), dim=-1).unsqueeze(-2)
        ious, generalized_ious = _pairwise_iou_and_giou(pred_xywh, target_xywh.expand_as(pred_xywh))

        responsible_index = ious.detach().argmax(dim=-1)
        responsible = F.one_hot(responsible_index, self.boxes_per_cell).bool() & object_mask.unsqueeze(-1)
        not_responsible = ~responsible

        coord_mask = responsible.unsqueeze(-1)
        xy_error = (pred_xy - target_xy.unsqueeze(-2)).square().sum(dim=-1, keepdim=True)
        sqrt_target_wh = target_boxes[..., 2:4].clamp_min(0).sqrt()
        wh_error = (pred_sqrt_wh - sqrt_target_wh.unsqueeze(-2)).square().sum(dim=-1, keepdim=True)
        coordinate = self.lambda_coord * ((xy_error + wh_error) * coord_mask).sum()
        giou = self.lambda_giou * ((1.0 - generalized_ious) * responsible).sum()

        object_confidence = ((pred_confidence - ious.detach()).square() * responsible).sum()
        no_object_confidence = self.lambda_noobj * (pred_confidence.square() * not_responsible).sum()

        class_predictions = predictions[..., self.boxes_per_cell * 5 :]
        safe_classes = classes.clamp_min(0)
        class_targets = F.one_hot(safe_classes, self.num_classes).to(predictions.dtype)
        class_error = (class_predictions - class_targets).square().sum(dim=-1)
        classification = (class_error * object_mask).sum()

        scale = max(batch_size, 1)
        components = {
            "coordinate": coordinate / scale,
            "giou": giou / scale,
            "object_confidence": object_confidence / scale,
            "no_object_confidence": no_object_confidence / scale,
            "classification": classification / scale,
        }
        components["total"] = sum(components.values())
        return components
