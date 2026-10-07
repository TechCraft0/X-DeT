"""YOLOv3's nine-anchor assignment and objectness/classification losses."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import DEFAULT_ANCHOR_MASKS, DEFAULT_VOC_ANCHORS


@dataclass
class YoloV3Targets:
    """Per-scale targets plus normalized ground truth used by ignore masks."""

    boxes: list[torch.Tensor]  # [N,gh,gw,3,4]: cell offsets and log anchor ratios
    classes: list[torch.Tensor]  # [N,gh,gw,3], -1 where no object is assigned
    object_masks: list[torch.Tensor]  # [N,gh,gw,3]
    box_scales: list[torch.Tensor]  # [N,gh,gw,3], Darknet's (2 - box area) weight
    all_boxes: list[torch.Tensor]  # normalized [cx,cy,w,h], per image
    ignored_ground_truths: int


def _dimension_iou(box_wh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    intersection = torch.minimum(box_wh.unsqueeze(0), anchors).prod(dim=-1)
    union = box_wh.prod() + anchors.prod(dim=-1) - intersection
    return intersection / union.clamp_min(1e-9)


def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half_size = boxes[..., 2:4] / 2
    return torch.cat((boxes[..., :2] - half_size, boxes[..., :2] + half_size), dim=-1)


def _pairwise_iou_xywh(boxes: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0 or targets.numel() == 0:
        return boxes.new_zeros((boxes.shape[0], targets.shape[0]))
    box_corners, target_corners = _xywh_to_xyxy(boxes), _xywh_to_xyxy(targets)
    top_left = torch.maximum(box_corners[:, None, :2], target_corners[None, :, :2])
    bottom_right = torch.minimum(box_corners[:, None, 2:], target_corners[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    box_area = boxes[:, 2:].clamp_min(0).prod(dim=-1, keepdim=True)
    target_area = targets[:, 2:].clamp_min(0).prod(dim=-1).unsqueeze(0)
    return intersection / (box_area + target_area - intersection).clamp_min(1e-9)


def build_targets(
    batch_boxes: Sequence[torch.Tensor],
    num_classes: int,
    grid_sizes: Sequence[tuple[int, int]],
    anchors: torch.Tensor | Sequence[Sequence[float]] = DEFAULT_VOC_ANCHORS,
    anchor_masks: Sequence[Sequence[int]] = DEFAULT_ANCHOR_MASKS,
) -> YoloV3Targets:
    """Assign each GT to its best-IoU anchor across all three detection scales."""
    anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32).detach().cpu()
    masks = tuple(tuple(int(index) for index in mask) for mask in anchor_masks)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if anchor_tensor.shape != (9, 2) or not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
        raise ValueError("anchors must be nine finite, positive width-height pairs")
    if len(grid_sizes) != 3 or len(masks) != 3 or any(len(mask) != 3 for mask in masks):
        raise ValueError("YOLOv3 needs three output grids and three groups of three anchors")
    if sorted(index for mask in masks for index in mask) != list(range(9)):
        raise ValueError("anchor_masks must use every anchor index exactly once")
    grids = [(int(height), int(width)) for height, width in grid_sizes]
    input_height, input_width = grids[0][0] * 32, grids[0][1] * 32
    for grid, stride in zip(grids, (32, 16, 8)):
        if grid[0] * stride != input_height or grid[1] * stride != input_width:
            raise ValueError("YOLOv3 output grids must correspond to strides 32, 16 and 8")

    batch_size = len(batch_boxes)
    boxes_by_scale = [torch.zeros((batch_size, height, width, 3, 4)) for height, width in grids]
    classes_by_scale = [
        torch.full((batch_size, height, width, 3), -1, dtype=torch.long) for height, width in grids
    ]
    masks_by_scale = [torch.zeros((batch_size, height, width, 3), dtype=torch.bool) for height, width in grids]
    scales_by_scale = [torch.zeros((batch_size, height, width, 3)) for height, width in grids]
    quality_by_scale = [torch.full((batch_size, height, width, 3), -1.0) for height, width in grids]
    all_boxes: list[torch.Tensor] = []
    ignored = 0

    for batch_index, image_boxes in enumerate(batch_boxes):
        if image_boxes.ndim != 2 or image_boxes.shape[-1] != 5:
            raise ValueError(f"Expected [objects,5] labels, got {tuple(image_boxes.shape)}")
        normalized_boxes = image_boxes[:, 1:5].detach().float().cpu().clone()
        all_boxes.append(normalized_boxes)
        for row in image_boxes:
            class_id = int(row[0].item())
            center_x, center_y, width, height = (float(value) for value in row[1:])
            if class_id < 0 or class_id >= num_classes:
                raise ValueError(f"class_id {class_id} outside [0, {num_classes - 1}]")
            if not (0 <= center_x < 1 and 0 <= center_y < 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("YOLO labels need centers in [0,1) and positive normalized sizes")

            box_wh_pixels = torch.tensor([width * input_width, height * input_height])
            anchor_quality = _dimension_iou(box_wh_pixels, anchor_tensor)
            anchor_index = int(anchor_quality.argmax().item())
            quality = float(anchor_quality[anchor_index])
            scale_index = next(index for index, mask in enumerate(masks) if anchor_index in mask)
            local_anchor_index = masks[scale_index].index(anchor_index)
            grid_height, grid_width = grids[scale_index]
            column = min(int(center_x * grid_width), grid_width - 1)
            grid_row = min(int(center_y * grid_height), grid_height - 1)
            target_mask = masks_by_scale[scale_index]
            target_quality = quality_by_scale[scale_index]
            if target_mask[batch_index, grid_row, column, local_anchor_index]:
                ignored += 1
                if quality <= float(target_quality[batch_index, grid_row, column, local_anchor_index]):
                    continue

            anchor_width, anchor_height = anchor_tensor[anchor_index]
            boxes_by_scale[scale_index][batch_index, grid_row, column, local_anchor_index] = torch.tensor(
                [
                    center_x * grid_width - column,
                    center_y * grid_height - grid_row,
                    torch.log(box_wh_pixels[0] / anchor_width),
                    torch.log(box_wh_pixels[1] / anchor_height),
                ]
            )
            classes_by_scale[scale_index][batch_index, grid_row, column, local_anchor_index] = class_id
            target_mask[batch_index, grid_row, column, local_anchor_index] = True
            scales_by_scale[scale_index][batch_index, grid_row, column, local_anchor_index] = 2.0 - width * height
            target_quality[batch_index, grid_row, column, local_anchor_index] = quality

    return YoloV3Targets(
        boxes_by_scale,
        classes_by_scale,
        masks_by_scale,
        scales_by_scale,
        all_boxes,
        ignored,
    )


class YoloV3Loss(nn.Module):
    """Darknet-style coordinate, objectness and independent class objectives."""

    def __init__(
        self,
        num_classes: int,
        anchors: torch.Tensor | Sequence[Sequence[float]] = DEFAULT_VOC_ANCHORS,
        anchor_masks: Sequence[Sequence[int]] = DEFAULT_ANCHOR_MASKS,
        coord_scale: float = 1.0,
        object_scale: float = 1.0,
        noobject_scale: float = 1.0,
        class_scale: float = 1.0,
        ignore_iou_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        anchor_tensor = torch.as_tensor(anchors, dtype=torch.float32)
        masks = tuple(tuple(int(index) for index in mask) for mask in anchor_masks)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if anchor_tensor.shape != (9, 2) or not torch.isfinite(anchor_tensor).all() or (anchor_tensor <= 0).any():
            raise ValueError("anchors must be nine finite, positive width-height pairs")
        if len(masks) != 3 or any(len(mask) != 3 for mask in masks):
            raise ValueError("anchor_masks must contain three groups of three indices")
        if sorted(index for mask in masks for index in mask) != list(range(9)):
            raise ValueError("anchor_masks must use every anchor index exactly once")
        if not 0 <= ignore_iou_threshold <= 1:
            raise ValueError("ignore_iou_threshold must be in [0,1]")
        if min(coord_scale, object_scale, noobject_scale, class_scale) < 0:
            raise ValueError("loss scales must be non-negative")
        self.num_classes = int(num_classes)
        self.anchor_masks = masks
        self.register_buffer("anchors", anchor_tensor)
        self.coord_scale = float(coord_scale)
        self.object_scale = float(object_scale)
        self.noobject_scale = float(noobject_scale)
        self.class_scale = float(class_scale)
        self.ignore_iou_threshold = float(ignore_iou_threshold)

    def forward(self, predictions: Sequence[torch.Tensor], targets: YoloV3Targets) -> dict[str, torch.Tensor]:
        if len(predictions) != 3 or len(targets.boxes) != 3:
            raise ValueError("YOLOv3 loss expects three prediction and target scales")
        device = predictions[0].device
        dtype = predictions[0].dtype
        batch = predictions[0].shape[0]
        input_height, input_width = predictions[0].shape[1] * 32, predictions[0].shape[2] * 32
        components = {
            "coordinate": predictions[0].new_zeros(()),
            "object_confidence": predictions[0].new_zeros(()),
            "no_object_confidence": predictions[0].new_zeros(()),
            "classification": predictions[0].new_zeros(()),
        }

        for scale_index, (prediction, stride, anchor_mask) in enumerate(
            zip(predictions, (32, 16, 8), self.anchor_masks)
        ):
            if prediction.ndim != 5:
                raise ValueError("Each YOLOv3 output must have shape [N,gh,gw,3,5+C]")
            batch_size, grid_height, grid_width, num_anchors, channels = prediction.shape
            if (
                batch_size != batch
                or num_anchors != 3
                or channels != 5 + self.num_classes
                or grid_height * stride != input_height
                or grid_width * stride != input_width
            ):
                raise ValueError(f"Invalid YOLOv3 prediction shape at stride {stride}: {tuple(prediction.shape)}")
            expected_shape = (batch, grid_height, grid_width, 3)
            if targets.object_masks[scale_index].shape != expected_shape:
                raise ValueError("Prediction and target grids do not match")

            target_boxes = targets.boxes[scale_index].to(device=device, dtype=dtype)
            classes = targets.classes[scale_index].to(device=device)
            object_mask = targets.object_masks[scale_index].to(device=device)
            box_scales = targets.box_scales[scale_index].to(device=device, dtype=dtype)
            anchors = self.anchors[list(anchor_mask)].to(device=device, dtype=dtype)
            anchor_grid = anchors.view(1, 1, 1, 3, 2)
            rows = torch.arange(grid_height, device=device, dtype=dtype).view(1, grid_height, 1, 1)
            columns = torch.arange(grid_width, device=device, dtype=dtype).view(1, 1, grid_width, 1)

            xy_loss = F.binary_cross_entropy_with_logits(
                prediction[..., :2], target_boxes[..., :2], reduction="none"
            ).sum(dim=-1)
            wh_loss = (prediction[..., 2:4] - target_boxes[..., 2:4]).square().sum(dim=-1)
            coordinate_values = (xy_loss + wh_loss) * box_scales * object_mask

            centers = torch.stack(
                ((columns + prediction[..., 0].sigmoid()) / grid_width,
                 (rows + prediction[..., 1].sigmoid()) / grid_height),
                dim=-1,
            )
            sizes = prediction[..., 2:4].clamp(-10, 10).exp() * anchor_grid / prediction.new_tensor(
                [input_width, input_height]
            )
            predicted_boxes = torch.cat((centers, sizes), dim=-1)
            negative_mask = ~object_mask
            for batch_index, ground_truth in enumerate(targets.all_boxes):
                if ground_truth.numel() == 0:
                    continue
                overlaps = _pairwise_iou_xywh(
                    predicted_boxes[batch_index].detach().reshape(-1, 4),
                    ground_truth.to(device=device, dtype=dtype),
                )
                best_overlap = overlaps.max(dim=1).values.view(grid_height, grid_width, 3)
                negative_mask[batch_index] &= best_overlap <= self.ignore_iou_threshold

            objectness_logits = prediction[..., 4]
            object_values = F.binary_cross_entropy_with_logits(
                objectness_logits, torch.ones_like(objectness_logits), reduction="none"
            )
            no_object_values = F.binary_cross_entropy_with_logits(
                objectness_logits, torch.zeros_like(objectness_logits), reduction="none"
            )
            class_targets = F.one_hot(classes.clamp_min(0), self.num_classes).to(dtype=dtype)
            class_values = F.binary_cross_entropy_with_logits(
                prediction[..., 5:], class_targets, reduction="none"
            ).sum(dim=-1)

            components["coordinate"] += self.coord_scale * coordinate_values.sum() / max(batch, 1)
            components["object_confidence"] += (
                self.object_scale * (object_values * object_mask).sum() / max(batch, 1)
            )
            components["no_object_confidence"] += (
                self.noobject_scale * (no_object_values * negative_mask).sum() / max(batch, 1)
            )
            components["classification"] += (
                self.class_scale * (class_values * object_mask).sum() / max(batch, 1)
            )

        components["total"] = sum(components.values())
        return components
