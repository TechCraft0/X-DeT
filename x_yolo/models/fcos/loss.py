"""FCOS point assignment and the paper's focal, IoU, and centerness losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _points(feature_maps: list[torch.Tensor], strides: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    points, point_strides = [], []
    for feature, stride in zip(feature_maps, strides):
        height, width = feature.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=torch.float32) + 0.5,
            torch.arange(width, device=feature.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        points.append(torch.stack((x, y), dim=-1).reshape(-1, 2) * stride)
        point_strides.append(torch.full((height * width,), stride, device=feature.device, dtype=torch.float32))
    return torch.cat(points), torch.cat(point_strides)


def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    probabilities = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (cross_entropy * alpha_t * (1.0 - p_t).pow(gamma)).sum()


def _aligned_iou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(predicted[:, :2], target[:, :2])
    right_bottom = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(dim=1)
    union = predicted_area + target_area - intersection
    return 1.0 - intersection / union.clamp(min=1e-7)


class FCOSLoss:
    """Assign points to the smallest valid GT using FCOS scale ranges."""

    def __init__(self, num_classes: int, strides: tuple[int, ...] = (8, 16, 32, 64, 128)) -> None:
        self.num_classes = num_classes
        self.strides = strides
        self.regression_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, float("inf")))

    def __call__(
        self, outputs: dict[str, list[torch.Tensor]], box_lists: list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        features = outputs["class_logits"]
        batch_size = features[0].shape[0]
        points, point_strides = _points(features, self.strides)
        level_ranges = torch.cat([
            torch.tensor(regression_range, device=points.device).expand(feature.shape[-2] * feature.shape[-1], 2)
            for feature, regression_range in zip(features, self.regression_ranges)
        ])
        all_logits, all_distances, all_centerness = [], [], []
        for name in ("class_logits", "box_distances", "centerness_logits"):
            tensors = outputs[name]
            channels = tensors[0].shape[1]
            flattened = [tensor.permute(0, 2, 3, 1).reshape(batch_size, -1, channels) for tensor in tensors]
            if name == "class_logits":
                all_logits = torch.cat(flattened, dim=1)
            elif name == "box_distances":
                all_distances = torch.cat(flattened, dim=1) * point_strides[None, :, None]
            else:
                all_centerness = torch.cat(flattened, dim=1).squeeze(-1)

        target_labels = torch.full(
            (batch_size, len(points)), self.num_classes, device=points.device, dtype=torch.long
        )
        target_distances = torch.zeros((batch_size, len(points), 4), device=points.device)
        target_centerness = torch.zeros((batch_size, len(points)), device=points.device)
        for batch_index, rows in enumerate(box_lists):
            if rows.numel() == 0:
                continue
            rows = rows.to(device=points.device, dtype=torch.float32)
            classes = rows[:, 0].long()
            center_x, center_y, width, height = rows[:, 1:].unbind(dim=1)
            boxes = torch.stack((center_x - width / 2, center_y - height / 2,
                                 center_x + width / 2, center_y + height / 2), dim=1)
            # Dataset labels are normalized to the square network input.
            input_width = features[0].shape[-1] * self.strides[0]
            input_height = features[0].shape[-2] * self.strides[0]
            boxes = boxes * boxes.new_tensor((input_width, input_height, input_width, input_height))
            left_top = points[:, None, :] - boxes[None, :, :2]
            right_bottom = boxes[None, :, 2:] - points[:, None, :]
            distances = torch.cat((left_top, right_bottom), dim=-1)
            maximum = distances.max(dim=-1).values
            inside = distances.min(dim=-1).values > 0
            inside_range = (maximum >= level_ranges[:, :1]) & (maximum < level_ranges[:, 1:])
            valid = inside & inside_range
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            candidate_areas = areas[None, :].expand_as(maximum).masked_fill(~valid, float("inf"))
            best_area, best_gt = candidate_areas.min(dim=1)
            positive = torch.isfinite(best_area)
            if not positive.any():
                continue
            selected = distances[torch.arange(len(points), device=points.device), best_gt]
            target_labels[batch_index, positive] = classes[best_gt[positive]]
            target_distances[batch_index, positive] = selected[positive]
            l, t, r, b = selected[positive].unbind(dim=1)
            center_x_score = torch.minimum(l, r) / torch.maximum(l, r).clamp(min=1e-7)
            center_y_score = torch.minimum(t, b) / torch.maximum(t, b).clamp(min=1e-7)
            target_centerness[batch_index, positive] = torch.sqrt(center_x_score * center_y_score)

        positive = target_labels < self.num_classes
        positive_count = positive.sum()
        num_positive = positive.sum().clamp(min=1).to(torch.float32)
        one_hot = torch.zeros_like(all_logits)
        batch_ids, point_ids = positive.nonzero(as_tuple=True)
        if len(batch_ids):
            one_hot[batch_ids, point_ids, target_labels[batch_ids, point_ids]] = 1
        loss_classification = _focal_loss(all_logits.float(), one_hot) / num_positive
        if len(batch_ids):
            points_per_image = points.unsqueeze(0).expand(batch_size, -1, -1)
            pred_distances = all_distances[positive]
            pred_points = points_per_image[positive]
            pred_boxes = torch.cat((pred_points - pred_distances[:, :2], pred_points + pred_distances[:, 2:]), dim=1)
            target_dist = target_distances[positive]
            target_boxes = torch.cat((pred_points - target_dist[:, :2], pred_points + target_dist[:, 2:]), dim=1)
            centerness_target = target_centerness[positive]
            loss_box = (_aligned_iou_loss(pred_boxes, target_boxes) * centerness_target).sum() / num_positive
            loss_centerness = F.binary_cross_entropy_with_logits(
                all_centerness[positive].float(), centerness_target, reduction="sum"
            ) / num_positive
        else:
            loss_box = all_distances.sum() * 0
            loss_centerness = all_centerness.sum() * 0
        return {
            "total": loss_classification + loss_box + loss_centerness,
            "classification": loss_classification,
            "box": loss_box,
            "centerness": loss_centerness,
            "positives": positive_count.to(torch.float32),
        }
