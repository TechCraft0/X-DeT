"""Dynamic soft-label assignment, Quality Focal Loss, and GIoU for RTMDet."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _point_priors(features: list[torch.Tensor], strides: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    points, point_strides = [], []
    for feature, stride in zip(features, strides):
        height, width = feature.shape[-2:]
        y, x = torch.meshgrid(
            torch.arange(height, device=feature.device, dtype=torch.float32),
            torch.arange(width, device=feature.device, dtype=torch.float32),
            indexing="ij",
        )
        points.append(torch.stack((x, y), dim=-1).reshape(-1, 2) * stride)
        point_strides.append(torch.full((height * width,), stride, device=feature.device, dtype=torch.float32))
    return torch.cat(points), torch.cat(point_strides)


def _boxes_from_distances(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    return torch.cat((points - distances[:, :2], points + distances[:, 2:]), dim=-1)


def _pairwise_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(first[:, None, :2], second[None, :, :2])
    right_bottom = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(dim=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp(min=1e-7)


def _aligned_giou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(predicted[:, :2], target[:, :2])
    right_bottom = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)
    pred_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(dim=1)
    union = pred_area + target_area - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing_left = torch.minimum(predicted[:, :2], target[:, :2])
    enclosing_right = torch.maximum(predicted[:, 2:], target[:, 2:])
    enclosing_area = (enclosing_right - enclosing_left).clamp(min=0).prod(dim=1)
    return 1.0 - (iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-7))


class RTMDetLoss:
    """Match points with dynamic-k IoU/quality/class/center costs.

    The soft-center prior and dynamic-k matching follow MMDetection's public
    DynamicSoftLabelAssigner defaults (radius 3, top-k 13, IoU cost 3).
    """

    def __init__(self, num_classes: int, strides: tuple[int, ...] = (8, 16, 32),
                 topk: int = 13, soft_center_radius: float = 3.0,
                 iou_cost_weight: float = 3.0, bbox_loss_weight: float = 2.0) -> None:
        self.num_classes = num_classes
        self.strides = strides
        self.topk = topk
        self.soft_center_radius = soft_center_radius
        self.iou_cost_weight = iou_cost_weight
        self.bbox_loss_weight = bbox_loss_weight

    def __call__(self, outputs: dict[str, list[torch.Tensor]],
                 box_lists: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        feature_maps = outputs["class_logits"]
        batch_size = feature_maps[0].shape[0]
        points, point_strides = _point_priors(feature_maps, self.strides)
        logits = torch.cat([
            feature.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_classes)
            for feature in outputs["class_logits"]
        ], dim=1).float()
        distances = torch.cat([
            feature.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
            for feature in outputs["box_distances"]
        ], dim=1).float()
        expanded_points = points.unsqueeze(0).expand(batch_size, -1, -1)
        predicted_boxes = _boxes_from_distances(expanded_points.reshape(-1, 2),
                                               distances.reshape(-1, 4)).reshape(batch_size, -1, 4)
        target_scores = torch.zeros_like(logits)
        target_boxes = torch.zeros_like(predicted_boxes)
        assigned_quality = torch.zeros((batch_size, len(points)), device=points.device)
        positive_mask = torch.zeros((batch_size, len(points)), dtype=torch.bool, device=points.device)
        input_height = feature_maps[0].shape[-2] * self.strides[0]
        input_width = feature_maps[0].shape[-1] * self.strides[0]

        for image_index, rows in enumerate(box_lists):
            if rows.numel() == 0:
                continue
            rows = rows.to(device=points.device, dtype=torch.float32)
            labels = rows[:, 0].long()
            center_x, center_y, width, height = rows[:, 1:].unbind(dim=1)
            gt_boxes = torch.stack((center_x - width / 2, center_y - height / 2,
                                    center_x + width / 2, center_y + height / 2), dim=1)
            gt_boxes = gt_boxes * gt_boxes.new_tensor((input_width, input_height, input_width, input_height))
            deltas = torch.cat((points[:, None] - gt_boxes[None, :, :2],
                                gt_boxes[None, :, 2:] - points[:, None]), dim=-1)
            inside = deltas.min(dim=-1).values > 0
            candidate_indices = inside.any(dim=1).nonzero().squeeze(1)
            if candidate_indices.numel() == 0:
                continue

            # Assignment is a target-building step; never backpropagate through
            # the selected quality labels or matching costs.
            candidate_logits = logits[image_index, candidate_indices].detach()
            candidate_boxes = predicted_boxes[image_index, candidate_indices].detach()
            candidate_points = points[candidate_indices]
            pairwise_ious = _pairwise_iou(candidate_boxes, gt_boxes).clamp(min=0, max=1)
            soft_labels = F.one_hot(labels, self.num_classes).float()[None] * pairwise_ious[..., None]
            expanded_logits = candidate_logits[:, None, :].expand(-1, len(gt_boxes), -1)
            probability = expanded_logits.sigmoid()
            classification_cost = (
                F.binary_cross_entropy_with_logits(expanded_logits, soft_labels, reduction="none")
                * (probability - soft_labels).abs().pow(2)
            ).sum(dim=-1)
            iou_cost = -torch.log(pairwise_ious + 1e-7) * self.iou_cost_weight
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
            center_distances = torch.cdist(candidate_points, gt_centers) / point_strides[candidate_indices, None]
            center_cost = torch.pow(10.0, (center_distances - self.soft_center_radius).clamp(max=30))
            cost = classification_cost + iou_cost + center_cost

            topk_ious = pairwise_ious.topk(min(self.topk, pairwise_ious.shape[0]), dim=0).values
            dynamic_ks = topk_ious.sum(dim=0).to(torch.int64).clamp(min=1, max=pairwise_ious.shape[0])
            sorted_candidates = cost.argsort(dim=0)
            ranks = torch.arange(cost.shape[0], device=cost.device)[:, None]
            matching = torch.zeros_like(cost, dtype=torch.bool)
            matching.scatter_(0, sorted_candidates, ranks < dynamic_ks[None, :])
            multiply_matched = matching.sum(dim=1) > 1
            if multiply_matched.any():
                conflicting_costs = cost[multiply_matched]
                best_gt = conflicting_costs.argmin(dim=1)
                matching[multiply_matched] = False
                matching[multiply_matched, best_gt] = True
            candidate_positive = matching.any(dim=1).nonzero().squeeze(1)
            global_positive = candidate_indices[candidate_positive]
            matched_gt = matching[candidate_positive].to(torch.int64).argmax(dim=1)
            quality = pairwise_ious[candidate_positive, matched_gt]
            positive_mask[image_index, global_positive] = True
            assigned_quality[image_index, global_positive] = quality
            target_boxes[image_index, global_positive] = gt_boxes[matched_gt]
            target_scores[image_index, global_positive, labels[matched_gt]] = quality

        normalizer = assigned_quality.sum().clamp(min=1.0)
        probabilities = logits.sigmoid()
        loss_classification = (
            F.binary_cross_entropy_with_logits(logits, target_scores, reduction="none")
            * (probabilities - target_scores).abs().pow(2)
        ).sum() / normalizer
        if positive_mask.any():
            loss_box = (
                _aligned_giou_loss(predicted_boxes[positive_mask], target_boxes[positive_mask])
                * assigned_quality[positive_mask]
            ).sum() / normalizer * self.bbox_loss_weight
        else:
            loss_box = distances.sum() * 0
        return {
            "total": loss_classification + loss_box,
            "classification": loss_classification,
            "box": loss_box,
            "positives": positive_mask.sum().to(torch.float32),
        }
