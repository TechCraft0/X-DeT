"""RTMDet box decoding and class-aware NMS."""
from __future__ import annotations

import torch
from torchvision.ops import batched_nms


def decode_predictions(
    outputs: dict[str, list[torch.Tensor]],
    original_sizes: list[tuple[int, int]],
    *,
    strides: tuple[int, ...] = (8, 16, 32),
    confidence_threshold: float = 0.001,
    nms_iou_threshold: float = 0.65,
    max_detections: int = 100,
    pre_nms_topk: int = 3000,
) -> list[dict[str, torch.Tensor]]:
    batch_size = outputs["class_logits"][0].shape[0]
    results = []
    for batch_index in range(batch_size):
        boxes_by_level, scores_by_level, labels_by_level = [], [], []
        for class_logits, distances, stride in zip(outputs["class_logits"], outputs["box_distances"], strides):
            _, _, height, width = class_logits.shape
            y, x = torch.meshgrid(
                torch.arange(height, device=class_logits.device, dtype=torch.float32),
                torch.arange(width, device=class_logits.device, dtype=torch.float32),
                indexing="ij",
            )
            points = torch.stack((x, y), dim=-1).reshape(-1, 2) * stride
            scores = class_logits[batch_index].permute(1, 2, 0).reshape(-1, class_logits.shape[1]).float().sigmoid()
            point_indices, labels = (scores >= confidence_threshold).nonzero(as_tuple=True)
            if point_indices.numel() == 0:
                continue
            selected_scores = scores[point_indices, labels]
            distances_flat = distances[batch_index].permute(1, 2, 0).reshape(-1, 4).float()
            boxes = torch.cat((points[point_indices] - distances_flat[point_indices, :2],
                               points[point_indices] + distances_flat[point_indices, 2:]), dim=1)
            boxes_by_level.append(boxes)
            scores_by_level.append(selected_scores)
            labels_by_level.append(labels)
        width, height = original_sizes[batch_index]
        if boxes_by_level:
            boxes = torch.cat(boxes_by_level)
            scores = torch.cat(scores_by_level)
            labels = torch.cat(labels_by_level)
            if scores.numel() > pre_nms_topk:
                top = scores.topk(pre_nms_topk).indices
                boxes, scores, labels = boxes[top], scores[top], labels[top]
            input_width = outputs["class_logits"][0].shape[-1] * strides[0]
            input_height = outputs["class_logits"][0].shape[-2] * strides[0]
            boxes[:, (0, 2)] *= width / input_width
            boxes[:, (1, 3)] *= height / input_height
            boxes[:, (0, 2)].clamp_(0, width)
            boxes[:, (1, 3)].clamp_(0, height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
            keep = batched_nms(boxes, scores, labels, nms_iou_threshold)[:max_detections]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        else:
            device = outputs["class_logits"][0].device
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros((0,), device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)
        results.append({"boxes": boxes, "scores": scores, "labels": labels})
    return results
