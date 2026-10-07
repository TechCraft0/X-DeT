"""FCOS decoding and class-aware NMS."""
from __future__ import annotations

import torch
from torchvision.ops import batched_nms


def decode_predictions(
    outputs: dict[str, list[torch.Tensor]],
    original_sizes: list[tuple[int, int]],
    *,
    strides: tuple[int, ...] = (8, 16, 32, 64, 128),
    confidence_threshold: float = 0.05,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
    pre_nms_topk: int = 1000,
) -> list[dict[str, torch.Tensor]]:
    batch_size = outputs["class_logits"][0].shape[0]
    results: list[dict[str, torch.Tensor]] = []
    for batch_index in range(batch_size):
        image_boxes, image_scores, image_labels = [], [], []
        for logits, distances, centerness, stride in zip(
            outputs["class_logits"], outputs["box_distances"], outputs["centerness_logits"], strides
        ):
            _, _, height, width = logits.shape
            y, x = torch.meshgrid(
                torch.arange(height, device=logits.device, dtype=torch.float32) + 0.5,
                torch.arange(width, device=logits.device, dtype=torch.float32) + 0.5,
                indexing="ij",
            )
            points = torch.stack((x, y), dim=-1).reshape(-1, 2) * stride
            class_scores = logits[batch_index].permute(1, 2, 0).reshape(-1, logits.shape[1]).float().sigmoid()
            center_scores = centerness[batch_index, 0].reshape(-1, 1).float().sigmoid()
            scores = torch.sqrt(class_scores * center_scores)
            candidate_scores, candidate_labels = scores.max(dim=1)
            candidate_indices = (candidate_scores >= confidence_threshold).nonzero().squeeze(1)
            if candidate_indices.numel() == 0:
                continue
            if candidate_indices.numel() > pre_nms_topk:
                keep = candidate_scores[candidate_indices].topk(pre_nms_topk).indices
                candidate_indices = candidate_indices[keep]
            box_distances = distances[batch_index].permute(1, 2, 0).reshape(-1, 4).float()[candidate_indices] * stride
            selected_points = points[candidate_indices]
            boxes = torch.cat((selected_points - box_distances[:, :2], selected_points + box_distances[:, 2:]), dim=1)
            image_boxes.append(boxes)
            image_scores.append(candidate_scores[candidate_indices])
            image_labels.append(candidate_labels[candidate_indices])
        width, height = original_sizes[batch_index]
        if image_boxes:
            boxes = torch.cat(image_boxes)
            scores = torch.cat(image_scores)
            labels = torch.cat(image_labels)
            boxes[:, (0, 2)] *= width / (outputs["class_logits"][0].shape[-1] * strides[0])
            boxes[:, (1, 3)] *= height / (outputs["class_logits"][0].shape[-2] * strides[0])
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
