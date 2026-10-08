"""CornerNet top-left/bottom-right pairing and class-aware soft NMS."""
from __future__ import annotations

import numpy as np
import torch

from x_yolo.models.keypoint_utils import local_maximum, gather_feature_map


def _soft_nms(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor,
              *, sigma: float = 0.5, threshold: float = 0.05,
              max_detections: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = boxes.device
    box_array = boxes.detach().float().cpu().numpy()
    score_array = scores.detach().float().cpu().numpy().copy()
    label_array = labels.detach().cpu().numpy()
    kept_indices: list[int] = []
    kept_scores: list[float] = []
    kept_labels: list[int] = []
    for class_id in np.unique(label_array):
        pending = np.flatnonzero(label_array == class_id).tolist()
        while pending:
            local_scores = score_array[pending]
            best_slot = int(local_scores.argmax())
            best_index = pending.pop(best_slot)
            best_score = float(score_array[best_index])
            if best_score < threshold:
                break
            kept_indices.append(best_index)
            kept_scores.append(best_score)
            kept_labels.append(int(class_id))
            if len(kept_indices) >= max_detections or not pending:
                break

            rest = np.asarray(pending, dtype=np.int64)
            best_box = box_array[best_index]
            other_boxes = box_array[rest]
            top_left = np.maximum(best_box[:2], other_boxes[:, :2])
            bottom_right = np.minimum(best_box[2:], other_boxes[:, 2:])
            intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
            best_area = np.maximum(best_box[2:] - best_box[:2], 0).prod()
            other_areas = np.maximum(other_boxes[:, 2:] - other_boxes[:, :2], 0).prod(axis=1)
            iou = intersection / np.maximum(best_area + other_areas - intersection, 1e-8)
            score_array[rest] *= np.exp(-(iou ** 2) / sigma)
            pending = [index for index in pending if score_array[index] >= threshold]
    if not kept_indices:
        return boxes[:0], scores[:0], labels[:0]
    order = np.argsort(-np.asarray(kept_scores))[:max_detections]
    selected = np.asarray(kept_indices, dtype=np.int64)[order]
    kept_box_tensor = torch.as_tensor(box_array[selected], dtype=torch.float32, device=device)
    kept_score_tensor = torch.as_tensor(np.asarray(kept_scores)[order], dtype=torch.float32, device=device)
    kept_label_tensor = torch.as_tensor(np.asarray(kept_labels)[order], dtype=labels.dtype, device=device)
    return kept_box_tensor, kept_score_tensor, kept_label_tensor


@torch.inference_mode()
def decode_predictions(outputs: dict[str, object], original_sizes: list[tuple[int, int]], *,
                       confidence_threshold: float = 0.05,
                       nms_iou_threshold: float = 0.5,
                       max_detections: int = 100,
                       topk: int = 100,
                       embedding_distance: float = 0.5) -> list[dict[str, torch.Tensor]]:
    """Pair same-class local-max corners whose learned tags are close."""
    del nms_iou_threshold  # Gaussian soft-NMS uses its sigma parameter.
    stack = outputs["stacks"][-1]
    tl_heat = local_maximum(stack["top_left_heatmap_logits"].sigmoid())
    br_heat = local_maximum(stack["bottom_right_heatmap_logits"].sigmoid())
    batch, classes, feature_h, feature_w = tl_heat.shape
    count = min(topk, classes * feature_h * feature_w)
    tl_scores, tl_flat = tl_heat.reshape(batch, -1).topk(count, dim=1)
    br_scores, br_flat = br_heat.reshape(batch, -1).topk(count, dim=1)
    tl_labels, br_labels = tl_flat // (feature_h * feature_w), br_flat // (feature_h * feature_w)
    tl_indices, br_indices = tl_flat % (feature_h * feature_w), br_flat % (feature_h * feature_w)
    tl_emb = gather_feature_map(stack["top_left_embedding"], tl_indices)[..., 0]
    br_emb = gather_feature_map(stack["bottom_right_embedding"], br_indices)[..., 0]
    tl_offsets = gather_feature_map(stack["top_left_offsets"], tl_indices)
    br_offsets = gather_feature_map(stack["bottom_right_offsets"], br_indices)
    tl_x = (tl_indices % feature_w).to(tl_offsets.dtype) + tl_offsets[..., 0]
    tl_y = (tl_indices // feature_w).to(tl_offsets.dtype) + tl_offsets[..., 1]
    br_x = (br_indices % feature_w).to(br_offsets.dtype) + br_offsets[..., 0]
    br_y = (br_indices // feature_w).to(br_offsets.dtype) + br_offsets[..., 1]
    decoded = []
    input_h, input_w = outputs["input_shape"]
    for batch_index, (original_w, original_h) in enumerate(original_sizes):
        class_match = tl_labels[batch_index, :, None] == br_labels[batch_index, None, :]
        tag_match = (tl_emb[batch_index, :, None] - br_emb[batch_index, None, :]).abs() <= embedding_distance
        geometry = (br_x[batch_index, None, :] > tl_x[batch_index, :, None]) & (
            br_y[batch_index, None, :] > tl_y[batch_index, :, None]
        )
        pair_scores = (tl_scores[batch_index, :, None] + br_scores[batch_index, None, :]) / 2
        valid = class_match & tag_match & geometry & (pair_scores >= confidence_threshold)
        pair_indices = valid.nonzero(as_tuple=False)
        if pair_indices.numel() == 0:
            decoded.append({"boxes": tl_heat.new_zeros((0, 4)), "scores": tl_heat.new_zeros((0,)),
                            "labels": torch.zeros((0,), dtype=torch.long, device=tl_heat.device)})
            continue
        top_index, bottom_index = pair_indices.unbind(1)
        boxes = torch.stack((
            tl_x[batch_index, top_index] * (input_w / feature_w),
            tl_y[batch_index, top_index] * (input_h / feature_h),
            br_x[batch_index, bottom_index] * (input_w / feature_w),
            br_y[batch_index, bottom_index] * (input_h / feature_h),
        ), dim=1)
        boxes[:, 0] = boxes[:, 0].clamp(0, input_w) * (original_w / input_w)
        boxes[:, 2] = boxes[:, 2].clamp(0, input_w) * (original_w / input_w)
        boxes[:, 1] = boxes[:, 1].clamp(0, input_h) * (original_h / input_h)
        boxes[:, 3] = boxes[:, 3].clamp(0, input_h) * (original_h / input_h)
        scores = pair_scores[top_index, bottom_index]
        labels = tl_labels[batch_index, top_index]
        if scores.numel() > 1000:
            scores, order = scores.topk(1000)
            boxes, labels = boxes[order], labels[order]
        boxes, scores, labels = _soft_nms(boxes, scores, labels,
                                          threshold=confidence_threshold,
                                          max_detections=max_detections)
        decoded.append({"boxes": boxes, "scores": scores, "labels": labels})
    return decoded
