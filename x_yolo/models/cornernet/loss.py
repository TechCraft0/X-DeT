"""CornerNet Gaussian corner targets, offsets, and associative embeddings."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from x_yolo.models.keypoint_utils import draw_gaussian, gaussian_focal_loss, gaussian_radius


class CornerNetLoss:
    """Paper losses applied to both Hourglass-104 stacks."""

    def __init__(self, num_classes: int, pull_weight: float = 0.1,
                 push_weight: float = 0.1) -> None:
        self.num_classes = num_classes
        self.pull_weight = pull_weight
        self.push_weight = push_weight

    def build_targets(self, boxes_list: list[torch.Tensor], feature_shape: tuple[int, int],
                      device: torch.device) -> dict[str, object]:
        height, width = feature_shape
        # Per-object target construction stays on CPU: labels arrive there and
        # scalar CUDA reads inside this loop would add a synchronization per box.
        top_left_heatmap = torch.zeros((len(boxes_list), self.num_classes, height, width))
        bottom_right_heatmap = torch.zeros_like(top_left_heatmap)
        top_left_offsets = torch.zeros((len(boxes_list), 2, height, width))
        bottom_right_offsets = torch.zeros_like(top_left_offsets)
        matches: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []

        for batch_index, boxes in enumerate(boxes_list):
            per_image: list[tuple[tuple[int, int], tuple[int, int]]] = []
            if boxes.numel():
                boxes = boxes.detach().cpu()
                for class_id, center_x, center_y, box_w, box_h in boxes:
                    left = (center_x - box_w / 2).clamp(0, 1) * width
                    right = (center_x + box_w / 2).clamp(0, 1) * width
                    top = (center_y - box_h / 2).clamp(0, 1) * height
                    bottom = (center_y + box_h / 2).clamp(0, 1) * height
                    left_index = min(int(left), width - 1)
                    right_index = min(int(right), width - 1)
                    top_index = min(int(top), height - 1)
                    bottom_index = min(int(bottom), height - 1)
                    radius = max(0, int(gaussian_radius(
                        max(1.0, float(torch.ceil(bottom - top))),
                        max(1.0, float(torch.ceil(right - left))),
                    )))
                    channel = int(class_id)
                    draw_gaussian(top_left_heatmap[batch_index, channel], left_index, top_index, radius)
                    draw_gaussian(bottom_right_heatmap[batch_index, channel], right_index, bottom_index, radius)
                    top_left_offsets[batch_index, 0, top_index, left_index] = left - left_index
                    top_left_offsets[batch_index, 1, top_index, left_index] = top - top_index
                    bottom_right_offsets[batch_index, 0, bottom_index, right_index] = right - right_index
                    bottom_right_offsets[batch_index, 1, bottom_index, right_index] = bottom - bottom_index
                    per_image.append(((top_index, left_index), (bottom_index, right_index)))
            matches.append(per_image)

        return {name: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                for name, value in {
            "top_left_heatmap": top_left_heatmap,
            "bottom_right_heatmap": bottom_right_heatmap,
            "top_left_offsets": top_left_offsets,
            "bottom_right_offsets": bottom_right_offsets,
            "matches": matches,
        }.items()}

    def _embedding_loss(self, stack: dict[str, torch.Tensor], matches: list[list[tuple[tuple[int, int], tuple[int, int]]]]) -> tuple[torch.Tensor, torch.Tensor]:
        reference = stack["top_left_embedding"]
        pull_total = reference.sum() * 0
        push_total = reference.sum() * 0
        for batch_index, objects in enumerate(matches):
            if not objects:
                continue
            top_values = torch.stack([
                stack["top_left_embedding"][batch_index, 0, y, x]
                for (y, x), _ in objects
            ])
            bottom_values = torch.stack([
                stack["bottom_right_embedding"][batch_index, 0, y, x]
                for _, (y, x) in objects
            ])
            means = (top_values + bottom_values) / 2
            pull_total = pull_total + ((top_values - means).square() + (bottom_values - means).square()).sum() / len(objects)
            if len(objects) > 1:
                distances = (means[:, None] - means[None, :]).abs()
                off_diagonal = ~torch.eye(len(objects), dtype=torch.bool, device=reference.device)
                push_total = push_total + (1 - distances[off_diagonal]).clamp_min(0).sum() / (len(objects) * (len(objects) - 1))
        return pull_total * self.pull_weight, push_total * self.push_weight

    def __call__(self, outputs: dict[str, object], boxes_list: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        stacks = outputs["stacks"]
        if not isinstance(stacks, list) or not stacks:
            raise TypeError("CornerNet outputs must contain a non-empty stack list")
        shape = tuple(stacks[0]["top_left_heatmap_logits"].shape[-2:])
        targets = self.build_targets(boxes_list, shape, stacks[0]["top_left_heatmap_logits"].device)
        target_tl = targets["top_left_heatmap"]
        target_br = targets["bottom_right_heatmap"]
        tl_mask = target_tl.eq(1).any(dim=1, keepdim=True).float()
        br_mask = target_br.eq(1).any(dim=1, keepdim=True).float()
        positives_tl = target_tl.eq(1).sum().clamp_min(1)
        positives_br = target_br.eq(1).sum().clamp_min(1)

        totals = {name: stacks[0]["top_left_heatmap_logits"].sum() * 0
                  for name in ("heatmap", "offset", "pull", "push")}
        for stack in stacks:
            heat_tl = gaussian_focal_loss(stack["top_left_heatmap_logits"], target_tl, positives_tl)
            heat_br = gaussian_focal_loss(stack["bottom_right_heatmap_logits"], target_br, positives_br)
            totals["heatmap"] = totals["heatmap"] + (heat_tl + heat_br) / 2
            off_tl = F.smooth_l1_loss(stack["top_left_offsets"].float(), targets["top_left_offsets"], reduction="none")
            off_br = F.smooth_l1_loss(stack["bottom_right_offsets"].float(), targets["bottom_right_offsets"], reduction="none")
            totals["offset"] = totals["offset"] + (
                (off_tl * tl_mask).sum() / tl_mask.sum().clamp_min(1)
                + (off_br * br_mask).sum() / br_mask.sum().clamp_min(1)
            ) / 2
            pull, push = self._embedding_loss(stack, targets["matches"])
            totals["pull"] = totals["pull"] + pull
            totals["push"] = totals["push"] + push
        total = sum(totals.values())
        positives = target_tl.eq(1).sum() + target_br.eq(1).sum()
        return {"total": total, **totals, "positives": positives.float()}
