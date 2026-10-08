from __future__ import annotations

import torch

from x_yolo.models.factory import build_model
from x_yolo.models.centernet.loss import CenterNetLoss
from x_yolo.models.centernet.postprocess import decode_predictions as decode_centernet
from x_yolo.models.cornernet.loss import CornerNetLoss
from x_yolo.models.cornernet.postprocess import (
    _soft_nms,
    decode_predictions as decode_cornernet,
)


def test_centernet_factory_forward_targets_loss_and_decode() -> None:
    model = build_model({"architecture": "centernet"}, num_classes=3).eval()
    with torch.inference_mode():
        outputs = model(torch.rand(2, 3, 128, 128))
    assert outputs["center_heatmap_logits"].shape == (2, 3, 32, 32)
    assert outputs["box_sizes"].shape == (2, 2, 32, 32)

    loss_fn = CenterNetLoss(3)
    boxes = [torch.tensor([[1, 0.5, 0.5, 0.25, 0.5]], dtype=torch.float32),
             torch.zeros((0, 5), dtype=torch.float32)]
    train_outputs = model(torch.rand(2, 3, 128, 128))
    losses = loss_fn(train_outputs, boxes)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])

    feature = torch.zeros((1, 3, 32, 32))
    feature[0, 1, 8, 10] = 8
    sizes = torch.ones((1, 2, 32, 32)) * 4
    offsets = torch.zeros((1, 2, 32, 32))
    decoded = decode_centernet(
        {"center_heatmap_logits": feature, "box_sizes": sizes,
         "center_offsets": offsets, "input_shape": (128, 128)},
        [(256, 192)], confidence_threshold=0.5, topk=1,
    )[0]
    assert decoded["labels"].tolist() == [1]
    assert torch.allclose(decoded["boxes"][0], torch.tensor([64.0, 36.0, 96.0, 60.0]))


def test_cornernet_losses_and_corner_pair_decode() -> None:
    loss_fn = CornerNetLoss(3)
    stack = {
        "top_left_heatmap_logits": torch.zeros((1, 3, 16, 16), requires_grad=True),
        "bottom_right_heatmap_logits": torch.zeros((1, 3, 16, 16), requires_grad=True),
        "top_left_embedding": torch.zeros((1, 1, 16, 16), requires_grad=True),
        "bottom_right_embedding": torch.zeros((1, 1, 16, 16), requires_grad=True),
        "top_left_offsets": torch.zeros((1, 2, 16, 16), requires_grad=True),
        "bottom_right_offsets": torch.zeros((1, 2, 16, 16), requires_grad=True),
    }
    losses = loss_fn({"stacks": [stack], "input_shape": (64, 64)},
                     [torch.tensor([[2, 0.5, 0.5, 0.5, 0.5]])])
    losses["total"].backward()
    assert torch.isfinite(losses["total"])

    tl = torch.full((1, 3, 16, 16), -10.0)
    br = torch.full_like(tl, -10.0)
    tl[0, 2, 3, 2] = 4
    br[0, 2, 10, 12] = 4
    top_embeddings = torch.ones((1, 1, 16, 16))
    bottom_embeddings = torch.ones_like(top_embeddings)
    top_embeddings[0, 0, 3, 2] = 0.1
    bottom_embeddings[0, 0, 10, 12] = 0.2
    predictions = decode_cornernet({
        "stacks": [{
            "top_left_heatmap_logits": tl,
            "bottom_right_heatmap_logits": br,
            "top_left_embedding": top_embeddings,
            "bottom_right_embedding": bottom_embeddings,
            "top_left_offsets": torch.zeros((1, 2, 16, 16)),
            "bottom_right_offsets": torch.zeros((1, 2, 16, 16)),
        }],
        "input_shape": (64, 64),
    }, [(128, 96)], confidence_threshold=0.05, max_detections=10)[0]
    assert predictions["labels"].tolist() == [2]
    assert torch.allclose(predictions["boxes"][0], torch.tensor([16.0, 18.0, 96.0, 60.0]))


def test_cornernet_gaussian_soft_nms_decays_overlapping_boxes() -> None:
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7])
    labels = torch.zeros(3, dtype=torch.long)
    kept_boxes, kept_scores, _ = _soft_nms(boxes, scores, labels, threshold=0.05)
    assert kept_boxes.shape == (3, 4)
    decay = torch.exp(torch.tensor(-((81 / 119) ** 2) / 0.5))
    assert torch.allclose(kept_scores, torch.tensor([0.9, 0.7, 0.8 * decay]), atol=1e-4)


def test_cornernet_hourglass104_forward_shape() -> None:
    model = build_model({"architecture": "cornernet"}, num_classes=2).eval()
    with torch.inference_mode():
        outputs = model(torch.rand(1, 3, 128, 128))
    assert len(outputs["stacks"]) == 2
    assert outputs["stacks"][-1]["top_left_heatmap_logits"].shape == (1, 2, 32, 32)
