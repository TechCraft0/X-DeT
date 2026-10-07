from pathlib import Path
import struct

import pytest
import torch
import yaml

from x_yolo.models.factory import build_model
from x_yolo.models.yolov3.loss import YoloV3Loss, build_targets
from x_yolo.models.yolov3.model import (
    DEFAULT_ANCHOR_MASKS,
    DEFAULT_VOC_ANCHORS,
    YoloV3,
)
from x_yolo.models.yolov3.postprocess import decode_predictions


def test_yolov3_factory_and_three_scale_forward_shapes() -> None:
    model = build_model({"architecture": "yolov3"}, num_classes=3).eval()
    with torch.inference_mode():
        outputs = model(torch.zeros((1, 3, 64, 64)))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 2, 2, 3, 8),
        (1, 4, 4, 3, 8),
        (1, 8, 8, 3, 8),
    ]


def test_yolov3_default_recipe_selects_yolov3_model() -> None:
    recipe_path = Path(__file__).resolve().parents[1] / "configs/yolov3/voc0712.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    model = build_model(recipe.get("model"), num_classes=20)
    assert isinstance(model, YoloV3)
    assert model.anchor_masks == DEFAULT_ANCHOR_MASKS


def test_yolov3_convnext_backbone_returns_three_stride_aligned_heads() -> None:
    model = build_model(
        {"architecture": "yolov3", "backbone_name": "convnext_small", "backbone_weights": "IMAGENET1K_V1"},
        num_classes=2,
    ).eval()
    assert model.backbone_name == "convnext_small"
    assert model.backbone_weights == "IMAGENET1K_V1"
    with torch.inference_mode():
        outputs = model(torch.zeros((1, 3, 64, 64)))
    assert [tuple(output.shape) for output in outputs] == [
        (1, 2, 2, 3, 7),
        (1, 4, 4, 3, 7),
        (1, 8, 8, 3, 7),
    ]


def test_yolov3_rejects_input_that_is_not_stride_aligned() -> None:
    model = YoloV3(num_classes=2)
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.zeros((1, 3, 65, 64)))


def test_best_anchor_is_assigned_to_its_scale_and_collisions_are_counted() -> None:
    # A 10x13-pixel object at 64x64 exactly matches global anchor zero,
    # which belongs to local slot zero at the fine (stride-8) scale.
    labels = torch.tensor(
        [[0, 0.31, 0.56, 10 / 64, 13 / 64], [1, 0.31, 0.56, 10 / 64, 13 / 64]],
        dtype=torch.float32,
    )
    targets = build_targets(
        [labels],
        num_classes=2,
        grid_sizes=[(2, 2), (4, 4), (8, 8)],
        anchors=DEFAULT_VOC_ANCHORS,
        anchor_masks=DEFAULT_ANCHOR_MASKS,
    )
    assert sum(int(mask.sum()) for mask in targets.object_masks) == 1
    assert targets.object_masks[2][0, 4, 2, 0]
    assert targets.classes[2][0, 4, 2, 0].item() == 0
    assert targets.ignored_ground_truths == 1
    assert len(targets.all_boxes[0]) == 2


def test_yolov3_loss_is_finite_and_backpropagates_on_all_scales() -> None:
    labels = torch.tensor([[1, 0.43, 0.61, 0.14, 0.20]], dtype=torch.float32)
    targets = build_targets(
        [labels, torch.zeros((0, 5))],
        num_classes=2,
        grid_sizes=[(2, 2), (4, 4), (8, 8)],
    )
    predictions = [
        torch.randn((2, height, width, 3, 7), requires_grad=True)
        for height, width in ((2, 2), (4, 4), (8, 8))
    ]
    losses = YoloV3Loss(2)(predictions, targets)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(output.grad is not None and torch.isfinite(output.grad).all() for output in predictions)


def test_yolov3_decode_uses_multilabel_sigmoid_scores_and_pixel_boxes() -> None:
    outputs = [
        torch.full((1, 2, 2, 3, 7), -20.0),
        torch.full((1, 4, 4, 3, 7), -20.0),
        torch.full((1, 8, 8, 3, 7), -20.0),
    ]
    outputs[2][0, 4, 2, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0, 20.0, -20.0])
    detections = decode_predictions(outputs, [(128, 96)], confidence_threshold=0.5)[0]
    assert detections["boxes"].shape == (1, 4)
    assert detections["labels"].item() == 0
    assert torch.allclose(detections["scores"], torch.ones_like(detections["scores"]), atol=1e-5)
    assert detections["boxes"][0, 2].item() - detections["boxes"][0, 0].item() == pytest.approx(20.0)


def test_darknet53_weight_loader_rejects_truncated_weight_data(tmp_path: Path) -> None:
    weight_path = tmp_path / "truncated.weights"
    weight_path.write_bytes(struct.pack("<3iQ", 0, 2, 0, 0) + b"\0\0\0\0")
    with pytest.raises(ValueError, match="52 feature blocks"):
        YoloV3(num_classes=20).load_darknet53_weights(weight_path)
