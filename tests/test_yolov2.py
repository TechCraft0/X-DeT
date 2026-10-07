from pathlib import Path
import struct

import pytest
import torch
import yaml

from x_yolo.models.factory import build_model
from x_yolo.models.yolov2.loss import YoloV2Loss, build_targets
from x_yolo.models.yolov2.model import DEFAULT_VOC_ANCHORS, YoloV2
from x_yolo.models.yolov2.postprocess import decode_predictions


def test_yolov2_factory_and_forward_shape() -> None:
    model = build_model({"architecture": "yolov2"}, num_classes=3).eval()
    with torch.inference_mode():
        predictions = model(torch.zeros((1, 3, 64, 64)))
    assert predictions.shape == (1, 2, 2, 5, 8)


def test_yolov2_default_recipe_selects_yolov2_model() -> None:
    recipe_path = Path(__file__).resolve().parents[1] / "configs/yolov2/voc0712.yaml"
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    model = build_model(recipe.get("model"), num_classes=20)
    assert isinstance(model, YoloV2)


def test_yolov2_requires_stride_aligned_input() -> None:
    model = YoloV2(num_classes=2)
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.zeros((1, 3, 65, 64)))


def test_darknet_weight_loader_rejects_truncated_weight_data(tmp_path: Path) -> None:
    weight_path = tmp_path / "truncated.weights"
    # Darknet v2 header: three int32 values followed by a uint64 seen count.
    weight_path.write_bytes(struct.pack("<3iQ", 0, 2, 0, 0) + b"\0\0\0\0")
    with pytest.raises(ValueError, match="Expected Darknet-19 448 weights"):
        YoloV2(num_classes=20).load_darknet19_weights(weight_path)


def test_anchor_assignment_and_same_cell_collision_are_explicit() -> None:
    # Equal boxes choose the same best anchor and cell; one slot can represent
    # only one target, so the deterministic first label is retained.
    labels = torch.tensor(
        [[0, 0.20, 0.30, 0.10, 0.12], [1, 0.22, 0.30, 0.10, 0.12]],
        dtype=torch.float32,
    )
    targets = build_targets([labels], num_classes=2, anchors=DEFAULT_VOC_ANCHORS, grid_size=13)
    assert targets.object_mask.sum().item() == 1
    assert targets.ignored_ground_truths == 1
    assert len(targets.all_boxes[0]) == 2
    row, column = int(0.30 * 13), int(0.20 * 13)
    assert targets.classes[0, row, column].max().item() == 0


def test_yolov2_loss_is_finite_and_backpropagates_for_positive_and_empty_images() -> None:
    anchors = torch.tensor(DEFAULT_VOC_ANCHORS)
    labels = torch.tensor([[1, 0.43, 0.61, 0.14, 0.20]], dtype=torch.float32)
    targets = build_targets([labels, torch.zeros((0, 5))], 2, anchors, grid_size=2)
    predictions = torch.randn((2, 2, 2, 5, 7), requires_grad=True)
    losses = YoloV2Loss(2, anchors)(predictions, targets)
    losses["total"].backward()
    assert all(torch.isfinite(value) for value in losses.values())
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_yolov2_decode_returns_xyxy_pixels_and_objectness_class_score() -> None:
    output = torch.full((1, 13, 13, 5, 7), -20.0)
    output[0, 6, 6, 0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 20.0, 20.0, -20.0])
    decoded = decode_predictions(
        output,
        [(416, 416)],
        anchors=DEFAULT_VOC_ANCHORS,
        confidence_threshold=0.5,
    )[0]
    assert decoded["boxes"].shape == (1, 4)
    assert decoded["labels"].item() == 0
    assert torch.allclose(decoded["scores"], torch.ones_like(decoded["scores"]), atol=1e-5)
    expected_width = DEFAULT_VOC_ANCHORS[0][0] * 32
    assert decoded["boxes"][0, 2].item() - decoded["boxes"][0, 0].item() == pytest.approx(expected_width)


def test_yolov2_target_builder_rejects_invalid_labels() -> None:
    labels = torch.tensor([[2, 0.2, 0.3, 0.1, 0.1]])
    with pytest.raises(ValueError, match="outside"):
        build_targets([labels], 2, DEFAULT_VOC_ANCHORS, grid_size=13)
