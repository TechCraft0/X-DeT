import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw

from x_yolo.data.dataset import _apply_random_geometry
from x_yolo.evaluation.metrics import evaluate_predictions
from x_yolo.models.yolov1.loss import YoloV1Loss, build_targets
from x_yolo.models.yolov1.postprocess import class_aware_nms, decode_predictions
from x_yolo.training.trainer import (
    _optimizer_parameter_groups,
    _set_backbone_trainability,
    _should_save_last_checkpoint,
)


class _TinyTransferModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.freeze_backbone = True
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.freeze_backbone = not trainable
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(trainable)


def test_transfer_optimizer_keeps_frozen_backbone_and_uses_lower_lr() -> None:
    model = _TinyTransferModel()
    groups = _optimizer_parameter_groups(model, learning_rate=1e-4, backbone_lr_multiplier=0.1)
    assert [group["name"] for group in groups] == ["detector", "backbone"]
    assert all(not parameter.requires_grad for parameter in groups[1]["params"])
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-5

    _set_backbone_trainability(model, trainable=True)
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_last_checkpoint_saves_at_interval_and_at_final_epoch() -> None:
    assert _should_save_last_checkpoint(epoch_number=60, interval_epochs=10, is_final_epoch=False)
    assert not _should_save_last_checkpoint(epoch_number=61, interval_epochs=10, is_final_epoch=False)
    assert _should_save_last_checkpoint(epoch_number=61, interval_epochs=10, is_final_epoch=True)


def test_class_aware_nms_matches_torchvision_batched_nms() -> None:
    from torchvision.ops import batched_nms

    generator = torch.Generator().manual_seed(42)
    top_left = torch.rand((120, 2), generator=generator)
    box_size = torch.rand((120, 2), generator=generator) * 0.25
    boxes = torch.cat((top_left, top_left + box_size), dim=1)
    scores = torch.rand((120,), generator=generator)
    labels = torch.randint(0, 4, (120,), generator=generator)
    expected = batched_nms(boxes, scores, labels, 0.45)
    actual = class_aware_nms(boxes, scores, labels, 0.45)
    assert torch.equal(actual, expected)


def test_target_collision_uses_largest_box_and_records_ignored() -> None:
    labels = torch.tensor(
        [[0, 0.10, 0.10, 0.08, 0.08], [1, 0.12, 0.11, 0.20, 0.20]], dtype=torch.float32
    )
    targets = build_targets([labels], num_classes=2)
    assert targets.ignored_ground_truths == 1
    assert targets.classes[0, 0, 0].item() == 1
    assert targets.object_mask.sum().item() == 1


def test_grid14_target_assigns_objects_that_share_a_grid7_cell() -> None:
    labels = torch.tensor(
        [[0, 0.05, 0.05, 0.04, 0.04], [1, 0.10, 0.10, 0.04, 0.04]], dtype=torch.float32
    )
    targets = build_targets([labels], num_classes=2, grid_size=14)
    assert targets.boxes.shape == (1, 14, 14, 4)
    assert targets.ignored_ground_truths == 0
    assert targets.object_mask.sum().item() == 2


def test_random_geometry_keeps_image_and_box_centers_aligned() -> None:
    image = Image.new("RGB", (100, 100), (0, 0, 0))
    ImageDraw.Draw(image).rectangle((28, 28, 32, 32), fill=(255, 255, 255))
    boxes = torch.tensor([[0.0, 0.3, 0.3, 0.04, 0.04]])

    transformed_image, transformed_boxes = _apply_random_geometry(
        image, boxes, scale_range=(2.0, 2.0), translate=0.0
    )

    bright_pixels = torch.from_numpy(np.array(transformed_image))[..., 0] > 200
    pixel_y, pixel_x = torch.where(bright_pixels)
    expected_center = torch.tensor([0.1, 0.1])
    actual_box_center = transformed_boxes[0, 1:3]
    assert torch.allclose(actual_box_center, expected_center)
    # A source marker centered at (30, 30) must move to (10, 10) when the
    # image is scaled by 2 around its center. The old transform moved it to 60.
    assert abs(float(pixel_x.float().mean()) - 10.0) < 1.0
    assert abs(float(pixel_y.float().mean()) - 10.0) < 1.0


def test_yolov1_loss_has_finite_gradients() -> None:
    labels = torch.tensor([[0, 0.2, 0.25, 0.1, 0.2]], dtype=torch.float32)
    targets = build_targets([labels], num_classes=2)
    predictions = torch.zeros((1, 7, 7, 12), requires_grad=True)
    loss = YoloV1Loss(num_classes=2)(predictions, targets)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()


def test_giou_localization_term_is_zero_for_exact_boxes_and_has_gradients_when_offset() -> None:
    labels = torch.tensor([[0, 0.2, 0.25, 0.1, 0.2]], dtype=torch.float32)
    targets = build_targets([labels], num_classes=2)
    row, column = 1, 1
    target_box = targets.boxes[0, row, column]
    predictions = torch.zeros((1, 7, 7, 12))
    for box_index in range(2):
        offset = box_index * 5
        predictions[0, row, column, offset : offset + 5] = torch.tensor(
            [target_box[0], target_box[1], target_box[2].sqrt(), target_box[3].sqrt(), 1.0]
        )
    predictions[0, row, column, 10] = 1.0

    loss = YoloV1Loss(num_classes=2, lambda_giou=1.0)(predictions, targets)
    assert torch.allclose(loss["giou"], torch.tensor(0.0), atol=1e-6)

    predictions[0, row, column, 0] += 0.1
    predictions[0, row, column, 5] += 0.1
    predictions.requires_grad_()
    offset_loss = YoloV1Loss(num_classes=2, lambda_giou=1.0)(predictions, targets)
    assert offset_loss["giou"].item() > 0
    offset_loss["total"].backward()
    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
    assert predictions.grad[0, row, column, 0].abs().item() > 0


def test_decode_and_ap_use_xyxy_pixel_contract() -> None:
    # One cell: B=2 box tuples and shared two-class distribution.
    output = torch.zeros((1, 7, 7, 12))
    output[0, 0, 0, :5] = torch.tensor([0.5, 0.5, 0.2, 0.2, 0.9])
    output[0, 0, 0, 10:] = torch.tensor([0.8, 0.1])
    detections = decode_predictions(output, [(100, 80)], confidence_threshold=0.5)[0]
    assert detections["boxes"].shape == (1, 4)
    assert detections["labels"].item() == 0
    decoded_size = 0.2**2
    expected_box = torch.tensor([100 * (0.5 / 7 - decoded_size / 2), 80 * (0.5 / 7 - decoded_size / 2), 100 * (0.5 / 7 + decoded_size / 2), 80 * (0.5 / 7 + decoded_size / 2)])
    assert torch.allclose(detections["boxes"][0], expected_box)
    report = evaluate_predictions(
        [detections],
        [{"boxes": expected_box.unsqueeze(0), "labels": torch.tensor([0])}],
        ["person", "car"],
    )
    assert report["mAP50"] == 1.0
    assert report["mAP50_95"] == 1.0
