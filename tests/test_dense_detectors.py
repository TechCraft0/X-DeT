"""Small CPU checks for the FCOS and RTMDet model/loss/decode paths."""
from __future__ import annotations

import torch
from PIL import Image

from x_yolo.data.dataset import YoloTxtDetectionDataset
from x_yolo.models.factory import build_model
from x_yolo.models.fcos.loss import FCOSLoss
from x_yolo.models.fcos.postprocess import decode_predictions as decode_fcos
from x_yolo.models.rtmdet.loss import RTMDetLoss
from x_yolo.models.rtmdet.postprocess import decode_predictions as decode_rtmdet
from x_yolo.training.dense_trainer import _set_learning_rate


def test_fcos_forward_loss_backward_and_decode() -> None:
    model = build_model({"architecture": "fcos", "tower_depth": 1}, 3)
    images = torch.rand(1, 3, 128, 128)
    outputs = model(images)
    assert [tuple(feature.shape[-2:]) for feature in outputs["class_logits"]] == [
        (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)
    ]
    losses = FCOSLoss(3, model.strides)(outputs, [torch.tensor([[1, 0.5, 0.5, 0.2, 0.3]])])
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.classifier.weight.grad is not None
    predictions = decode_fcos(outputs, [(320, 240)], strides=model.strides,
                              confidence_threshold=1.0)
    assert all(result["boxes"].shape == (0, 4) for result in predictions)
    controlled = {
        "class_logits": [torch.full_like(value, -10) for value in outputs["class_logits"]],
        "box_distances": [torch.ones_like(value) for value in outputs["box_distances"]],
        "centerness_logits": [torch.full_like(value, -10) for value in outputs["centerness_logits"]],
    }
    controlled["class_logits"][0][0, 0, 8, 8] = 10
    controlled["centerness_logits"][0][0, 0, 8, 8] = 10
    decoded = decode_fcos(controlled, [(256, 64)], strides=model.strides, confidence_threshold=0.5)[0]
    assert torch.allclose(decoded["boxes"][0], torch.tensor([120.0, 30.0, 152.0, 38.0]))


def test_rtmdet_forward_loss_backward_and_decode() -> None:
    model = build_model({"architecture": "rtmdet"}, 3)
    outputs = model(torch.rand(1, 3, 128, 128))
    assert [tuple(feature.shape[-2:]) for feature in outputs["class_logits"]] == [(16, 16), (8, 8), (4, 4)]
    losses = RTMDetLoss(3, model.strides)(outputs, [torch.tensor([[1, 0.5, 0.5, 0.2, 0.3]])])
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.bbox_head.rtm_cls[0].weight.grad is not None
    predictions = decode_rtmdet(outputs, [(320, 240)], strides=model.strides,
                                confidence_threshold=1.0)
    assert all(result["boxes"].shape == (0, 4) for result in predictions)
    controlled = {
        "class_logits": [torch.full_like(value, -10) for value in outputs["class_logits"]],
        "box_distances": [torch.ones_like(value) for value in outputs["box_distances"]],
    }
    controlled["class_logits"][0][0, 0, 8, 8] = 10
    decoded = decode_rtmdet(controlled, [(256, 64)], strides=model.strides, confidence_threshold=0.5)[0]
    assert torch.allclose(decoded["boxes"][0], torch.tensor([126.0, 31.5, 130.0, 32.5]))


def test_horizontal_flip_updates_normalized_box_center(tmp_path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10), (100, 120, 140)).save(image_dir / "sample.jpg")
    (label_dir / "sample.txt").write_text("0 0.2 0.5 0.1 0.2\n", encoding="utf-8")
    config = {
        "class_names": ["object"],
        "splits": {"train": {"images": "images/train", "labels": "labels/train"},
                   "val": {"images": "images/train", "labels": "labels/train"}},
    }
    dataset = YoloTxtDetectionDataset(
        config, tmp_path, "train", input_size=16, augment=True,
        augmentation={"horizontal_flip": 1.0, "scale_range": (1.0, 1.0), "translate": 0,
                      "saturation": 1, "exposure": 1, "hue": 0},
    )
    _, boxes, _, _ = dataset[0]
    assert torch.isclose(boxes[0, 1], torch.tensor(0.8), atol=1e-6)


def test_cosine_learning_rate_checkpoint_values_are_plain_floats() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([{"params": [parameter], "lr_multiplier": 0.25}], lr=5e-4)
    learning_rate = _set_learning_rate(
        optimizer,
        {"learning_rate": 5e-4, "warmup_epochs": 0, "schedule": "cosine",
         "cosine_start_epoch": 50, "eta_min_ratio": 0.05},
        progress_epoch=65.5,
        planned_epochs=100,
    )

    assert type(learning_rate) is float
    assert type(optimizer.param_groups[0]["lr"]) is float
    assert optimizer.param_groups[0]["lr"] == learning_rate * 0.25
