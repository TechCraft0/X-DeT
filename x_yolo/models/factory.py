"""Small explicit factory for implemented model variants."""
from __future__ import annotations

from typing import Any

from x_yolo.models.yolov1.model import YoloV1


def build_model(model_config: dict[str, Any] | None, num_classes: int):
    """Build one documented architecture without hiding version differences."""
    config = model_config or {}
    architecture = str(config.get("architecture", "yolov1"))
    if architecture == "yolov1":
        return YoloV1(num_classes)
    if architecture == "yolov1_resnet50_transfer":
        from x_yolo.models.yolov1.resnet50_transfer import YoloV1ResNet50

        checkpoint = config.get("backbone_checkpoint")
        if not checkpoint:
            raise ValueError(
                "yolov1_resnet50_transfer requires an explicit model.backbone_checkpoint; "
                "weights are never downloaded automatically"
            )
        return YoloV1ResNet50(
            num_classes,
            checkpoint,
            freeze_backbone=bool(config.get("freeze_backbone", True)),
        )
    if architecture == "yolov1_torchvision_convhead":
        from x_yolo.models.yolov1.torchvision_conv_transfer import YoloV1TorchVisionConvHead

        checkpoint = config.get("backbone_checkpoint")
        weights = config.get("backbone_weights")
        if bool(checkpoint) == bool(weights):
            raise ValueError("yolov1_torchvision_convhead requires exactly one backbone_checkpoint or backbone_weights")
        return YoloV1TorchVisionConvHead(
            num_classes,
            backbone_name=str(config.get("backbone_name", "resnet50")),
            backbone_checkpoint=checkpoint,
            backbone_weights=weights,
            freeze_backbone=bool(config.get("freeze_backbone", True)),
            grid_size=int(config.get("grid_size", 7)),
        )
    if architecture == "yolov2":
        from x_yolo.models.yolov2.model import YoloV2

        return YoloV2(num_classes, anchors=config.get("anchors"))
    raise ValueError(
        f"Unknown model architecture {architecture!r}; available: "
        "yolov1, yolov1_resnet50_transfer, yolov1_torchvision_convhead, yolov2"
    )
