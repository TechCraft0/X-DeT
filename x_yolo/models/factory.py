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
    if architecture == "yolov3":
        from x_yolo.models.yolov3.model import YoloV3

        return YoloV3(
            num_classes,
            anchors=config.get("anchors"),
            anchor_masks=config.get("anchor_masks"),
            backbone_name=str(config.get("backbone_name", "darknet53")),
            backbone_weights=config.get("backbone_weights"),
        )
    if architecture == "fcos":
        from x_yolo.models.fcos.model import FCOS

        return FCOS(
            num_classes,
            backbone_checkpoint=config.get("backbone_checkpoint"),
            trainable_layers=int(config.get("trainable_layers", 3)),
            tower_depth=int(config.get("tower_depth", 4)),
            channels=int(config.get("channels", 256)),
        )
    if architecture == "rtmdet":
        from x_yolo.models.rtmdet.model import RTMDet

        return RTMDet(num_classes, channels=int(config.get("channels", 96)))
    if architecture == "centernet":
        from x_yolo.models.centernet.model import CenterNet

        return CenterNet(
            num_classes,
            backbone_weights=config.get("backbone_weights"),
        )
    if architecture == "cornernet":
        from x_yolo.models.cornernet.model import CornerNet

        return CornerNet(num_classes)
    raise ValueError(
        f"Unknown model architecture {architecture!r}; available: "
        "yolov1, yolov1_resnet50_transfer, yolov1_torchvision_convhead, yolov2, yolov3, "
        "fcos, rtmdet, centernet, cornernet"
    )
