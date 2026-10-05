#!/usr/bin/env python3
"""Run YOLOv1 inference on one image and print the documented xyxy contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image
from x_yolo.data.dataset import load_dataset_config
from x_yolo.models.factory import build_model
from x_yolo.models.yolov1.postprocess import decode_predictions
from x_yolo.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recipe", default=str(ROOT / "configs/yolov1/train.yaml"))
    parser.add_argument("--dataset-config", default=None, help="Override the dataset config stored in the recipe")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    dataset_config_path = args.dataset_config or str(ROOT / recipe["dataset_config"])
    config = load_dataset_config(dataset_config_path)
    input_size = args.input_size or int(recipe["input_size"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
        original_size = image.size
        image = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    model = build_model(recipe.get("model"), len(config["class_names"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    with torch.inference_mode():
        output = model(tensor)
        detections = decode_predictions(output, [original_size], args.confidence, args.nms_iou)[0]
    report = [
        {"xyxy": [round(float(value), 2) for value in box], "class_id": int(label), "class_name": config["class_names"][int(label)], "confidence": round(float(score), 6)}
        for box, label, score in zip(detections["boxes"].cpu(), detections["labels"].cpu(), detections["scores"].cpu())
    ]
    print(json.dumps({"image": str(Path(args.image).resolve()), "detections": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
