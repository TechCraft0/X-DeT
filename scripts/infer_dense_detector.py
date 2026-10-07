#!/usr/bin/env python3
"""Run one FCOS or RTMDet checkpoint on an image or directory of images."""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image

from scripts.visualize_yolov1_predictions import _draw_boxes
from x_yolo.data.dataset import IMAGE_SUFFIXES
from x_yolo.models.factory import build_model
from x_yolo.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image file or directory containing images")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=None)
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    paths = [args.input] if args.input.is_file() else sorted(
        path for path in args.input.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found in {args.input}")
    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    dataset_config_path = Path(recipe["dataset_config"])
    if not dataset_config_path.is_absolute():
        dataset_config_path = ROOT / dataset_config_path
    dataset_config = yaml.safe_load(dataset_config_path.read_text(encoding="utf-8"))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    model = build_model(recipe["model"], len(dataset_config["class_names"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    architecture = recipe["model"]["architecture"]
    if architecture == "fcos":
        from x_yolo.models.fcos.postprocess import decode_predictions
    elif architecture == "rtmdet":
        from x_yolo.models.rtmdet.postprocess import decode_predictions
    else:
        raise ValueError(f"Unsupported dense detector: {architecture}")
    nms_iou = float(args.nms_iou if args.nms_iou is not None
                    else recipe["evaluation"]["nms_iou_threshold"])
    decoder = partial(decode_predictions, strides=model.strides)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    reports = []
    input_size = int(recipe["input_size"])
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            resized = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
            outputs = model(tensor)
            detection = decoder(
                outputs, [image.size], confidence_threshold=args.confidence,
                nms_iou_threshold=nms_iou,
                max_detections=int(recipe["evaluation"].get("max_detections", 100)),
            )[0]
            boxes = detection["boxes"].cpu().tolist()
            labels = detection["labels"].cpu().tolist()
            scores = detection["scores"].cpu().tolist()
            annotated = _draw_boxes(image, boxes, labels, scores, dataset_config["class_names"])
            annotated.save(output_dir / f"{path.stem}_prediction.jpg", quality=92)
            reports.append({
                "image": str(path.resolve()),
                "detections": [
                    {"class_id": class_id, "class_name": dataset_config["class_names"][class_id],
                     "score": score, "xyxy": box}
                    for box, class_id, score in zip(boxes, labels, scores)
                ],
            })
    report_path = output_dir / "predictions.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(paths)} annotated image(s) and {report_path}")


if __name__ == "__main__":
    main()
