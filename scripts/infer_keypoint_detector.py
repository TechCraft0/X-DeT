#!/usr/bin/env python3
"""Run CenterNet or CornerNet on an image or a directory of images."""
from __future__ import annotations

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
from x_yolo.data.dataset import IMAGE_SUFFIXES, load_dataset_config
from x_yolo.models.factory import build_model
from x_yolo.training.checkpoint import load_checkpoint


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Image file or directory")
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=None)
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
    config = load_dataset_config(dataset_config_path)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    model = build_model(recipe["model"], len(config["class_names"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    if recipe["model"]["architecture"] == "centernet":
        from x_yolo.models.centernet.postprocess import decode_predictions
    elif recipe["model"]["architecture"] == "cornernet":
        from x_yolo.models.cornernet.postprocess import decode_predictions
    else:
        raise ValueError(f"Unsupported keypoint architecture: {recipe['model']['architecture']}")
    confidence = float(recipe["evaluation"]["confidence_threshold"] if args.confidence is None else args.confidence)
    decoder = partial(decode_predictions, **recipe.get("decoder", {}))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    reports = []
    input_size = int(recipe["input_size"])
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as source:
                original = source.convert("RGB")
            resized = original.resize((input_size, input_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
            images = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
            detections = decoder(
                model(images), [original.size], confidence_threshold=confidence,
                nms_iou_threshold=float(recipe["evaluation"].get("nms_iou_threshold", 0.5)),
                max_detections=int(recipe["evaluation"].get("max_detections", 100)),
            )[0]
            boxes = detections["boxes"].cpu().tolist()
            labels = detections["labels"].cpu().tolist()
            scores = detections["scores"].cpu().tolist()
            annotated = _draw_boxes(original, boxes, labels, scores, config["class_names"])
            annotated.save(output_dir / f"{path.stem}_prediction.jpg", quality=92)
            reports.append({
                "image": str(path.resolve()),
                "detections": [
                    {"class_id": class_id, "class_name": config["class_names"][class_id],
                     "score": score, "xyxy": box}
                    for box, class_id, score in zip(boxes, labels, scores)
                ],
            })
    report = output_dir / "predictions.json"
    report.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(paths)} annotated image(s) and {report}")


if __name__ == "__main__":
    main()
