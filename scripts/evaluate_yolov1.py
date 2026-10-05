#!/usr/bin/env python3
"""Evaluate a YOLOv1 checkpoint on the configured validation split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from x_yolo.data.dataset import YoloTxtDetectionDataset, detection_collate, load_dataset_config
from x_yolo.evaluation.metrics import evaluate_model
from x_yolo.models.factory import build_model
from x_yolo.training.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--recipe", default=str(ROOT / "configs/yolov1/train.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", help="Configured split to evaluate (val or test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-detections",
        type=int,
        default=None,
        help="Override the recipe's per-image detection cap",
    )
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    max_detections = args.max_detections
    if max_detections is None:
        max_detections = int(recipe["evaluation"].get("max_detections", 100))
    if max_detections < 1:
        parser.error("--max-detections must be at least 1")
    dataset_config = load_dataset_config(ROOT / recipe["dataset_config"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    dataset = YoloTxtDetectionDataset(dataset_config, args.dataset_root, args.split, recipe["input_size"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=recipe.get("num_workers", 0), collate_fn=detection_collate)
    model = build_model(recipe.get("model"), len(dataset_config["class_names"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate_model(
        model, loader, dataset_config["class_names"], device,
        confidence_threshold=recipe["evaluation"]["confidence_threshold"],
        nms_iou_threshold=recipe["evaluation"]["nms_iou_threshold"],
        max_detections=max_detections,
    )
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": dataset_config.get("name"),
        "split": args.split,
        "metrics": metrics,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
