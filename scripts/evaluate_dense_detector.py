#!/usr/bin/env python3
"""Evaluate an FCOS or RTMDet checkpoint on a configured dataset split."""
from __future__ import annotations

import argparse
import json
import sys
from functools import partial
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
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    dataset_config_path = Path(recipe["dataset_config"])
    if not dataset_config_path.is_absolute():
        dataset_config_path = ROOT / dataset_config_path
    config = load_dataset_config(dataset_config_path)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dataset = YoloTxtDetectionDataset(config, args.dataset_root, args.split, int(recipe["input_size"]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=int(recipe.get("num_workers", 0)), collate_fn=detection_collate)
    model = build_model(recipe["model"], len(config["class_names"])).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    architecture = recipe["model"]["architecture"]
    if architecture == "fcos":
        from x_yolo.models.fcos.postprocess import decode_predictions
    elif architecture == "rtmdet":
        from x_yolo.models.rtmdet.postprocess import decode_predictions
    else:
        raise ValueError(f"Unsupported dense detector: {architecture}")
    metrics = evaluate_model(
        model, loader, config["class_names"], device,
        confidence_threshold=float(recipe["evaluation"]["confidence_threshold"]),
        nms_iou_threshold=float(recipe["evaluation"]["nms_iou_threshold"]),
        max_detections=int(recipe["evaluation"].get("max_detections", 100)),
        decoder=partial(decode_predictions, strides=model.strides),
    )
    report = {"checkpoint": str(args.checkpoint.resolve()), "dataset": config.get("name"),
              "split": args.split, "metrics": metrics}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
