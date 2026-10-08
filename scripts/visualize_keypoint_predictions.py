#!/usr/bin/env python3
"""Save CenterNet/CornerNet VOC ground-truth comparisons and a contact sheet."""
from __future__ import annotations

import json
import random
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from scripts.visualize_yolov1_predictions import (
    _add_header, _draw_boxes, _ground_truth_pixels, _select_one_per_class,
)
from x_yolo.data.dataset import YoloTxtDetectionDataset, load_dataset_config, parse_yolo_label
from x_yolo.models.factory import build_model
from x_yolo.training.checkpoint import load_checkpoint


def visualize_predictions(recipe_path: Path, dataset_root: Path, checkpoint_path: Path,
                          output_dir: Path, split: str, confidence: float,
                          seed: int, device: torch.device) -> tuple[Path, Path]:
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    dataset_config_path = Path(recipe["dataset_config"])
    if not dataset_config_path.is_absolute():
        dataset_config_path = ROOT / dataset_config_path
    config = load_dataset_config(dataset_config_path)
    dataset = YoloTxtDetectionDataset(config, dataset_root, split, int(recipe["input_size"]))
    selected = _select_one_per_class(dataset, seed)
    if not selected:
        raise ValueError(f"No labeled images found in split {split!r}")
    model = build_model(recipe["model"], len(config["class_names"])).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    architecture = recipe["model"]["architecture"]
    if architecture == "centernet":
        from x_yolo.models.centernet.postprocess import decode_predictions
    elif architecture == "cornernet":
        from x_yolo.models.cornernet.postprocess import decode_predictions
    else:
        raise ValueError(f"Unsupported keypoint detector: {architecture}")
    decoder = partial(decode_predictions, **recipe.get("decoder", {}))
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    examples: list[tuple[str, Image.Image]] = []
    manifest: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()), "dataset": config.get("name", dataset_root.name),
        "split": split, "architecture": architecture,
        "selection": "one deterministic unique image per class", "seed": seed,
        "confidence_threshold": confidence, "examples": [],
    }
    with torch.inference_mode():
        for selected_class, index in selected:
            image_path, label_path = dataset.samples[index]
            labels = parse_yolo_label(label_path, len(config["class_names"]))
            with Image.open(image_path) as source:
                original = source.convert("RGB")
            resized = original.resize((dataset.input_size, dataset.input_size), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
            predictions = decoder(
                model(tensor), [original.size], confidence_threshold=confidence,
                nms_iou_threshold=float(recipe["evaluation"].get("nms_iou_threshold", 0.5)),
                max_detections=int(recipe["evaluation"].get("max_detections", 100)),
            )[0]
            pred_boxes = predictions["boxes"].cpu().tolist()
            pred_labels = predictions["labels"].cpu().tolist()
            pred_scores = predictions["scores"].cpu().tolist()
            gt_boxes, gt_labels = _ground_truth_pixels(labels, original.size)
            gt_panel = _add_header(
                _draw_boxes(original, gt_boxes, gt_labels, None, config["class_names"]),
                f"Ground truth: {len(gt_boxes)} objects",
            )
            pred_panel = _add_header(
                _draw_boxes(original, pred_boxes, pred_labels, pred_scores, config["class_names"]),
                f"Prediction: score >= {confidence:g} ({len(pred_boxes)} boxes)",
            )
            comparison = Image.new("RGB", (original.width * 2, original.height + 30), "white")
            comparison.paste(gt_panel, (0, 0))
            comparison.paste(pred_panel, (original.width, 0))
            name = f"{selected_class:02d}_{config['class_names'][selected_class]}_{image_path.stem}.jpg"
            comparison.save(output_dir / name, quality=92)
            examples.append((config["class_names"][selected_class], comparison))
            manifest["examples"].append({
                "selected_class": config["class_names"][selected_class],
                "source_image": image_path.name, "comparison_image": name,
                "ground_truth": [
                    {"class_id": class_id, "class_name": config["class_names"][class_id], "xyxy": box}
                    for box, class_id in zip(gt_boxes, gt_labels)
                ],
                "predictions": [
                    {"class_id": class_id, "class_name": config["class_names"][class_id],
                     "score": score, "xyxy": box}
                    for box, class_id, score in zip(pred_boxes, pred_labels, pred_scores)
                ],
            })

    cell_w, cell_h, columns = 560, 280, 4
    contact_sheet = Image.new("RGB", (columns * cell_w, ((len(examples) + columns - 1) // columns) * cell_h),
                              (245, 245, 245))
    draw = ImageDraw.Draw(contact_sheet)
    for index, (class_name, panel) in enumerate(examples):
        column, row = index % columns, index // columns
        thumbnail = panel.copy()
        thumbnail.thumbnail((cell_w - 12, cell_h - 38), Image.Resampling.LANCZOS)
        left = column * cell_w + (cell_w - thumbnail.width) // 2
        top = row * cell_h + 30 + (cell_h - 38 - thumbnail.height) // 2
        contact_sheet.paste(thumbnail, (left, top))
        draw.text((column * cell_w + 8, row * cell_h + 6), f"Selected GT class: {class_name}", fill="black")
    sheet_path = output_dir / "contact_sheet.jpg"
    manifest_path = output_dir / "manifest.json"
    contact_sheet.save(sheet_path, quality=93)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sheet_path, manifest_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    confidence = args.confidence
    if confidence is None:
        confidence = float(recipe["evaluation"]["confidence_threshold"])
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    sheet, manifest = visualize_predictions(
        args.recipe, args.dataset_root, args.checkpoint, args.output_dir,
        args.split, confidence, args.seed, device,
    )
    print(f"Saved: {sheet}\nSaved: {manifest}")


if __name__ == "__main__":
    main()
