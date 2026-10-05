#!/usr/bin/env python3
"""Save deterministic test examples with ground truth and model predictions."""
from __future__ import annotations

import argparse
import colorsys
import json
import random
import sys
from pathlib import Path
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from x_yolo.data.dataset import YoloTxtDetectionDataset, load_dataset_config, parse_yolo_label
from x_yolo.models.factory import build_model
from x_yolo.models.yolov1.postprocess import decode_predictions as decode_yolov1
from x_yolo.training.checkpoint import load_checkpoint


def _color(class_id: int, num_classes: int) -> tuple[int, int, int]:
    red, green, blue = colorsys.hsv_to_rgb(class_id / max(num_classes, 1), 0.85, 0.95)
    return round(255 * red), round(255 * green), round(255 * blue)


def _font() -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), 15) if path.is_file() else ImageFont.load_default()


def _draw_boxes(
    source: Image.Image,
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float] | None,
    class_names: list[str],
) -> Image.Image:
    image = source.copy()
    draw = ImageDraw.Draw(image)
    font = _font()
    line_width = max(2, round(min(image.size) / 180))
    for index, (box, class_id) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = box
        x1, x2 = sorted((max(0, min(image.width, x1)), max(0, min(image.width, x2))))
        y1, y2 = sorted((max(0, min(image.height, y1)), max(0, min(image.height, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _color(class_id, len(class_names))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = class_names[class_id]
        if scores is not None:
            label += f" {scores[index]:.2f}"
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0] + 6
        text_height = text_box[3] - text_box[1] + 4
        text_x = min(max(0, round(x1)), max(0, image.width - text_width))
        text_y = max(0, round(y1) - text_height)
        draw.rectangle((text_x, text_y, text_x + text_width, text_y + text_height), fill=color)
        draw.text((text_x + 3, text_y + 1), label, fill="black", font=font)
    return image


def _add_header(image: Image.Image, title: str) -> Image.Image:
    header_height = 30
    panel = Image.new("RGB", (image.width, image.height + header_height), (30, 30, 30))
    panel.paste(image, (0, header_height))
    ImageDraw.Draw(panel).text((8, 6), title, fill="white", font=_font())
    return panel


def _ground_truth_pixels(rows: torch.Tensor, image_size: tuple[int, int]) -> tuple[list[list[float]], list[int]]:
    width, height = image_size
    boxes: list[list[float]] = []
    labels: list[int] = []
    for class_id, cx, cy, box_width, box_height in rows.tolist():
        boxes.append(
            [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ]
        )
        labels.append(int(class_id))
    return boxes, labels


def _select_one_per_class(dataset: YoloTxtDetectionDataset, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    candidates: list[list[int]] = [[] for _ in dataset.class_names]
    for index, label_path in enumerate(dataset.label_paths):
        rows = parse_yolo_label(label_path, dataset.num_classes)
        for class_id in {int(row[0]) for row in rows.tolist()}:
            candidates[class_id].append(index)
    selected: list[tuple[int, int]] = []
    used_images: set[int] = set()
    for class_id, image_indices in enumerate(candidates):
        rng.shuffle(image_indices)
        chosen = next((index for index in image_indices if index not in used_images), None)
        if chosen is not None:
            selected.append((class_id, chosen))
            used_images.add(chosen)
    return selected


def visualize_predictions(
    recipe_path: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    output_dir: Path,
    split: str,
    confidence: float,
    nms_iou: float,
    seed: int,
    device: torch.device,
) -> tuple[Path, Path]:
    recipe = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    config = load_dataset_config(ROOT / recipe["dataset_config"])
    dataset = YoloTxtDetectionDataset(config, dataset_root, split, int(recipe["input_size"]))
    selected = _select_one_per_class(dataset, seed)
    if not selected:
        raise ValueError(f"No labeled images found in split {split!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(recipe.get("model"), len(config["class_names"])).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    architecture = str(recipe.get("model", {}).get("architecture", "yolov1"))
    if architecture == "yolov2":
        from x_yolo.models.yolov2.postprocess import decode_predictions as decode_yolov2

        decoder = partial(decode_yolov2, anchors=model.anchors)
    else:
        decoder = decode_yolov1
    model.eval()
    manifest: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": config.get("name", dataset_root.name),
        "split": split,
        "architecture": architecture,
        "selection": "one randomly chosen unique image per class with a fixed seed",
        "seed": seed,
        "confidence_threshold": confidence,
        "nms_iou_threshold": nms_iou,
        "max_detections": 100,
        "examples": [],
    }
    thumbnails: list[tuple[str, Image.Image]] = []
    with torch.inference_mode():
        for selected_class, index in selected:
            image_path, label_path = dataset.samples[index]
            labels = parse_yolo_label(label_path, len(config["class_names"]))
            with Image.open(image_path) as source:
                original = source.convert("RGB")
            resized = original.resize((int(recipe["input_size"]), int(recipe["input_size"])), Image.Resampling.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)
            output = model(tensor)
            detections = decoder(
                output,
                [original.size],
                confidence_threshold=confidence,
                nms_iou_threshold=nms_iou,
                max_detections=100,
            )[0]
            pred_boxes = detections["boxes"].cpu().tolist()
            pred_labels = detections["labels"].cpu().tolist()
            pred_scores = detections["scores"].cpu().tolist()
            gt_boxes, gt_labels = _ground_truth_pixels(labels, original.size)

            gt_panel = _add_header(
                _draw_boxes(original, gt_boxes, gt_labels, None, dataset.class_names),
                f"Ground truth: {len(gt_boxes)} objects",
            )
            pred_image = _draw_boxes(original, pred_boxes, pred_labels, pred_scores, dataset.class_names)
            pred_panel = _add_header(pred_image, f"Prediction: score >= {confidence:g} ({len(pred_boxes)} boxes)")
            comparison = Image.new("RGB", (original.width * 2, original.height + 30), "white")
            comparison.paste(gt_panel, (0, 0))
            comparison.paste(pred_panel, (original.width, 0))
            output_name = f"{selected_class:02d}_{dataset.class_names[selected_class]}_{image_path.stem}.jpg"
            comparison.save(output_dir / output_name, quality=92)
            thumbnails.append((dataset.class_names[selected_class], comparison))
            manifest["examples"].append(
                {
                    "selected_class": dataset.class_names[selected_class],
                    "source_image": image_path.name,
                    "comparison_image": output_name,
                    "ground_truth": [
                        {"class_id": class_id, "class_name": dataset.class_names[class_id], "xyxy": box}
                        for box, class_id in zip(gt_boxes, gt_labels)
                    ],
                    "predictions": [
                        {"class_id": class_id, "class_name": dataset.class_names[class_id], "score": score, "xyxy": box}
                        for box, class_id, score in zip(pred_boxes, pred_labels, pred_scores)
                    ],
                }
            )

    cell_width, cell_height, columns = 560, 280, 4
    rows = (len(thumbnails) + columns - 1) // columns
    contact_sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (245, 245, 245))
    font = _font()
    for index, (class_name, panel) in enumerate(thumbnails):
        column, row = index % columns, index // columns
        thumbnail = panel.copy()
        thumbnail.thumbnail((cell_width - 12, cell_height - 38), Image.Resampling.LANCZOS)
        left = column * cell_width + (cell_width - thumbnail.width) // 2
        top = row * cell_height + 30 + (cell_height - 38 - thumbnail.height) // 2
        contact_sheet.paste(thumbnail, (left, top))
        ImageDraw.Draw(contact_sheet).text(
            (column * cell_width + 8, row * cell_height + 6),
            f"Selected GT class: {class_name}",
            fill="black",
            font=font,
        )
    sheet_path = output_dir / "contact_sheet.jpg"
    manifest_path = output_dir / "manifest.json"
    contact_sheet.save(sheet_path, quality=93)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sheet_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    sheet, manifest = visualize_predictions(
        args.recipe,
        args.dataset_root,
        args.checkpoint,
        args.output_dir,
        args.split,
        args.confidence,
        args.nms_iou,
        args.seed,
        device,
    )
    print(f"Saved: {sheet}\nSaved: {manifest}")


if __name__ == "__main__":
    main()
