#!/usr/bin/env python3
"""Convert a PASCAL VOC 2007 devkit tree into this project's YOLO TXT layout."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
CLASS_TO_ID = {name: index for index, name in enumerate(VOC_CLASSES)}


def _split_ids(voc_root: Path, split: str) -> list[str]:
    path = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"VOC split file does not exist: {path}")
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _convert_one(voc_root: Path, output_root: Path, image_id: str, split: str, include_difficult: bool) -> dict[str, object]:
    image_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
    annotation_path = voc_root / "Annotations" / f"{image_id}.xml"
    if not image_path.is_file() or not annotation_path.is_file():
        raise FileNotFoundError(f"Missing VOC image or annotation for {image_id}")
    document = ET.parse(annotation_path).getroot()
    size = document.find("size")
    if size is None:
        raise ValueError(f"VOC annotation has no <size>: {annotation_path}")
    width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {annotation_path}: {width}x{height}")

    labels: list[str] = []
    class_counts: Counter[str] = Counter()
    difficult_count = unknown_count = 0
    for obj in document.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASS_TO_ID:
            unknown_count += 1
            continue
        if int(obj.findtext("difficult", "0")) and not include_difficult:
            difficult_count += 1
            continue
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Object without <bndbox> in {annotation_path}")
        # VOC coordinates are 1-based inclusive. Convert to normalized YOLO xywh.
        x1 = max(0.0, float(box.findtext("xmin", "0")) - 1.0)
        y1 = max(0.0, float(box.findtext("ymin", "0")) - 1.0)
        x2 = min(float(width), float(box.findtext("xmax", "0")))
        y2 = min(float(height), float(box.findtext("ymax", "0")))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid VOC box in {annotation_path}: {(x1, y1, x2, y2)}")
        xc, yc = (x1 + x2) / (2 * width), (y1 + y2) / (2 * height)
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        labels.append(f"{CLASS_TO_ID[name]} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
        class_counts[name] += 1

    image_out = output_root / "images" / split / image_path.name
    label_out = output_root / "labels" / split / f"{image_id}.txt"
    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_out)
    label_out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
    return {
        "image": image_id,
        "objects": len(labels),
        "difficult_objects_omitted": difficult_count,
        "unknown_objects_omitted": unknown_count,
        "classes": dict(class_counts),
    }


def convert_voc(voc_root: Path, output_root: Path, splits: dict[str, str], include_difficult: bool = False) -> dict[str, object]:
    voc_root = voc_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not (voc_root / "JPEGImages").is_dir() or not (voc_root / "Annotations").is_dir():
        raise FileNotFoundError(f"Expected VOC2007 JPEGImages/ and Annotations/ under {voc_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new path: {output_root}")

    report: dict[str, object] = {
        "source": "PASCAL VOC 2007",
        "source_root": str(voc_root),
        "output_root": str(output_root),
        "class_names": list(VOC_CLASSES),
        "difficult_policy": "included" if include_difficult else "omitted",
        "splits": {},
    }
    split_reports: dict[str, object] = {}
    for output_split, source_split in splits.items():
        ids = _split_ids(voc_root, source_split)
        classes: Counter[str] = Counter()
        boxes = difficult = unknown = empty = 0
        for image_id in ids:
            result = _convert_one(voc_root, output_root, image_id, output_split, include_difficult)
            boxes += int(result["objects"])
            difficult += int(result["difficult_objects_omitted"])
            unknown += int(result["unknown_objects_omitted"])
            empty += int(result["objects"] == 0)
            classes.update(result["classes"])
        split_reports[output_split] = {
            "source_split": source_split,
            "images": len(ids),
            "objects": boxes,
            "objects_by_class": {name: classes[name] for name in VOC_CLASSES},
            "empty_label_files": empty,
            "difficult_objects_omitted": difficult,
            "unknown_objects_omitted": unknown,
        }
    report["splits"] = split_reports
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "conversion_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voc-root", required=True, type=Path, help="VOC2007 directory containing JPEGImages, Annotations, and ImageSets")
    parser.add_argument("--output-root", required=True, type=Path, help="New output directory; non-empty directories are never overwritten")
    parser.add_argument("--train-split", default="train", help="Source VOC split used for training")
    parser.add_argument("--val-split", default="val", help="Source VOC split used for validation")
    parser.add_argument("--test-split", default="test", help="Source VOC split copied for one final held-out evaluation")
    parser.add_argument("--include-difficult", action="store_true", help="Include difficult objects (default omits and reports them)")
    args = parser.parse_args()
    report = convert_voc(
        args.voc_root,
        args.output_root,
        {"train": args.train_split, "val": args.val_split, "test": args.test_split},
        include_difficult=args.include_difficult,
    )
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
