#!/usr/bin/env python3
"""Save YOLO TXT class, box-size, center-density, and grid analyses."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from x_yolo.data.dataset import YoloTxtDetectionDataset, load_dataset_config, parse_yolo_label, scan_dataset


def analyze_dataset(config_path: Path, dataset_root: Path, output_prefix: Path, grid_size: int) -> tuple[Path, Path, Path]:
    config = load_dataset_config(config_path)
    scan = scan_dataset(config, dataset_root, grid_size)
    distributions: dict[str, dict[str, np.ndarray]] = {}
    for split in config["splits"]:
        dataset = YoloTxtDetectionDataset(config, dataset_root, split)
        all_boxes: list[np.ndarray] = []
        objects_per_image: list[int] = []
        for label_path in dataset.label_paths:
            boxes = parse_yolo_label(label_path, len(config["class_names"]))
            objects_per_image.append(len(boxes))
            if len(boxes):
                all_boxes.append(boxes[:, 1:5].numpy())
        xywh = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), dtype=np.float32)
        distributions[split] = {
            "xywh": xywh,
            "objects_per_image": np.asarray(objects_per_image, dtype=np.int32),
        }
        areas = xywh[:, 2] * xywh[:, 3]
        scan["splits"][split]["normalized_box_area_percentiles"] = {
            str(percentile): float(np.percentile(areas, percentile)) if len(areas) else 0.0
            for percentile in (10, 50, 90)
        }

    split_names = list(config["splits"])
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    fig, axes = plt.subplots(2, 3, figsize=(21, 13), constrained_layout=True)
    fig.suptitle(f"{config.get('name', dataset_root.name)} dataset analysis ({grid_size}x{grid_size} center grid)", fontsize=17)

    # Keep every class visible; absolute counts expose long-tail classes.
    y = np.arange(len(config["class_names"]))
    bar_height = 0.78 / len(split_names)
    for split_index, split in enumerate(split_names):
        counts = scan["splits"][split]["boxes_by_class"]
        values = [counts.get(name, 0) for name in config["class_names"]]
        axes[0, 0].barh(
            y + (split_index - (len(split_names) - 1) / 2) * bar_height,
            values,
            height=bar_height,
            label=split,
            color=colors[split_index % len(colors)],
        )
    axes[0, 0].set(yticks=y, yticklabels=config["class_names"], xlabel="Labeled objects", title="Objects by class and split")
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis="x", alpha=0.25)
    axes[0, 0].legend()

    bins = np.arange(-0.5, 21.5, 1)
    for split_index, split in enumerate(split_names):
        counts = np.minimum(distributions[split]["objects_per_image"], 20)
        axes[0, 1].hist(
            counts,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=split,
            color=colors[split_index % len(colors)],
        )
    axes[0, 1].set(xlabel="Objects per image (20 includes 20+)", ylabel="Fraction of images", title="Image density")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.25)

    train_split = "train" if "train" in distributions else split_names[0]
    train_boxes = distributions[train_split]["xywh"]
    if len(train_boxes):
        width_height, _, _, mesh = axes[0, 2].hist2d(
            train_boxes[:, 2], train_boxes[:, 3], bins=35, range=[[0, 1], [0, 1]], norm=LogNorm(), cmap="viridis"
        )
        fig.colorbar(mesh, ax=axes[0, 2], label="Boxes per bin")
    axes[0, 2].set(xlabel="Normalized width", ylabel="Normalized height", title=f"Box sizes ({train_split})")

    if len(train_boxes):
        center_density, _, _ = np.histogram2d(
            train_boxes[:, 1], train_boxes[:, 0], bins=28, range=[[0, 1], [0, 1]]
        )
        mesh = axes[1, 0].imshow(center_density, origin="lower", extent=(0, 1, 0, 1), cmap="magma", aspect="equal")
        fig.colorbar(mesh, ax=axes[1, 0], label="Object centers per bin")
        for boundary in np.arange(1, grid_size) / grid_size:
            axes[1, 0].axvline(boundary, color="white", alpha=0.3, linewidth=0.7)
            axes[1, 0].axhline(boundary, color="white", alpha=0.3, linewidth=0.7)
    axes[1, 0].set(xlabel="Normalized x center", ylabel="Normalized y center", title=f"Object centers and {grid_size}x{grid_size} grid ({train_split})")

    area_bins = np.logspace(-3, 2, 55)
    for split_index, split in enumerate(split_names):
        boxes = distributions[split]["xywh"]
        areas_percent = 100 * boxes[:, 2] * boxes[:, 3]
        axes[1, 1].hist(
            areas_percent[areas_percent > 0],
            bins=area_bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=split,
            color=colors[split_index % len(colors)],
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set(xlabel="Box area / image area (%)", ylabel="Density", title="Box area distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    collision_image_rates = [
        100 * scan["splits"][split]["images_with_grid_collisions"] / max(scan["splits"][split]["images"], 1)
        for split in split_names
    ]
    ignored_box_rates = [
        100 * scan["splits"][split]["extra_objects_in_colliding_cells"] / max(scan["splits"][split]["boxes"], 1)
        for split in split_names
    ]
    x = np.arange(len(split_names))
    axes[1, 2].bar(x - 0.18, collision_image_rates, width=0.36, label="Images with shared cell centers")
    axes[1, 2].bar(x + 0.18, ignored_box_rates, width=0.36, label="Extra boxes per occupied cell")
    axes[1, 2].set(xticks=x, xticklabels=split_names, ylabel="Percent (%)", title=f"{grid_size}x{grid_size} center collisions")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(axis="y", alpha=0.25)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    json_path = output_prefix.with_suffix(".json")
    fig.savefig(png_path, dpi=160)
    fig.savefig(svg_path)
    plt.close(fig)
    json_path.write_text(json.dumps(scan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return png_path, svg_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--grid-size", type=int, default=7)
    args = parser.parse_args()
    paths = analyze_dataset(args.config, args.dataset_root, args.output_prefix, args.grid_size)
    for path in paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
