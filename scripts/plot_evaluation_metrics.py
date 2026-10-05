#!/usr/bin/env python3
"""Plot per-class AP from an evaluate_yolov1.py JSON report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_metrics(report_path: Path, output_prefix: Path) -> tuple[Path, Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    ap50 = metrics["AP50"]
    ap50_95 = metrics["AP50_95"]
    class_names = sorted(ap50, key=lambda name: ap50[name])
    y = np.arange(len(class_names))

    fig, ax = plt.subplots(figsize=(14, 10), constrained_layout=True)
    ax.barh(y + 0.18, [100 * ap50[name] for name in class_names], height=0.36, color="tab:blue", label="AP@0.50")
    ax.barh(
        y - 0.18,
        [100 * ap50_95[name] for name in class_names],
        height=0.36,
        color="tab:orange",
        label="AP@[0.50:0.95]",
    )
    ax.axvline(100 * metrics["mAP50"], color="tab:blue", linestyle="--", alpha=0.65)
    ax.axvline(100 * metrics["mAP50_95"], color="tab:orange", linestyle="--", alpha=0.65)
    ax.set(
        yticks=y,
        yticklabels=class_names,
        xlim=(0, 100),
        xlabel="Average precision (%)",
        title=(
            f"{report['dataset']} {report['split']} | "
            f"mAP@0.50 {100 * metrics['mAP50']:.2f}% | "
            f"mAP@[0.50:0.95] {100 * metrics['mAP50_95']:.2f}%"
        ),
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    for path in plot_evaluation_metrics(args.report, args.output_prefix):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
