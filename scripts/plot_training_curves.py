#!/usr/bin/env python3
"""Plot training losses, validation AP, learning rate, and ignored targets."""
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
import yaml


def read_metrics(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"No epoch metrics found in {path}")
    return rows


def plot_curves(run_dir: Path, output_prefix: Path | None = None) -> tuple[Path, Path]:
    metrics_path = run_dir / "metrics.jsonl"
    rows = read_metrics(metrics_path)
    output_prefix = output_prefix or run_dir / "training_curves"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / "run_config.yaml"
    run_config = yaml.safe_load(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    architecture = str(run_config.get("recipe", {}).get("model", {}).get("architecture", "yolov1"))
    architecture_label = {"yolov1": "YOLOv1", "yolov2": "YOLOv2"}.get(architecture, architecture)

    epochs = [int(row["epoch"]) for row in rows]
    validation_rows = [row for row in rows if isinstance(row.get("validation"), dict)]
    validation_epochs = [int(row["epoch"]) for row in validation_rows]
    class_names = list(validation_rows[0]["validation"]["AP50"]) if validation_rows else []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"{architecture_label} training history", fontsize=16)

    loss_axis = axes[0, 0]
    for key, label in (
        ("train_total", "total"),
        ("train_coordinate", "coordinate"),
        ("train_object_confidence", "object confidence"),
        ("train_no_object_confidence", "no-object confidence"),
        ("train_giou", "GIoU localization"),
        ("train_classification", "classification"),
    ):
        values = [float(row[key]) for row in rows if key in row]
        value_epochs = [int(row["epoch"]) for row in rows if key in row]
        if values:
            loss_axis.plot(value_epochs, values, marker="o", markersize=3, label=label)
    loss_axis.set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(fontsize=8)

    mean_axis, ap50_axis, ap_range_axis = axes[0, 1], axes[0, 2], axes[1, 0]
    for axis, metric_key, title, mean_key in (
        (ap50_axis, "AP50", "Per-class AP@0.50", "mAP50"),
        (ap_range_axis, "AP50_95", "Per-class AP@[0.50:0.95]", "mAP50_95"),
    ):
        for class_name in class_names:
            values = [float(row["validation"][metric_key][class_name]) for row in validation_rows]
            if values:
                axis.plot(validation_epochs, values, marker="o", markersize=2, linewidth=1, label=class_name)
        observed_values = [
            float(row["validation"][metric_key][class_name])
            for row in validation_rows
            for class_name in class_names
        ]
        y_max = min(1.0, max(1e-4, max(observed_values, default=0.0) * 1.15))
        axis.set(title=f"{title} (y max {y_max:.2g})", xlabel="Epoch", ylabel="AP", ylim=(0, y_max))
        axis.grid(alpha=0.25)
        if validation_rows:
            axis.legend(fontsize=6, ncol=3, loc="upper right")
        else:
            axis.text(0.5, 0.5, "No validation records yet", ha="center", va="center", transform=axis.transAxes)

    for key, label, color in (
        ("mAP50", "mAP@0.50", "tab:blue"),
        ("mAP50_95", "mAP@[0.50:0.95]", "tab:orange"),
    ):
        means = [float(row["validation"][key]) for row in validation_rows]
        if means:
            mean_axis.plot(validation_epochs, means, marker="o", label=label, color=color)
    mean_values = [float(row["validation"][key]) for row in validation_rows for key in ("mAP50", "mAP50_95")]
    mean_y_max = min(1.0, max(1e-4, max(mean_values, default=0.0) * 1.15))
    mean_axis.set(title=f"Mean validation AP (y max {mean_y_max:.2g})", xlabel="Epoch", ylabel="mAP", ylim=(0, mean_y_max))
    mean_axis.grid(alpha=0.25)
    if validation_rows:
        mean_axis.legend()
    else:
        mean_axis.text(0.5, 0.5, "No validation records yet", ha="center", va="center", transform=mean_axis.transAxes)

    aux_axis = axes[1, 1]
    learning_rates = [float(row["learning_rate"]) for row in rows]
    has_backbone_lr = any("backbone_learning_rate" in row for row in rows)
    head_label = "detector learning rate" if has_backbone_lr else "learning rate"
    aux_axis.plot(epochs, learning_rates, color="tab:purple", marker="o", markersize=3, label=head_label)
    if has_backbone_lr:
        backbone_lr_epochs = [int(row["epoch"]) for row in rows if "backbone_learning_rate" in row]
        backbone_lrs = [float(row["backbone_learning_rate"]) for row in rows if "backbone_learning_rate" in row]
        aux_axis.plot(
            backbone_lr_epochs,
            backbone_lrs,
            color="tab:green",
            marker="o",
            markersize=3,
            label="backbone learning rate",
        )
    aux_axis.set(title="Learning rate and ignored targets", xlabel="Epoch", ylabel="Learning rate")
    aux_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    aux_axis.grid(alpha=0.25)
    ignored_axis = aux_axis.twinx()
    ignored = [int(row.get("ignored_ground_truths", 0)) for row in rows]
    ignored_axis.plot(epochs, ignored, color="tab:orange", marker="o", markersize=3, alpha=0.8, label="ignored labels")
    ignored_axis.set_ylabel("Ignored labels per epoch")
    if (run_dir / "stage1_run_config.yaml").is_file():
        aux_axis.axvline(30.5, color="black", linestyle="--", alpha=0.65, label="resume after epoch 30")
    first_unfrozen = next(
        (int(row["epoch"]) for row in rows if row.get("backbone_trainable") is True),
        None,
    )
    if first_unfrozen is not None and any(row.get("backbone_trainable") is False for row in rows):
        aux_axis.axvline(
            first_unfrozen - 0.5,
            color="tab:red",
            linestyle="--",
            alpha=0.7,
            label=f"unfreeze backbone before epoch {first_unfrozen}",
        )
    lines, labels = aux_axis.get_legend_handles_labels()
    other_lines, other_labels = ignored_axis.get_legend_handles_labels()
    aux_axis.legend(lines + other_lines, labels + other_labels, fontsize=8, loc="best")
    axes[1, 2].axis("off")

    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Directory containing metrics.jsonl")
    parser.add_argument("--output-prefix", type=Path, default=None, help="Output path without .png/.svg")
    args = parser.parse_args()
    png_path, svg_path = plot_curves(args.run_dir.resolve(), args.output_prefix)
    print(f"Saved: {png_path}\nSaved: {svg_path}")


if __name__ == "__main__":
    main()
