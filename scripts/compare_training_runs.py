#!/usr/bin/env python3
"""Plot comparable YOLOv1 training runs on shared epoch axes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_run(specification: str) -> tuple[str, list[dict]]:
    if "=" not in specification:
        raise ValueError(f"Expected LABEL=RUN_DIR, got {specification!r}")
    label, directory = specification.split("=", 1)
    if not label or not directory:
        raise ValueError(f"Expected LABEL=RUN_DIR, got {specification!r}")
    path = Path(directory) / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics in {path}")
    return label, rows


def plot_comparison(runs: list[tuple[str, list[dict]]], output: Path, max_epoch: int | None) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "train_total", "Training loss", "Loss"),
        (axes[0, 1], "mAP50", "Validation mAP@0.50", "mAP"),
        (axes[1, 0], "mAP50_95", "Validation mAP@[0.50:0.95]", "mAP"),
        (axes[1, 1], "learning_rate", "Detector learning rate", "Learning rate"),
    )
    for label, original_rows in runs:
        rows = [row for row in original_rows if max_epoch is None or int(row["epoch"]) <= max_epoch]
        if not rows:
            raise ValueError(f"No epochs within limit for {label}")
        for axis, key, _title, _unit in panels:
            points = [
                (int(row["epoch"]), float(row["validation"][key]))
                if key.startswith("mAP") else (int(row["epoch"]), float(row[key]))
                for row in rows
                if ("validation" in row if key.startswith("mAP") else key in row)
            ]
            if points:
                epochs, values = zip(*points)
                axis.plot(epochs, values, marker="o" if key.startswith("mAP") else None,
                          markersize=3, label=label)
    for axis, _key, title, unit in panels:
        axis.set(title=title, xlabel="Epoch", ylabel=unit)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1, 1].set_yscale("log")
    for axis in (axes[0, 1], axes[1, 0]):
        axis.set_ylim(bottom=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=RUN_DIR")
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("Pass at least two --run entries")
    plot_comparison([read_run(specification) for specification in args.run], args.output, args.max_epoch)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
