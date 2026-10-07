#!/usr/bin/env python3
"""Train the educational YOLOv3 implementation on a YOLO TXT dataset."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x_yolo.training.trainer import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root containing images/train, labels/train, etc.")
    parser.add_argument("--recipe", default=str(ROOT / "configs/yolov3/voc0712.yaml"))
    parser.add_argument("--output", default=None, help="Run directory; defaults to runs/yolov3/<timestamp>")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Bounded smoke run; not a convergence result")
    parser.add_argument("--resume", default=None, help="Resume from last.pt or another project checkpoint")
    parser.add_argument(
        "--pretrained-darknet53",
        default=None,
        help="Explicit Darknet-53 .conv.74 weights; never downloaded automatically",
    )
    args = parser.parse_args()
    output = Path(args.output) if args.output else ROOT / "runs/yolov3" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = train_detector(
        args.recipe,
        args.dataset_root,
        output,
        device_name=args.device,
        epochs_override=args.epochs,
        micro_batch_size_override=args.micro_batch_size,
        precision_override=args.precision,
        max_steps=args.max_steps,
        resume=args.resume,
        backbone_checkpoint=args.pretrained_darknet53,
    )
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
