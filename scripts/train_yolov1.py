#!/usr/bin/env python3
"""Train the educational YOLOv1 implementation on a YOLO TXT dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x_yolo.training.trainer import train_yolov1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Root containing images/train, labels/train, etc.")
    parser.add_argument("--recipe", default=str(ROOT / "configs/yolov1/train.yaml"))
    parser.add_argument("--output", default=None, help="Run directory; defaults to runs/yolov1/<timestamp>")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    parser.add_argument("--epochs", type=int, default=None, help="Override recipe epochs")
    parser.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="Override physical batch size; it must divide the recipe's effective batch size",
    )
    parser.add_argument(
        "--precision", choices=("fp32", "bf16"), default=None,
        help="Override training forward precision; YOLOv1 loss remains FP32",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many batches for a bounded trial")
    parser.add_argument("--resume", default=None, help="Resume from last.pt or another project checkpoint")
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="Explicit local pretrained backbone weights for variants that require them; never downloaded automatically",
    )
    args = parser.parse_args()
    if args.output:
        output = Path(args.output)
    else:
        from datetime import datetime
        output = ROOT / "runs/yolov1" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = train_yolov1(
        args.recipe,
        args.dataset_root,
        output,
        device_name=args.device,
        epochs_override=args.epochs,
        micro_batch_size_override=args.micro_batch_size,
        precision_override=args.precision,
        max_steps=args.max_steps,
        resume=args.resume,
        backbone_checkpoint=args.backbone_checkpoint,
    )
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
