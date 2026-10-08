#!/usr/bin/env python3
"""Train CenterNet or CornerNet from a YAML recipe and VOC YOLO TXT labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x_yolo.training.keypoint_trainer import train_keypoint_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pretrained", type=Path, help="MMDetection detector checkpoint; required by the supplied recipes")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--max-steps", type=int, help="Bounded smoke run; stops after this many optimizer steps")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train_keypoint_detector(
        args.recipe, args.dataset_root, args.output,
        pretrained_checkpoint=args.pretrained, device_name=args.device,
        epochs_override=args.epochs, micro_batch_size_override=args.micro_batch_size,
        precision_override=args.precision, max_steps=args.max_steps, resume=args.resume,
    )


if __name__ == "__main__":
    main()
