#!/usr/bin/env python3
"""Train the documented FCOS or RTMDet recipe on a YOLO-TXT dataset."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from x_yolo.training.dense_trainer import train_dense_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--pretrained", required=False, type=Path,
                        help="Explicit local ResNet-50 weights (FCOS) or RTMDet-Tiny COCO checkpoint")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Bounded code-path check; not a convergence result")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "runs" / args.recipe.parent.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = train_dense_detector(
        args.recipe, args.dataset_root, output,
        pretrained_checkpoint=args.pretrained,
        device_name=args.device,
        epochs_override=args.epochs,
        micro_batch_size_override=args.micro_batch_size,
        precision_override=args.precision,
        max_steps=args.max_steps,
        resume=args.resume,
    )
    print(f"Run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
