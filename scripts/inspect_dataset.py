#!/usr/bin/env python3
"""Validate a configured YOLO TXT dataset and report model-grid collisions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from x_yolo.data.dataset import load_dataset_config, scan_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the dataset YAML config")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument("--grid-size", type=int, default=7, help="YOLOv1 output grid size")
    args = parser.parse_args()
    config = load_dataset_config(args.config)
    report = scan_dataset(config, args.root, args.grid_size)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
