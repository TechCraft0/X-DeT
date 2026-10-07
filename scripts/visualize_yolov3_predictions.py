#!/usr/bin/env python3
"""Save YOLOv3 ground-truth/prediction comparisons and a contact sheet."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.visualize_yolov1_predictions import main


if __name__ == "__main__":
    main()
