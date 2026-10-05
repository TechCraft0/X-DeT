from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_voc import convert_voc


def test_voc_conversion_uses_voc_pixel_bounds_and_reports_difficult(tmp_path: Path) -> None:
    voc_root = tmp_path / "VOC2007"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "Annotations").mkdir()
    (voc_root / "ImageSets" / "Main").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (voc_root / "ImageSets" / "Main" / f"{split}.txt").write_text("sample\n", encoding="utf-8")
    (voc_root / "JPEGImages" / "sample.jpg").write_bytes(b"placeholder")
    (voc_root / "Annotations" / "sample.xml").write_text(
        """<annotation>
          <size><width>100</width><height>50</height><depth>3</depth></size>
          <object><name>car</name><difficult>0</difficult>
            <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>10</xmax><ymax>20</ymax></bndbox>
          </object>
          <object><name>person</name><difficult>1</difficult>
            <bndbox><xmin>20</xmin><ymin>10</ymin><xmax>40</xmax><ymax>40</ymax></bndbox>
          </object>
        </annotation>""",
        encoding="utf-8",
    )

    output_root = tmp_path / "converted"
    report = convert_voc(voc_root, output_root, {"train": "train", "val": "val", "test": "test"})

    label = (output_root / "labels" / "train" / "sample.txt").read_text(encoding="utf-8").strip().split()
    assert label == ["6", "0.05000000", "0.20000000", "0.10000000", "0.40000000"]
    assert report["splits"]["train"]["images"] == 1
    assert report["splits"]["train"]["objects"] == 1
    assert report["splits"]["train"]["difficult_objects_omitted"] == 1
    persisted = json.loads((output_root / "conversion_report.json").read_text(encoding="utf-8"))
    assert persisted["splits"]["test"]["source_split"] == "test"
