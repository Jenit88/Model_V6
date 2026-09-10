#!/usr/bin/env python3
"""Build a YOLO dataset that is provably the same data Model V6.2 trained on.

Two problems make this necessary rather than just pointing Ultralytics at
Split_Data:

**The shipped data.yaml is off by one.** The label files use class ids 1-4, but
that file names classes 0-3. Trained against it, YOLO would learn id 1 as
"Rectangle_concave", id 2 as "circle", id 3 as "circle_full", and choke on id 4.
Model V6.2 maps 1->Rectangle, 2->Rectangle_concave, 3->circle, 4->circle_full,
and the class frequencies confirm that is the correct reading: id 4 is the most
common (circle_full, 52.6% in the audit) and id 2 the rarest (Rectangle_concave,
0.47%). This writes labels remapped to 0-3 with names that match.

**The old run's dataset is gone.** It referenced Split_Data_yolo, which no
longer exists, so there is no way to prove the previous YOLO baseline saw the
same images as V6.2. Rebuilding from Split_Data removes that doubt: both models
then demonstrably train, validate and test on identical files.

Images are copied to the WSL ext4 disk rather than read over /mnt/c. The
previous YOLO run averaged 74.9 min/epoch, and reading 4,680 large PNGs per
epoch across the 9p mount is a strong candidate for why.

    ./run.sh build-yolo-dataset
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE = Path(os.environ.get(
    "PCB_DATASET_ROOT",
    "/mnt/c/Users/u117134/Desktop/Data_preprocessing/1000_images/Split_Data",
))
TARGET = Path(os.environ.get("PCB_YOLO_DATASET", str(Path.home() / "data/yolo_fair")))

# Label id in the files -> contiguous YOLO index, and the name for that index.
ID_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
NAMES = {0: "Rectangle", 1: "Rectangle_concave", 2: "circle", 3: "circle_full"}
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def convert_label(source: Path, destination: Path) -> tuple[int, int]:
    kept = dropped = 0
    lines = []
    for raw in source.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        try:
            original = int(parts[0])
        except ValueError:
            dropped += 1
            continue
        if original not in ID_MAP:
            dropped += 1
            continue
        lines.append(" ".join([str(ID_MAP[original])] + parts[1:]))
        kept += 1
    destination.write_text("\n".join(lines) + ("\n" if lines else ""))
    return kept, dropped


def main() -> None:
    if not SOURCE.is_dir():
        sys.exit(f"source not found: {SOURCE}")
    print(f"source: {SOURCE}")
    print(f"target: {TARGET}\n")

    total_kept = total_dropped = total_images = 0
    for split in SPLITS:
        image_source = SOURCE / "images" / split
        label_source = SOURCE / "labels" / split
        image_target = TARGET / "images" / split
        label_target = TARGET / "labels" / split
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)

        images = sorted(p for p in image_source.iterdir()
                        if p.suffix.lower() in IMAGE_SUFFIXES)
        started = time.time()

        def copy_one(path: Path) -> None:
            destination = image_target / path.name
            if not destination.exists() or destination.stat().st_size != path.stat().st_size:
                shutil.copy2(path, destination)

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(copy_one, images))

        kept = dropped = 0
        for path in images:
            label = label_source / (path.stem + ".txt")
            if label.is_file():
                a, b = convert_label(label, label_target / label.name)
                kept += a
                dropped += b
            else:
                (label_target / (path.stem + ".txt")).write_text("")

        total_kept += kept
        total_dropped += dropped
        total_images += len(images)
        print(f"  {split:<5} {len(images):>5,} images  {kept:>7,} polygons"
              f"{'  ' + str(dropped) + ' dropped' if dropped else ''}"
              f"   {time.time() - started:.0f}s")

    yaml = TARGET / "data.yaml"
    yaml.write_text(
        f"# Built by build_yolo_dataset.py from {SOURCE}\n"
        f"# Class ids remapped 1,2,3,4 -> 0,1,2,3 so they match these names.\n"
        f"# The shipped Split_Data/data.yaml is off by one and must not be used.\n"
        f"path: {TARGET}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in NAMES.items())
    )
    print(f"\n{total_images:,} images, {total_kept:,} polygons"
          f"{', ' + str(total_dropped) + ' dropped' if total_dropped else ''}")
    print(f"wrote {yaml}")
    print(yaml.read_text())


if __name__ == "__main__":
    main()
