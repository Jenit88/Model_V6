#!/usr/bin/env python3
"""Object scale per image, against the scale the model was trained at.

`measure_falt_scale.py` established that V6.2's failures on real boards track
object size: training's Rectangle p50 is 35.7 px and augmentation only ever
widened that by ~30%, so anything much larger is outside what the decoder has
seen. This asks the same question of these 12 boards, using V6.2's own
detections as the size estimate since there are no labels to measure.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "ro2"
TRAIN_RECTANGLE_P50 = 35.7  # dataset audit, train split


def main() -> None:
    with (OUT / "v62_records.pkl").open("rb") as handle:
        v62 = pickle.load(handle)
    with (OUT / "yolo_records.pkl").open("rb") as handle:
        yolo = pickle.load(handle)
    meta = json.loads((OUT / "frames_meta.json").read_text())

    print(f"{'#':<4}{'name':<24}{'n':>5}"
          f"{'p50':>8}{'p90':>8}{'max':>8}   {'xtrain':>7}   {'YOLO med conf':>13}")
    for record in meta:
        index = record["index"]
        image = v62["per_image"][index]
        instance_map = image["instance_map"]
        sizes = []
        for item in image["instances"]:
            if item["confidence"] < 0.5:
                continue
            area = int((instance_map == item["instance_id"]).sum())
            if area > 0:
                sizes.append(area ** 0.5)
        if not sizes:
            continue
        sizes = np.asarray(sizes)
        p50 = float(np.percentile(sizes, 50))
        scores = np.asarray(yolo["per_image"][index]["scores"], dtype=float)
        median_conf = float(np.median(scores)) if scores.size else float("nan")
        print(f"{index:<4}{record['name'][:22]:<24}{len(sizes):>5}"
              f"{p50:>8.1f}{np.percentile(sizes, 90):>8.1f}{sizes.max():>8.1f}"
              f"{p50 / TRAIN_RECTANGLE_P50:>7.2f}x{median_conf:>14.3f}")

    print(f"\ntrain Rectangle p50 = {TRAIN_RECTANGLE_P50} px (dataset audit)")
    print("xtrain is this image's detected p50 over that, on the same 512 frame.")


if __name__ == "__main__":
    main()
