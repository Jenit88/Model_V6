#!/usr/bin/env python3
"""Measure detected object scale on the failure images against training scale.

The overlays show large objects shattering into strips while small ones survive
intact. The hypothesis is a scale-range failure: augmentation spans roughly
0.50x (ZOOM_OUT_MAX_PERCENT=50) up to fill_zoom x 1.12 (EXTRA_ZOOM_RANGE=
(1.06, 1.12)), while these images contain objects several times larger than
anything inside that window.

Diameters in instances.json are in ORIGINAL image pixels; the training medians
in the dataset audit are in the 512 letterboxed frame the model actually sees.
Everything is converted to that frame using each image's own letterbox scale,
or the comparison would be off by 3x on these 1536x1024 sources.

    PCB_PREDICTIONS=~/Models/Model_v6_2_scratch_rtx/predictions ./measure_falt_scale.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

# Medians from memory/dataset-split-data-audit.md, in the 512 letterboxed frame.
TRAIN_MEDIAN = {"Rectangle": 35.7, "circle": 33.8, "circle_full": 21.0}
# The widest object the augmentation ever produced, relative to labelled scale:
# zoom-to-fill for a 3:2 source is ~1.5, times EXTRA_ZOOM_RANGE's 1.12 ceiling.
AUGMENTATION_CEILING = 1.5 * 1.12


def main() -> None:
    root = os.path.expanduser(os.environ["PCB_PREDICTIONS"])
    paths = sorted(glob.glob(os.path.join(root, "*", "instances.json")))
    if not paths:
        raise SystemExit(f"no instances.json under {root}")

    by_class: dict[str, list[float]] = defaultdict(list)
    bar_fractions: list[float] = []

    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        letterbox = payload["letterbox"]
        scale = float(letterbox["scale"])
        bars = int(letterbox["top"]) + int(letterbox["bottom"]) \
            + int(letterbox["left"]) + int(letterbox["right"])
        bar_fractions.append(bars / 512.0)

        for instance in payload.get("instances", []):
            diameter = instance.get("equivalent_diameter_pixels")
            name = instance.get("class_name")
            if diameter is None or not name:
                continue
            by_class[str(name)].append(float(diameter) * scale)

    total = sum(len(v) for v in by_class.values())
    print(f"{len(paths)} images, {total} detections")
    print(f"letterbox bars occupy {np.mean(bar_fractions):.1%} of the 512 frame\n")

    header = (f"{'class':<20}{'n':>6}{'p50':>8}{'p90':>8}{'max':>8}"
              f"{'train p50':>11}{'ratio':>8}")
    print(header)
    print("-" * len(header))
    for name in sorted(by_class):
        values = np.asarray(by_class[name])
        train = TRAIN_MEDIAN.get(name)
        train_text = f"{train:.1f}" if train else "-"
        ratio_text = f"{np.median(values) / train:.2f}x" if train else "-"
        print(f"{name:<20}{len(values):>6}{np.median(values):>8.1f}"
              f"{np.percentile(values, 90):>8.1f}{values.max():>8.1f}"
              f"{train_text:>11}{ratio_text:>8}")

    everything = np.concatenate([np.asarray(v) for v in by_class.values()])
    print()
    print(f"all detections (512 frame): p50 {np.median(everything):.1f} px, "
          f"p90 {np.percentile(everything, 90):.1f}, "
          f"p99 {np.percentile(everything, 99):.1f}, "
          f"max {everything.max():.1f}")

    # Anything past the ceiling is extrapolation the model was never trained for.
    for name, train in sorted(TRAIN_MEDIAN.items()):
        limit = train * AUGMENTATION_CEILING
        values = np.asarray(by_class.get(name, [0.0]))
        print(f"  {name:<14} augmentation only ever reached {limit:5.1f} px; "
              f"{float((values > limit).mean()):5.1%} of detections exceed it")


if __name__ == "__main__":
    main()
