#!/usr/bin/env python3
"""Why does YOLO collapse on image 0 and nowhere else?

Image 0 is not a scale outlier -- every board here sits at 0.41-0.65x training's
Rectangle p50, and image 0's 0.65x is barely above the pack. Yet YOLO's median
confidence there is 0.011 against ~0.89 on the other eleven. A cliff that steep
wants a categorical explanation, not a gradient one.

The session log has an open item that predicts exactly this: YOLO was trained at
`degrees=0.0` while V6.2 sweeps 45-degree rotations. If image 0's objects are
rotated well off-axis and the other eleven are not, the cliff is the augmentation
asymmetry showing up on real data rather than anything about these boards.

Orientation is measured from V6.2's own masks with minAreaRect, folded to
[0, 45] since a rectangle at 80 degrees is a rectangle at 10 degrees for this
purpose. Elongated objects only -- a square or a circle has no meaningful angle.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "results" / "ro2"
MIN_ELONGATION = 1.6


def main() -> None:
    import cv2

    with (OUT / "v62_records.pkl").open("rb") as handle:
        v62 = pickle.load(handle)
    with (OUT / "yolo_records.pkl").open("rb") as handle:
        yolo = pickle.load(handle)
    meta = json.loads((OUT / "frames_meta.json").read_text())

    print(f"{'#':<4}{'name':<24}{'elongated':>10}{'angle p50':>11}"
          f"{'p10':>7}{'p90':>7}   {'YOLO med conf':>13}")
    rows = []
    for record in meta:
        index = record["index"]
        image = v62["per_image"][index]
        instance_map = image["instance_map"]
        angles = []
        for item in image["instances"]:
            if item["confidence"] < 0.5:
                continue
            mask = (instance_map == item["instance_id"]).astype(np.uint8)
            if not mask.any():
                continue
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            (_, (width, height), angle) = cv2.minAreaRect(
                max(contours, key=cv2.contourArea)
            )
            if min(width, height) <= 0:
                continue
            if max(width, height) / min(width, height) < MIN_ELONGATION:
                continue
            if width < height:            # minAreaRect's axis convention
                angle += 90.0
            angle = abs(angle) % 90.0
            angles.append(min(angle, 90.0 - angle))
        if not angles:
            continue
        angles = np.asarray(angles)
        scores = np.asarray(yolo["per_image"][index]["scores"], dtype=float)
        median_conf = float(np.median(scores)) if scores.size else float("nan")
        print(f"{index:<4}{record['name'][:22]:<24}{len(angles):>10}"
              f"{np.percentile(angles, 50):>11.1f}{np.percentile(angles, 10):>7.1f}"
              f"{np.percentile(angles, 90):>7.1f}{median_conf:>14.3f}")
        rows.append((index, float(np.percentile(angles, 50)), median_conf))

    print("\nangles are degrees off the nearest axis, folded to [0, 45];")
    print("0 = axis-aligned, 45 = fully diagonal.")
    if len(rows) > 2:
        angle_values = np.asarray([r[1] for r in rows])
        conf_values = np.asarray([r[2] for r in rows])
        correlation = float(np.corrcoef(angle_values, conf_values)[0, 1])
        print(f"\ncorrelation(median angle, YOLO median confidence) = "
              f"{correlation:+.3f} over {len(rows)} images")
        print("n is 12, so this is a direction, not an effect size.")


if __name__ == "__main__":
    main()
