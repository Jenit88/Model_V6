#!/usr/bin/env python3
"""Find core-extraction parameters that survive noise without losing thin rings.

Two requirements pull in opposite directions:

  thin annuli   a 3 px ring has a maximum inner distance of ~1.5 px, so its
                core is a one-pixel-wide centreline. Any smoothing or
                morphological opening comparable to that width destroys it --
                and this is the `circle` class, AP50-95 0.5630, the single
                largest recoverable loss in the model.
  noisy fields  thresholding a raw prediction speckles, and every speck
                expands into a full instance because cores claim territory by
                nearest-distance.

Smoothing fixes the second and breaks the first. A minimum core area might fix
the second without touching the first: speckle cores are a few pixels, while
even a one-pixel-wide ring of radius 26 has ~163 core pixels. This sweeps both
knobs against both requirements to find out.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_tier2_grouping import iou, load, scenes  # noqa: E402


def evaluate(m, smoothing, min_core, size, head, rng_seed=0):
    """(clean_failures, noisy_count_error) over every scene."""
    clean_failures = 0
    noisy_error = 0
    rng = np.random.default_rng(rng_seed)

    for _, instance_full in scenes(size):
        semantic_full = np.where(instance_full > 0, 1, 0).astype(np.uint8)
        target = m.build_inner_distance_target(semantic_full, instance_full)
        foreground = target[..., 1] > 0.5
        instance_small = np.rint(
            cv2.resize(instance_full.astype(np.float32), (head, head),
                       interpolation=0)
        ).astype(np.int32)
        expected = sum(
            1 for v in np.unique(instance_small)
            if v > 0 and (instance_small == v).sum() >= m.MIN_INSTANCE_AREA
        )

        # Clean: must reproduce exactly.
        predicted = m.watershed_instances(
            foreground, target[..., 0],
            core_smoothing=smoothing, minimum_core_area=min_core,
        )
        worst = 1.0
        for value in np.unique(instance_small):
            if value <= 0:
                continue
            truth = instance_small == value
            if truth.sum() < m.MIN_INSTANCE_AREA:
                continue
            scores = [iou(truth, predicted == f)
                      for f in np.unique(predicted) if f > 0]
            worst = min(worst, max(scores) if scores else 0.0)
        if int(len(np.unique(predicted)) - 1) != expected or worst < 0.95:
            clean_failures += 1

        # Mild degradation: the object count must hold.
        field = cv2.GaussianBlur(target[..., 0], (0, 0), 1.0)
        field = np.clip(
            field + rng.normal(0.0, 0.05, field.shape).astype(np.float32),
            0.0, 1.0,
        )
        predicted = m.watershed_instances(
            foreground, field,
            core_smoothing=smoothing, minimum_core_area=min_core,
        )
        noisy_error += abs(int(len(np.unique(predicted)) - 1) - expected)

    return clean_failures, noisy_error


def main() -> None:
    m = load("model_v6_4")
    size, head = m.IMG_SIZE, m.INSTANCE_HEAD_SIZE

    print(f"{'smoothing':>10}{'min core':>10}{'clean fails':>13}"
          f"{'noisy count error':>19}")
    print("-" * 52)
    best = None
    for smoothing in (0.0, 0.5, 1.0):
        for min_core in (1, 4, 8, 16, 32, 64):
            clean, noisy = evaluate(m, smoothing, min_core, size, head)
            mark = ""
            if clean == 0 and (best is None or noisy < best[0]):
                best = (noisy, smoothing, min_core)
                mark = "  <-- best so far"
            print(f"{smoothing:>10.1f}{min_core:>10}{clean:>13}"
                  f"{noisy:>19}{mark}")

    print()
    if best is None:
        print("no setting reproduces the clean scenes; the design needs work")
        raise SystemExit(1)
    print(f"best with zero clean failures: smoothing={best[1]}, "
          f"min_core_area={best[2]}, residual noisy count error {best[0]}")


if __name__ == "__main__":
    main()
