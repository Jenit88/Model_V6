#!/usr/bin/env python3
"""Is image 0's disagreement a detection difference or a confidence difference?

Phase C floors both models at 0.50 and reports 22 V6.2-only detections on
image 0 -- by far the worst image. But YOLO emitted 73 raw detections there and
kept only 13 at that floor, so the question is whether YOLO missed those
objects or found them and scored them low. Re-matching the same V6.2 detections
against YOLO's *unfloored* output separates the two.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ro2_compare import OUT, greedy_match, v62_masks, yolo_masks


def main() -> None:
    with (OUT / "v62_records.pkl").open("rb") as handle:
        v62 = pickle.load(handle)
    with (OUT / "yolo_records.pkl").open("rb") as handle:
        yolo = pickle.load(handle)

    print(f"{'#':<4}{'v62>=.5':>9}{'Y>=.5':>7}{'both':>6}"
          f"{'only62':>8}   {'vs unfloored YOLO':>18}{'still only62':>14}"
          f"{'  recovered YOLO conf':>22}")
    for index in sorted(v62["per_image"]):
        left = v62_masks(v62["per_image"][index], 0.50)
        right_floored = yolo_masks(yolo["per_image"][index], 0.50)
        right_raw = yolo_masks(yolo["per_image"][index], 0.0)

        pairs, used_left, _ = greedy_match(left, right_floored)
        only62 = len(left) - len(used_left)

        pairs_raw, used_left_raw, _ = greedy_match(left, right_raw)
        still = len(left) - len(used_left_raw)

        # The confidence YOLO gave the objects it "missed" only by threshold.
        recovered = sorted(
            right_raw[j][2] for i, j, _ in pairs_raw if i not in used_left
        )
        text = (f"{np.median(recovered):.3f} median, "
                f"{max(recovered):.3f} max" if recovered else "--")
        print(f"{index:<4}{len(left):>9}{len(right_floored):>7}{len(pairs):>6}"
              f"{only62:>8}   {len(pairs_raw):>18}{still:>14}   {text}")


if __name__ == "__main__":
    main()
