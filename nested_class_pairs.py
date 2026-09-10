#!/usr/bin/env python3
"""Which classes nest inside which? Decides whether "smaller wins" is correct.

Phase 1 showed nesting is 98.3% of the label damage on test, but not what is
nested in what. If a `circle_full` sits inside a `circle`, the pair is a pad
with its interior annotated separately and the inner polygon genuinely owns
that region -- "smaller wins" is then the semantically right rule, not merely a
convenient one. If instead two same-class polygons nest, that is a different
problem and needs a different answer.

Letterboxing is an affine map (uniform scale plus translation), so containment
ratios survive it unchanged. This therefore works straight from the label files
in normalised coordinates and never decodes an image -- which is what makes it
seconds rather than the minutes phase 1 spent reading 13 GB of PNGs.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
GRID = 512          # rasterisation grid for the unit square
CONTAINMENT = 0.90
NAMES = {1: "Rectangle", 2: "Rectangle_concave", 3: "circle", 4: "circle_full"}


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v6_2", HERE / "model_v6_2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v6_2"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = load_model_module()
    split = os.environ.get("PCB_SPLIT", "test")
    records = m.list_records(split)

    pairs = Counter()
    inner_area_ratio = []
    same_class = 0

    for _, label_path in records:
        entries = []
        for semantic_id, points in m.parse_yolo_polygons(label_path):
            polygon = np.asarray(points, dtype=np.float64) * (GRID - 1)
            polygon = np.round(polygon).astype(np.int32)
            if len(polygon) < 3:
                continue
            left, top = polygon[:, 0].min(), polygon[:, 1].min()
            right, bottom = polygon[:, 0].max() + 1, polygon[:, 1].max() + 1
            if left >= right or top >= bottom:
                continue
            mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon - np.array([[left, top]], np.int32)], 1)
            area = int(mask.sum())
            if area:
                entries.append((int(semantic_id), (top, bottom, left, right),
                                mask.astype(bool), area))

        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (at, ab, al, ar), amask, aarea = entries[i][1], entries[i][2], entries[i][3]
                (bt, bb, bl, br), bmask, barea = entries[j][1], entries[j][2], entries[j][3]
                top, bottom = max(at, bt), min(ab, bb)
                left, right = max(al, bl), min(ar, br)
                if top >= bottom or left >= right:
                    continue
                shared = int(np.count_nonzero(
                    amask[top - at:bottom - at, left - al:right - al]
                    & bmask[top - bt:bottom - bt, left - bl:right - bl]
                ))
                if shared <= 0:
                    continue
                union = aarea + barea - shared
                if shared / union >= 0.90:
                    continue                      # duplicate, not nesting
                if max(shared / aarea, shared / barea) < CONTAINMENT:
                    continue                      # bleed, not nesting

                # Name the pair inner-first: that is the polygon the rule
                # would hand the shared region to.
                if aarea <= barea:
                    inner, outer, inner_area, outer_area = entries[i][0], entries[j][0], aarea, barea
                else:
                    inner, outer, inner_area, outer_area = entries[j][0], entries[i][0], barea, aarea
                pairs[(inner, outer)] += 1
                inner_area_ratio.append(inner_area / outer_area)
                if inner == outer:
                    same_class += 1

    total = sum(pairs.values())
    print(f"{split}: {len(records):,} images, {total} nested pairs\n")
    print(f"  {'inner (wins region)':<22}{'outer':<22}{'pairs':>7}{'share':>8}")
    for (inner, outer), count in pairs.most_common():
        print(f"  {NAMES.get(inner, inner):<22}{NAMES.get(outer, outer):<22}"
              f"{count:>7}{count / max(total, 1):>8.1%}")
    if inner_area_ratio:
        ratios = np.asarray(inner_area_ratio)
        print(f"\ninner/outer area ratio: p50 {np.median(ratios):.3f}, "
              f"min {ratios.min():.3f}, max {ratios.max():.3f}")
    print(f"same-class nesting: {same_class} of {total}")


if __name__ == "__main__":
    main()
