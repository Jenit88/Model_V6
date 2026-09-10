#!/usr/bin/env python3
"""Phase 1 of the label repair: classify what the overlapping polygons actually are.

`rasterize_instances` paints polygons in file order and lets the last one win
every contested pixel (model_v6_2.py:781-782). File order carries no meaning --
it is an accident of how the annotation tool serialised them -- so an earlier
instance can be holed or erased outright. Measured by the dataset audit: 292,656
doubly-claimed pixels and 104 destroyed instances in train, 15,164 and 16 in
test.

Three things could produce that, and they need different fixes:

  nested     an inner feature inside an outer one. The SMALLER polygon must
             own the shared region, and file order may put it either way.
  duplicate  the same pad annotated twice.
  bleed      neighbouring pads whose outlines cross along a shared edge.

A rule that fixes one makes another worse, so this measures the mix before
anything is changed. Geometry uses the same helpers as the real rasteriser, so
the numbers describe the actual training targets rather than an approximation.

**What it found, on test:** nesting is 98.3% of the shared pixels (36 pairs,
14,906 px) and every erasure. Bleed is 78 pairs sharing 258 px between them --
3.3 each, outline rounding. There are no duplicates at all on test and four
pairs on val, and no cross-class duplicates anywhere, so nothing here needs a
human to adjudicate. Every nested pair is one convention: a small `Rectangle`
inside a `Rectangle_concave` at a 3.5-3.8% area ratio.

Duplicates were the expected culprit going in, because Ultralytics reports
"1 duplicate labels removed" on several training images. They are not: that
message is about identical polygon *lines* in a label file, which is a
different and much rarer thing than the overlapping geometry counted here.

    PCB_SPLITS=train,val,test ./classify_label_overlap.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Same-class polygons this similar are the same annotation twice over.
DUPLICATE_IOU = 0.90
# One polygon this fully inside another is an inner feature, not a neighbour.
CONTAINMENT = 0.90


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v6_2", HERE / "model_v6_2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v6_2"] = module
    spec.loader.exec_module(module)
    return module


def polygon_masks(m, label_path, metadata):
    """Rasterise each polygon separately: (semantic_id, bbox, mask, area)."""
    import cv2

    out = []
    for semantic_id, normalized_points in m.parse_yolo_polygons(label_path):
        polygon = m.letterboxed_polygon_pixels(normalized_points, metadata)
        if len(polygon) < 3:
            continue
        left = int(polygon[:, 0].min())
        right = int(polygon[:, 0].max()) + 1
        top = int(polygon[:, 1].min())
        bottom = int(polygon[:, 1].max()) + 1
        if left >= right or top >= bottom:
            continue
        mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon - np.array([[left, top]], np.int32)], color=1)
        area = int(mask.sum())
        if area <= 0:
            continue
        out.append((int(semantic_id), (top, bottom, left, right),
                    mask.astype(bool), area))
    return out


def intersection_area(a, b) -> int:
    """Pixels shared by two bbox-local masks, in the global frame."""
    (at, ab, al, ar), amask = a[1], a[2]
    (bt, bb, bl, br), bmask = b[1], b[2]
    top, bottom = max(at, bt), min(ab, bb)
    left, right = max(al, bl), min(ar, br)
    if top >= bottom or left >= right:
        return 0
    a_window = amask[top - at:bottom - at, left - al:right - al]
    b_window = bmask[top - bt:bottom - bt, left - bl:right - bl]
    return int(np.count_nonzero(a_window & b_window))


def main() -> None:
    m = load_model_module()
    splits = os.environ.get("PCB_SPLITS", "train,val,test").split(",")
    summary: dict[str, dict] = {}

    for split in [s.strip() for s in splits if s.strip()]:
        records = m.list_records(split)
        kinds = Counter()
        shared_pixels = Counter()
        cross_class_duplicates = []
        images_with_overlap = 0
        polygons_total = 0
        pairs_total = 0

        for image_path, label_path in records:
            original = m.read_rgb_image(image_path)
            _, metadata = m.letterbox_rgb(original)
            masks = polygon_masks(m, label_path, metadata)
            polygons_total += len(masks)

            found = False
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    shared = intersection_area(masks[i], masks[j])
                    if shared <= 0:
                        continue
                    found = True
                    pairs_total += 1
                    area_i, area_j = masks[i][3], masks[j][3]
                    union = area_i + area_j - shared
                    iou = shared / union if union else 0.0
                    containment = max(shared / area_i, shared / area_j)

                    if iou >= DUPLICATE_IOU:
                        kind = "duplicate"
                        if masks[i][0] != masks[j][0]:
                            kind = "duplicate_cross_class"
                            cross_class_duplicates.append({
                                "label": str(label_path),
                                "classes": [masks[i][0], masks[j][0]],
                                "iou": round(iou, 4),
                            })
                    elif containment >= CONTAINMENT:
                        kind = "nested"
                    else:
                        kind = "bleed"
                    kinds[kind] += 1
                    shared_pixels[kind] += shared
            if found:
                images_with_overlap += 1

        total_shared = sum(shared_pixels.values())
        summary[split] = {
            "images": len(records),
            "images_with_overlap": images_with_overlap,
            "polygons": polygons_total,
            "overlapping_pairs": pairs_total,
            "by_kind": {k: {"pairs": kinds[k], "shared_pixels": shared_pixels[k]}
                        for k in sorted(kinds)},
            "cross_class_duplicates": cross_class_duplicates[:20],
            "cross_class_duplicate_count": len(cross_class_duplicates),
        }

        print(f"\n===== {split} =====")
        print(f"{len(records):,} images, {polygons_total:,} polygons, "
              f"{images_with_overlap:,} images with overlap "
              f"({images_with_overlap / max(len(records), 1):.1%})")
        print(f"{pairs_total:,} overlapping pairs, {total_shared:,} shared pixels\n")
        print(f"  {'kind':<24}{'pairs':>9}{'share':>9}{'px':>12}{'px share':>10}")
        for kind in sorted(kinds, key=lambda k: -kinds[k]):
            print(f"  {kind:<24}{kinds[kind]:>9,}"
                  f"{kinds[kind] / max(pairs_total, 1):>9.1%}"
                  f"{shared_pixels[kind]:>12,}"
                  f"{shared_pixels[kind] / max(total_shared, 1):>10.1%}")
        if cross_class_duplicates:
            # These cannot be auto-resolved: two near-identical polygons that
            # disagree about the class is a real annotation conflict.
            print(f"\n  *** {len(cross_class_duplicates)} cross-class duplicates "
                  "-- these need a human, not a rule ***")

    out = HERE / "results" / "label_overlap_classification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
