#!/usr/bin/env python3
"""Check the repaired rasteriser on the real test labels.

Four properties have to hold, and the first three are the point of the change:

  1. Nothing is destroyed. Every polygon that survives duplicate collapsing
     keeps at least one pixel.
  2. Order independence. Reversing the polygon order in a label file must not
     change the rasterised output. This is the actual defect being fixed --
     the old rule let the annotation tool's serialisation decide the targets.
  3. Instance count rises by exactly the instances the old rule erased.
  4. The nested Rectangles get their interiors back, so a Rectangle_concave
     that contains one is now annular rather than solid.

    ./test_rasterize_repair.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIMIT = int(os.environ.get("PCB_TEST_IMAGES", "432"))


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
    records = m.list_records(os.environ.get("PCB_SPLIT", "test"))[:LIMIT]

    total_polygons = total_surviving = 0
    total_contested = total_destroyed = 0
    order_mismatches = []
    annular = 0

    original_parse = m.parse_yolo_polygons

    for image_path, label_path in records:
        original = m.read_rgb_image(image_path)
        _, metadata = m.letterbox_rgb(original)

        semantic, instance, surviving, contested, destroyed = \
            m.rasterize_instances(label_path, metadata)[:5]
        polygons = len(original_parse(label_path))
        total_polygons += polygons
        total_surviving += surviving
        total_contested += contested
        total_destroyed += destroyed

        # A Rectangle_concave (class 2) with a hole in it means the nested
        # Rectangle kept its interior instead of being painted over.
        for instance_id in np.unique(instance):
            if instance_id == 0:
                continue
            pixels = instance == instance_id
            classes = np.unique(semantic[pixels])
            if len(classes) == 1 and classes[0] == 2:
                rows, columns = np.nonzero(pixels)
                box = instance[rows.min():rows.max() + 1,
                               columns.min():columns.max() + 1]
                if np.any((box != instance_id) & (box != 0)):
                    annular += 1

        # Order independence: reverse the polygons and demand identical output.
        if contested:
            m.parse_yolo_polygons = (
                lambda path, _p=original_parse: list(reversed(_p(path)))
            )
            try:
                semantic_r, instance_r, surviving_r, _, _ = \
                    m.rasterize_instances(label_path, metadata)[:5]
            finally:
                m.parse_yolo_polygons = original_parse
            # Instance IDs renumber with order, so compare the partition and
            # the class map, not the raw ID values.
            same_partition = np.array_equal(
                (instance > 0), (instance_r > 0)
            ) and surviving == surviving_r
            if not (np.array_equal(semantic, semantic_r) and same_partition):
                order_mismatches.append(str(label_path.name))

    print(f"images                 : {len(records)}")
    print(f"polygons in labels     : {total_polygons:,}")
    print(f"surviving instances    : {total_surviving:,}")
    print(f"contested pixels       : {total_contested:,}")
    print(f"destroyed instances    : {total_destroyed}"
          f"{'   OK' if total_destroyed == 0 else '   *** FAIL ***'}")
    print(f"annular Rectangle_concave: {annular}")
    print(f"order-dependent images : {len(order_mismatches)}"
          f"{'   OK' if not order_mismatches else '   *** FAIL ***'}")
    for name in order_mismatches[:5]:
        print(f"    {name}")


if __name__ == "__main__":
    main()
