#!/usr/bin/env python3
"""What does the decoder's centre-less fallback path actually contribute?

The final test evaluation showed `Rectangle_concave` at precision 0.4414 (98 TP
against 124 FP) while its AP50-95 was 0.8105 -- a rank-aware metric that healthy
alongside a fixed-threshold metric that poor means the cut-off is wrong, not the
model. Rendering six test images showed every false positive carrying
`centre_score` exactly FALLBACK_INSTANCE_SCORE: they are all instances the
decoder recovered with no centre peak.

Before changing any threshold, measure what that path is worth. This script
re-runs the official matcher over the same predictions with instances filtered
out, so every row is the real metric under a real intervention rather than an
estimate:

  keep everything          what ships today
  drop fallback-only       what deleting the path would give
  score >= t               what raising the threshold would give

Deleting and thresholding are different interventions: thresholding also
discards the fallback path's genuine recoveries, and this is the table that
says whether there are any.

    ./run.sh measure-fallback            # full validation split
    PCB_MEASURE_IMAGES=200 ./run.sh measure-fallback
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v6_2", HERE / "model_v6_2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v6_2"] = module
    spec.loader.exec_module(module)
    return module


def filtered_map(instance_map, classes, keep_ids):
    """Return (map, classes) containing only `keep_ids`."""
    if len(keep_ids) == len(classes):
        return instance_map, classes
    keep = np.zeros(int(instance_map.max()) + 1, dtype=bool)
    for i in keep_ids:
        if i < keep.size:
            keep[i] = True
    out = np.where(keep[instance_map], instance_map, 0)
    return out, {i: c for i, c in classes.items() if i in keep_ids}


def main() -> None:
    m = load_model_module()
    import tensorflow as tf

    m.REQUIRE_GPU = False
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    output_dir = Path(os.environ["PCB_MODEL_OUTPUT_DIR"])
    model_path = output_dir / "best_model_v6_2_instance.keras"
    split = os.environ.get("PCB_MEASURE_SPLIT", "val")
    limit = int(os.environ.get("PCB_MEASURE_IMAGES", "0"))

    print(f"model : {model_path.name}")
    print(f"split : {split}")
    model = tf.keras.models.load_model(model_path, compile=False)
    arrays = m.load_split_arrays(split, array_dir=Path(os.environ["PCB_ARRAY_DIR"]))
    total = len(arrays["images"])
    count = total if limit <= 0 else min(limit, total)
    print(f"images: {count} of {total}\n", flush=True)

    # Interventions to compare. `None` means "keep every instance".
    thresholds = [None, "drop_fallback", 0.20, 0.30, 0.40, 0.50, 0.60]
    tally = {str(t): defaultdict(lambda: defaultdict(int)) for t in thresholds}
    fallback_score = float(m.FALLBACK_INSTANCE_SCORE)
    started = time.time()

    for position in range(count):
        image = m.normalize_image(arrays["images"][position])[None, ...]
        outputs = model.predict_on_batch(image)
        semantic, centre, offset, boundary = m.output_probabilities(outputs, 0)

        target_semantic, target_instance = m.sanitize_semantic_and_instances(
            np.asarray(arrays["semantic"][position], dtype=np.int32),
            np.asarray(arrays["instance"][position], dtype=np.int32),
        )
        target_classes = m.ground_truth_instance_classes(
            target_instance, target_semantic
        )

        instance_map, classes, confidence, centre_scores = m.decode_instances(
            semantic, centre, offset, boundary
        )

        # The detection score the evaluator ranks by: sqrt(centre) * mean
        # semantic confidence of the instance's own pixels.
        detection = {}
        for iid in classes:
            pixels = instance_map == iid
            mean_conf = float(confidence[pixels].mean()) if pixels.any() else 0.0
            detection[iid] = math.sqrt(
                max(centre_scores.get(iid, fallback_score), 0.0)
            ) * mean_conf

        for t in thresholds:
            if t is None:
                keep = set(classes)
            elif t == "drop_fallback":
                keep = {i for i in classes
                        if centre_scores.get(i, fallback_score) > fallback_score}
            else:
                keep = {i for i in classes if detection[i] >= float(t)}
            fmap, fclasses = filtered_map(instance_map, classes, keep)
            counts = m.match_instances_at_iou(
                fmap, fclasses, target_instance, target_classes,
                m.INSTANCE_EVALUATION_IOU,
            )
            for class_id, c in counts.items():
                for k in ("tp", "fp", "fn"):
                    tally[str(t)][class_id][k] += int(c.get(k, 0))

        if (position + 1) % 25 == 0 or position + 1 == count:
            rate = (position + 1) / (time.time() - started)
            print(f"  {position + 1}/{count}  ({rate:.2f} img/s)", flush=True)

    def summarise(t):
        per = tally[str(t)]
        tp = sum(c["tp"] for c in per.values())
        fp = sum(c["fp"] for c in per.values())
        fn = sum(c["fn"] for c in per.values())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return tp, fp, fn, p, r, f1

    print(f"\n{'intervention':<18}{'TP':>8}{'FP':>8}{'FN':>7}"
          f"{'precision':>11}{'recall':>9}{'F1':>9}")
    base = summarise(None)
    for t in thresholds:
        tp, fp, fn, p, r, f1 = summarise(t)
        label = ("keep everything" if t is None else
                 "drop fallback" if t == "drop_fallback" else f"score >= {t:.2f}")
        mark = ""
        if t is not None:
            mark = f"   ({tp - base[0]:+,} TP, {fp - base[1]:+,} FP)"
        print(f"{label:<18}{tp:>8,}{fp:>8,}{fn:>7,}{p:>11.4f}{r:>9.4f}{f1:>9.4f}{mark}")

    print(f"\nper class, keep-everything vs drop-fallback:")
    print(f"{'class':<20}{'TP':>7}{'FP':>7}{'P':>9}  ->{'TP':>7}{'FP':>7}{'P':>9}")
    for class_id in sorted(tally["None"]):
        a, b = tally["None"][class_id], tally["drop_fallback"][class_id]
        pa = a["tp"] / max(a["tp"] + a["fp"], 1)
        pb = b["tp"] / max(b["tp"] + b["fp"], 1)
        print(f"{m.CLASS_NAMES[class_id]:<20}{a['tp']:>7,}{a['fp']:>7,}{pa:>9.4f}"
              f"  ->{b['tp']:>7,}{b['fp']:>7,}{pb:>9.4f}")

    out = HERE / "results" / f"fallback_measurement_{split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {str(t): {str(k): dict(v) for k, v in tally[str(t)].items()}
         for t in thresholds}, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
