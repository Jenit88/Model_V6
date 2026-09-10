#!/usr/bin/env python3
"""Step 7, phase A: capture Model V6.2's per-image mask-AP records on test.

The comparison has to score both models with ONE metric implementation, and it
has to be able to resample images for bootstrap intervals. Both fall out of
capturing the per-image records that `mask_average_precision_report` already
consumes.

This does not reimplement V6.2's inference. It wraps
`mask_ap_records_for_image` and calls `evaluate("test")` unchanged, so the
records captured here come from exactly the pipeline that produced the
published 0.7710 -- a reimplementation could drift from it in ways no one
would notice.

The ground truth captured alongside (per-image instance-id -> class, and the
per-class target counts) is what phase B scores YOLO against, so both models
are matched to identical targets rather than each deriving its own.

    ./run_step7.sh capture-v62
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


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
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    m.REQUIRE_GPU = False

    output_dir = Path(os.environ["PCB_MODEL_OUTPUT_DIR"])
    m.MODEL_FOR_INFERENCE = output_dir / "best_model_v6_2_instance.keras"
    m.ARRAY_DIR = Path(os.environ["PCB_ARRAY_DIR"])
    m.EVALUATION_MAX_IMAGES = 0          # every test image, no subsampling
    m.USE_TTA_AT_EVALUATION = (
        os.environ.get("PCB_USE_TTA", "1").strip().lower() in ("1", "true", "yes")
    )

    captured: dict[int, dict] = {}
    original = m.mask_ap_records_for_image

    def capture(predicted_map, predicted_classes, predicted_scores,
                target_map, target_classes, image_index):
        records, target_counts = original(
            predicted_map, predicted_classes, predicted_scores,
            target_map, target_classes, image_index,
        )
        index = int(image_index)
        if index in captured:
            raise RuntimeError(
                f"image index {index} captured twice -- the evaluation visited "
                "an image more than once and the bootstrap would double-count it"
            )
        captured[index] = {
            "records": {int(k): list(v) for k, v in records.items()},
            "target_counts": {int(k): int(v) for k, v in target_counts.items()},
            # Phase B scores YOLO against exactly these targets.
            "target_classes": {int(k): int(v) for k, v in target_classes.items()},
        }
        return records, target_counts

    m.mask_ap_records_for_image = capture

    print(f"model  : {m.MODEL_FOR_INFERENCE.name}")
    print(f"TTA    : {m.USE_TTA_AT_EVALUATION}")
    print(f"arrays : {m.ARRAY_DIR}\n", flush=True)

    semantic_report, instance_report = m.evaluate("test")

    if not captured:
        raise SystemExit("captured nothing -- the wrapper was never called")

    out = HERE / "results" / "step7_v62_records.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(
            {
                "model": "Model V6.2",
                "tta": bool(m.USE_TTA_AT_EVALUATION),
                "num_classes": int(m.NUM_CLASSES),
                "class_names": getattr(m, "CLASS_NAMES", None),
                "per_image": captured,
                # The pipeline's own number, so phase C can prove its
                # reimplementation-free scoring reproduces it exactly.
                "reference_map50_95": instance_report.get("mask_map50_95"),
                "reference_map50": instance_report.get("mask_map50"),
                "reference_map75": instance_report.get("mask_map75"),
            },
            handle,
            protocol=4,
        )

    print(f"\ncaptured {len(captured)} images")
    print(f"pipeline mask mAP50-95 : {instance_report.get('mask_map50_95'):.4f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
