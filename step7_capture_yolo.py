#!/usr/bin/env python3
"""Step 7, phase B: capture YOLO11l-seg's per-image mask-AP records on test.

Runs in the torch venv, so it cannot import model_v6_2 (TensorFlow). It
therefore rebuilds the record format in numpy -- but not the metric. Scoring
happens once, in phase C, with V6.2's `mask_average_precision_report` applied
to both models' records.

Three things make this a matched comparison rather than two separate ones:

**The same pixels.** YOLO is fed the 512x512 letterboxed frames out of
`X_test.npy` -- the identical array V6.2 consumes -- not the source PNGs
re-letterboxed by Ultralytics. Both models therefore see the same input to the
byte. Note `model_v6_2.py:554` stores RGB while Ultralytics reads a numpy array
as BGR, so the channels are flipped on the way in; without that, YOLO would be
scored on colour-swapped images and quietly lose.

**The same targets.** Ground-truth instance-to-class assignment is read from
phase A's capture rather than re-derived here, so neither model can be scored
against a different reading of the labels.

**No partition constraint.** V6.2 emits a label map (one instance per pixel);
YOLO emits independently overlapping masks. Forcing YOLO's output through a
label map would resolve its overlaps arbitrarily and cost it detections, so
IoUs are computed per predicted mask directly. `mask_average_precision_report`
only needs {target_id: iou} per prediction, which both formats supply honestly.

Detection settings are Ultralytics' own validation defaults (conf=0.001,
max_det=300) so the ranking curve is complete. retina_masks=True gives YOLO its
best available mask quality, which is the generous choice.

    ./run_step7.sh capture-yolo
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# build_yolo_dataset.py remapped 1,2,3,4 -> 0,1,2,3; invert that here.
YOLO_CLASS_TO_V62 = {0: 1, 1: 2, 2: 3, 3: 4}


def main() -> None:
    from ultralytics import YOLO

    array_dir = Path(os.environ["PCB_ARRAY_DIR"])
    weights = Path(os.environ.get(
        "PCB_YOLO_BEST",
        str(Path.home() / "Models/yolo11_fair/seg/weights/best.pt"),
    ))
    if not weights.is_file():
        raise SystemExit(f"YOLO weights not found: {weights}")

    v62_path = HERE / "results" / "step7_v62_records.pkl"
    if not v62_path.is_file():
        raise SystemExit(f"run phase A first: {v62_path} missing")
    with v62_path.open("rb") as handle:
        v62 = pickle.load(handle)
    per_image_gt = v62["per_image"]

    images = np.load(array_dir / "X_test.npy", mmap_mode="r")
    instances = np.load(array_dir / "Y_instance_test.npy", mmap_mode="r")
    if len(images) != len(instances):
        raise SystemExit("image and instance arrays disagree in length")

    indices = sorted(int(k) for k in per_image_gt)
    print(f"weights : {weights}")
    print(f"arrays  : {array_dir}")
    print(f"images  : {len(indices)} (matched to phase A)\n", flush=True)

    model = YOLO(str(weights))
    captured: dict[int, dict] = {}
    total_detections = 0
    overlapping_pixels = 0
    predicted_pixels = 0

    for position, index in enumerate(indices):
        rgb = np.asarray(images[index], dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])

        result = model.predict(
            bgr,
            imgsz=512,
            conf=0.001,
            iou=0.7,
            max_det=300,
            retina_masks=True,
            device=0,
            verbose=False,
        )[0]

        target_map = np.asarray(instances[index]).astype(np.int64)
        target_classes = {
            int(k): int(v)
            for k, v in per_image_gt[index]["target_classes"].items()
        }
        target_areas = np.bincount(target_map.ravel())
        ids_by_class: dict[int, set[int]] = {}
        for instance_id, class_id in target_classes.items():
            ids_by_class.setdefault(class_id, set()).add(instance_id)

        records: dict[int, list[dict]] = {c: [] for c in YOLO_CLASS_TO_V62.values()}

        if result.masks is not None and len(result.masks) > 0:
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy().astype(float)

            coverage = np.zeros(target_map.shape, dtype=np.int32)
            for detection in range(masks.shape[0]):
                mask = masks[detection] > 0.5
                if mask.shape != target_map.shape:
                    raise SystemExit(
                        f"mask {mask.shape} does not match target {target_map.shape}"
                    )
                area = int(mask.sum())
                if area <= 0:
                    continue
                coverage += mask
                predicted_pixels += area

                class_id = YOLO_CLASS_TO_V62[int(classes[detection])]
                same_class = ids_by_class.get(class_id, set())

                target_ious: dict[int, float] = {}
                if same_class:
                    hit_ids, hit_counts = np.unique(
                        target_map[mask], return_counts=True
                    )
                    for raw_id, raw_count in zip(hit_ids.tolist(), hit_counts.tolist()):
                        target_id = int(raw_id)
                        if target_id <= 0 or target_id not in same_class:
                            continue
                        intersection = int(raw_count)
                        union = area + int(target_areas[target_id]) - intersection
                        if union <= 0:
                            continue
                        target_ious[target_id] = float(intersection / union)

                records[class_id].append({
                    "image_index": index,
                    "instance_id": detection + 1,
                    "confidence": float(min(max(scores[detection], 0.0), 1.0)),
                    "target_ious": target_ious,
                })
                total_detections += 1

            overlapping_pixels += int((coverage > 1).sum())

        captured[index] = {
            "records": records,
            # Reused verbatim from phase A: identical denominators for both models.
            "target_counts": {
                int(k): int(v)
                for k, v in per_image_gt[index]["target_counts"].items()
            },
        }

        if (position + 1) % 50 == 0 or position + 1 == len(indices):
            print(f"  {position + 1}/{len(indices)} images, "
                  f"{total_detections} detections", flush=True)

    out = HERE / "results" / "step7_yolo_records.pkl"
    with out.open("wb") as handle:
        pickle.dump(
            {
                "model": "YOLO11l-seg",
                "weights": str(weights),
                "per_image": captured,
                "detections": total_detections,
                "retina_masks": True,
                "conf": 0.001,
            },
            handle,
            protocol=4,
        )

    overlap_rate = (overlapping_pixels / predicted_pixels) if predicted_pixels else 0.0
    print(f"\n{total_detections} detections over {len(indices)} images")
    # If this is ~0, forcing a label map would have cost YOLO nothing and the
    # per-mask treatment above is merely belt-and-braces. If it is large, the
    # per-mask treatment was load-bearing and must be stated in the writeup.
    print(f"predicted pixels claimed by >1 mask: {overlap_rate:.4%}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
