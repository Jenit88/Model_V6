#!/usr/bin/env python3
"""Compare phase B: YOLO11l-seg on the identical frames phase A wrote.

Runs in the torch venv, so it cannot import model_v6_2. Three things keep this
matched to phase A rather than being a second, separate run:

**The same pixels.** Frames come from `frames.npy`, produced by V6.2's own
`letterbox_rgb`. `model_v6_2.py:554` stores RGB and Ultralytics reads a numpy
array as BGR, so the channels are flipped on the way in -- without that, YOLO
is scored on colour-swapped images and quietly loses.

**The same confidence range.** conf=0.001 and max_det=300, Ultralytics' own
validation defaults, so the full ranking reaches phase C and both models can be
thresholded together. `retina_masks=True` gives YOLO its best mask quality,
which is the generous choice.

**No partition constraint.** V6.2 emits a label map, one instance per pixel;
YOLO emits independently overlapping masks. Those are kept as separate masks
here, exactly as step 7 did -- forcing them through a label map would resolve
overlaps arbitrarily and cost YOLO real detections.

    ./run_ro2.sh capture-yolo
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np

from ro2_overlay import draw_overlay

HERE = Path(__file__).resolve().parent

# build_yolo_dataset.py remapped 1,2,3,4 -> 0,1,2,3; invert that here.
YOLO_CLASS_TO_V62 = {0: 1, 1: 2, 2: 3, 3: 4}
CLASS_NAMES = {1: "Rectangle", 2: "Rectangle_concave", 3: "circle", 4: "circle_full"}


def main() -> None:
    import cv2
    from ultralytics import YOLO

    out_dir = Path(os.environ.get("PCB_RO2_OUT", str(HERE / "results" / "ro2")))
    (out_dir / "overlays_yolo").mkdir(parents=True, exist_ok=True)

    frames_path = out_dir / "frames.npy"
    if not frames_path.is_file():
        raise SystemExit(f"run phase A first: {frames_path} missing")
    frames = np.load(frames_path)

    weights = Path(os.environ.get(
        "PCB_YOLO_BEST", str(Path.home() / "Models/yolo11_fair/seg/weights/best.pt")
    ))
    if not weights.is_file():
        raise SystemExit(f"YOLO weights not found: {weights}")

    print(f"weights : {weights}")
    print(f"frames  : {frames.shape}\n", flush=True)

    model = YOLO(str(weights))
    captured: dict[int, dict] = {}
    total = 0
    overlapping_pixels = 0
    predicted_pixels = 0
    started = time.perf_counter()

    for index in range(len(frames)):
        rgb = np.asarray(frames[index], dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1])

        result = model.predict(
            bgr, imgsz=512, conf=0.001, iou=0.7, max_det=300,
            retina_masks=True, device=0, verbose=False,
        )[0]

        masks: list[np.ndarray] = []
        classes: list[int] = []
        scores: list[float] = []
        if result.masks is not None and len(result.masks) > 0:
            raw = result.masks.data.cpu().numpy()
            raw_classes = result.boxes.cls.cpu().numpy().astype(int)
            raw_scores = result.boxes.conf.cpu().numpy().astype(float)
            coverage = np.zeros(rgb.shape[:2], dtype=np.int32)
            for detection in range(raw.shape[0]):
                mask = raw[detection] > 0.5
                if mask.shape != rgb.shape[:2]:
                    raise SystemExit(
                        f"mask {mask.shape} does not match frame {rgb.shape[:2]}"
                    )
                if not mask.any():
                    continue
                masks.append(mask)
                classes.append(YOLO_CLASS_TO_V62[int(raw_classes[detection])])
                scores.append(float(min(max(raw_scores[detection], 0.0), 1.0)))
                coverage += mask
                predicted_pixels += int(mask.sum())
            overlapping_pixels += int((coverage > 1).sum())

        overlay = draw_overlay(rgb, masks, classes, scores)
        cv2.imwrite(
            str(out_dir / "overlays_yolo" / f"{index:03d}.png"), overlay[..., ::-1]
        )

        captured[index] = {
            # Packed: 300 full-resolution bool masks per image would be 78 MB.
            "masks": np.packbits(
                np.asarray(masks, dtype=bool).reshape(len(masks), -1), axis=1
            ) if masks else np.zeros((0, 0), dtype=np.uint8),
            "mask_shape": rgb.shape[:2],
            "classes": np.asarray(classes, dtype=np.int16),
            "scores": np.asarray(scores, dtype=np.float32),
        }
        total += len(masks)
        print(f"[{index + 1}/{len(frames)}] {len(masks)} detections", flush=True)

    elapsed = time.perf_counter() - started
    out = out_dir / "yolo_records.pkl"
    with out.open("wb") as handle:
        pickle.dump(
            {
                "model": "YOLO11l-seg",
                "weights": str(weights),
                "class_names": CLASS_NAMES,
                "conf": 0.001,
                "retina_masks": True,
                "per_image": captured,
                "seconds_per_image": elapsed / len(frames),
            },
            handle,
            protocol=4,
        )
    rate = (overlapping_pixels / predicted_pixels) if predicted_pixels else 0.0
    print(f"\n{total} detections over {len(frames)} images, "
          f"{elapsed / len(frames):.2f} s/image")
    print(f"predicted pixels claimed by >1 mask: {rate:.4%}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
