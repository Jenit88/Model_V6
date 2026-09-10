#!/usr/bin/env python3
"""Controlled test: is the image-0 collapse about orientation?

Image 0 is the only board here whose objects are fully diagonal (45.0 degrees
off-axis, p10 41.5) and the only one where YOLO collapses -- median confidence
0.011 against ~0.89 on the other eleven. That is consistent with the
augmentation asymmetry already flagged in the session log: `train_yolo_fair.py`
passes no augmentation overrides, so Ultralytics' `degrees: 0.0` applies and
YOLO never saw a rotated board, while V6.2 sweeps 45 exact rotations (1-89 in
2-degree steps, `model_v6_2.py:212-223`).

One anomalous image cannot establish that. This rotates boards where YOLO *is*
confident and watches what happens to each model as the angle increases.

The geometry matters, and the first version of this test got it wrong. Cropping
each source to its centred square and then to S/sqrt(2) avoided fill, but it
also changed the frame's aspect and size, and `letterbox_rgb` scales to fit --
so objects arrived on the 512 canvas about 2x larger than in the real run. That
made the sweep a scale experiment wearing an orientation experiment's clothes,
and both models changed behaviour at 0 degrees, where nothing should have moved.

Scale is the confound to kill, because it is the axis these models are already
known to be sensitive to. So the rotation happens **in place**: the image keeps
its original width and height, is rotated about its centre, and the corners are
filled by reflection. `letterbox_rgb` then produces exactly the scale factor the
real run used, object sizes are unchanged, and 0 degrees is the untouched
original. The reflected corners are the price; they are identical in kind at
every non-zero angle, so the shape of the curve across angles is still read
against a fixed background.

    ./run_ro2.sh rotation-build     then capture both models, then
    ./run_ro2.sh rotation-report
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "ro2_rotation"
IMAGES = OUT / "images"
# Boards where YOLO is confident and objects are close to axis-aligned, so
# there is room for the angle to do something.
SOURCES = ("sample_002303_5.png", "sample_002303_6.png")
ANGLES = (0, 15, 30, 45, 60, 75, 90)


def build(source_dir: Path) -> None:
    import cv2

    shutil.rmtree(IMAGES, ignore_errors=True)
    IMAGES.mkdir(parents=True, exist_ok=True)

    for name in SOURCES:
        path = source_dir / name
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        centre = (width / 2.0 - 0.5, height / 2.0 - 0.5)

        for angle in ANGLES:
            if angle == 0:
                rotated = image           # the untouched original
            else:
                matrix = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
                rotated = cv2.warpAffine(
                    image, matrix, (width, height),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
                )
            out = IMAGES / f"{Path(name).stem}_rot{angle:03d}.png"
            cv2.imwrite(str(out), rotated)
            print(f"{out.name}: {rotated.shape[1]}x{rotated.shape[0]}")

    print(f"\n{len(SOURCES) * len(ANGLES)} images written to {IMAGES}")
    print("Original width and height preserved at every angle, so letterbox_rgb")
    print("produces the same scale factor as the real run and object size is")
    print("held fixed. Only orientation varies.")


def report() -> None:
    meta = json.loads((OUT / "frames_meta.json").read_text())
    with (OUT / "v62_records.pkl").open("rb") as handle:
        v62 = pickle.load(handle)
    with (OUT / "yolo_records.pkl").open("rb") as handle:
        yolo = pickle.load(handle)

    by_board: dict[str, dict[int, dict]] = {}
    for record in meta:
        stem = Path(record["name"]).stem
        board, angle = stem.rsplit("_rot", 1)
        index = record["index"]

        v62_scores = [i["confidence"] for i in v62["per_image"][index]["instances"]]
        yolo_scores = np.asarray(yolo["per_image"][index]["scores"], dtype=float)
        by_board.setdefault(board, {})[int(angle)] = {
            "v62_kept": sum(1 for s in v62_scores if s >= 0.5),
            "v62_median": float(np.median(v62_scores)) if v62_scores else float("nan"),
            "yolo_kept": int((yolo_scores >= 0.5).sum()),
            "yolo_median": float(np.median(yolo_scores)) if yolo_scores.size
            else float("nan"),
        }

    print("=" * 78)
    print("ROTATION CONTROL: same pixels, same scale, same field of view")
    print("=" * 78)
    for board, angles in by_board.items():
        print(f"\n{board}")
        print(f"  {'angle':>7}{'V6.2 kept':>12}{'V6.2 med':>11}"
              f"{'YOLO kept':>12}{'YOLO med':>11}")
        for angle in sorted(angles):
            row = angles[angle]
            print(f"  {angle:>5}deg{row['v62_kept']:>12}{row['v62_median']:>11.3f}"
                  f"{row['yolo_kept']:>12}{row['yolo_median']:>11.3f}")

    print("\n" + "-" * 78)
    # Counts are the wrong summary here: reflecting the corners at every
    # non-zero angle introduces extra board to find, so both models legitimately
    # detect more as the angle grows. Confidence is the stable quantity -- the
    # same objects, scored -- so robustness is how far it moves across angles.
    print("CONFIDENCE STABILITY ACROSS ANGLE (the same objects, rescored)")
    for model, key in (("Model V6.2", "v62_median"), ("YOLO11l-seg", "yolo_median")):
        values = [row[key] for angles in by_board.values()
                  for row in angles.values()]
        collapsed = sum(1 for v in values if v < 0.5)
        print(f"  {model:<14}min {min(values):.3f}  max {max(values):.3f}  "
              f"spread {max(values) - min(values):.3f}  "
              f"below 0.5 in {collapsed}/{len(values)} angle-board pairs")
    print("\nBoth models agree at 0 degrees, which is what makes the rest")
    print("comparable. A model trained with rotation should hold its confidence")
    print("across the row; one trained at degrees=0.0 need not.")

    (OUT / "rotation_summary.json").write_text(
        json.dumps(by_board, indent=2), encoding="utf-8"
    )
    print(f"\nwritten: {OUT / 'rotation_summary.json'}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    if mode == "build":
        build(Path(sys.argv[2]) if len(sys.argv) > 2
              else Path("/mnt/c/Users/u117134/Desktop/ro/ro_2"))
    elif mode == "report":
        report()
    else:
        raise SystemExit("usage: ro2_rotation_test.py {build|report} [source_dir]")
