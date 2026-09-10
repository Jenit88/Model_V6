#!/usr/bin/env python3
"""Train YOLO11l-seg on exactly the data Model V6.2 was trained on.

The existing baseline cannot carry a comparison: it stopped at epoch 23 of 300
-- 8% of its schedule, still near peak learning rate -- and it referenced a
`Split_Data_yolo` directory that no longer exists, so there is no way to prove
it saw the same images. Both problems are fixed here: the dataset is rebuilt
from Split_Data by build_yolo_dataset.py, and training runs to a completed
schedule.

**Why `epochs` and not `time`.** This budgeted by wall clock first, on the
argument that equal compute is the fairer comparison. Measured, that argument
inverts. YOLO runs at 4.3 min/epoch here against V6.2's ~39, so matching V6.2's
~78 GPU-hours buys roughly 1,090 epochs -- nine times what V6.2 got, and far
outside anything the recipe was designed for. Two things go wrong:

  1. 1,090 passes over 4,680 images with 27.6 M parameters invites overfitting,
     which would weaken the baseline and flatter the comparison.
  2. Ultralytics stretches the learning-rate decay across the whole projected
     schedule, while `patience=100` early-stops on a plateau. Together those
     halt the run mid-decay at a high learning rate -- which is precisely the
     truncated-schedule flaw that made the epoch-23 baseline unusable.

So the budget is a completed 300-epoch schedule: YOLO's standard recipe for
this task, what the original baseline aimed at, learning rate decayed to the
end, and still 2.5x the 120 epochs V6.2 received. Report both the epoch counts
and the wall clock; neither alone describes the comparison honestly.

If a previous attempt left a checkpoint behind, this resumes from it. Unlike
the time-based budget, resuming an epoch-based schedule continues to the same
total and does not restart the clock.

    PCB_YOLO_EPOCHS=300 ./run_yolo.sh train
"""
from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    dataset = Path(os.environ.get(
        "PCB_YOLO_DATASET", str(Path.home() / "data/yolo_fair"))) / "data.yaml"
    epochs = int(os.environ.get("PCB_YOLO_EPOCHS", "300"))
    weights = os.environ.get("PCB_YOLO_WEIGHTS", "yolo11l-seg.pt")
    project = Path.home() / "Models" / "yolo11_fair"
    name = os.environ.get("PCB_YOLO_RUN_NAME", "seg")

    if not dataset.is_file():
        raise SystemExit(
            f"{dataset} not found. Run ./run_yolo.sh build first."
        )

    last = project / name / "weights" / "last.pt"
    resuming = last.is_file()

    print("=" * 72)
    print("YOLO11l-seg, matched protocol against Model V6.2")
    print(f"  data      : {dataset}")
    if resuming:
        print(f"  RESUMING  : {last}")
        print(f"  schedule  : continues to the checkpoint's own epoch total")
    else:
        print(f"  weights   : {weights} (COCO-pretrained, as the original baseline)")
        print(f"  schedule  : {epochs} epochs, run to completion (~4.3 min/epoch)")
    print(f"  imgsz     : 512          same as V6.2")
    print(f"  output    : {project / name}")
    print("=" * 72, flush=True)

    if resuming:
        YOLO(str(last)).train(resume=True)
        return

    model = YOLO(weights)
    model.train(
        data=str(dataset),
        imgsz=512,
        batch=4,             # what the original baseline used, and 8 GB allows
        epochs=epochs,
        device=0,
        workers=8,
        amp=True,
        cache=False,
        project=str(project),
        name=name,
        exist_ok=True,
        plots=True,
        seed=0,
        deterministic=True,
        val=True,
    )


if __name__ == "__main__":
    main()
