#!/usr/bin/env python3
"""Step 7, phase C: score both models with one metric, with bootstrap intervals.

Phases A and B captured per-image mask-AP records for Model V6.2 and
YOLO11l-seg against identical targets. This scores both with the *same*
function -- `mask_average_precision_report` out of model_v6_2.py -- so no part
of the difference can come from two metric implementations disagreeing.

Two properties make the interval meaningful:

**It is paired.** Each bootstrap draw resamples the 432 test images once and
scores both models on that same draw. The interval on the difference therefore
removes the shared per-image difficulty that dominates the variance of either
model taken alone; an unpaired interval would be far wider and would understate
the evidence.

**Resampled duplicates are re-indexed.** The metric keys matched targets on
(image_index, target_id). Sampling an image twice without renumbering would let
one copy's match block the other's, silently deflating both models. Each slot
in a draw therefore gets a fresh index.

The claim the project needs is not that one mean exceeds the other. It is that
the interval on the *difference* excludes zero. Overlapping intervals mean
nothing has been beaten.

    ./run_step7.sh score [draws]
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
METRICS = ("map50_95", "map50", "map75")


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v6_2", HERE / "model_v6_2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v6_2"] = module
    spec.loader.exec_module(module)
    return module


def load_capture(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path} -- run the capture phase first")
    with path.open("rb") as handle:
        return pickle.load(handle)


def score(report_fn, per_image: dict, indices, class_ids) -> dict:
    """Score one model over `indices`, renumbering each slot uniquely."""
    records = {class_id: [] for class_id in class_ids}
    counts = {class_id: 0 for class_id in class_ids}

    for slot, index in enumerate(indices):
        image = per_image[index]
        for class_id in class_ids:
            for record in image["records"].get(class_id, ()):
                # Only image_index changes; everything else is carried through.
                records[class_id].append({
                    "image_index": slot,
                    "instance_id": record["instance_id"],
                    "confidence": record["confidence"],
                    "target_ious": record["target_ious"],
                })
            counts[class_id] += int(image["target_counts"].get(class_id, 0))

    return report_fn(records, counts)


def interval(values: np.ndarray) -> tuple[float, float]:
    return (float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)))


def main() -> None:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    m = load_model_module()
    report_fn = m.mask_average_precision_report
    class_ids = list(range(1, m.NUM_CLASSES))

    v62 = load_capture(HERE / "results" / "step7_v62_records.pkl")
    yolo = load_capture(HERE / "results" / "step7_yolo_records.pkl")

    shared = sorted(set(v62["per_image"]) & set(yolo["per_image"]))
    if len(shared) != len(v62["per_image"]) or len(shared) != len(yolo["per_image"]):
        raise SystemExit(
            f"models were scored on different images: V6.2 has "
            f"{len(v62['per_image'])}, YOLO has {len(yolo['per_image'])}, "
            f"{len(shared)} in common"
        )
    print(f"images : {len(shared)} (identical set for both models)")
    print(f"draws  : {draws}\n", flush=True)

    point = {
        "Model V6.2": score(report_fn, v62["per_image"], shared, class_ids),
        "YOLO11l-seg": score(report_fn, yolo["per_image"], shared, class_ids),
    }

    # The capture wrapper must not have changed what the pipeline computed.
    reference = v62.get("reference_map50_95")
    rescored = point["Model V6.2"]["map50_95"]
    if reference is not None:
        drift = abs(float(reference) - float(rescored))
        print(f"V6.2 pipeline mAP50-95   : {float(reference):.6f}")
        print(f"V6.2 rescored here       : {rescored:.6f}")
        print(f"drift                    : {drift:.2e}"
              f"{'  OK' if drift < 1e-9 else '  *** MISMATCH ***'}\n", flush=True)
        if drift >= 1e-6:
            raise SystemExit(
                "rescoring does not reproduce the pipeline number; the capture "
                "is not faithful and no interval computed from it would mean "
                "anything"
            )

    rng = np.random.default_rng(0)
    samples = {name: {metric: [] for metric in METRICS} for name in point}
    differences = {metric: [] for metric in METRICS}

    started = time.time()
    for draw in range(draws):
        picked = [shared[i] for i in rng.integers(0, len(shared), len(shared))]
        drawn = {
            "Model V6.2": score(report_fn, v62["per_image"], picked, class_ids),
            "YOLO11l-seg": score(report_fn, yolo["per_image"], picked, class_ids),
        }
        for metric in METRICS:
            a = float(drawn["Model V6.2"][metric])
            b = float(drawn["YOLO11l-seg"][metric])
            samples["Model V6.2"][metric].append(a)
            samples["YOLO11l-seg"][metric].append(b)
            differences[metric].append(a - b)

        if (draw + 1) % 50 == 0 or draw + 1 == draws:
            rate = (time.time() - started) / (draw + 1)
            print(f"  draw {draw + 1}/{draws}  "
                  f"({rate:.2f}s each, {(draws - draw - 1) * rate / 60:.1f} min left)",
                  flush=True)

    summary = {
        "images": len(shared),
        "draws": draws,
        "metric_implementation": "model_v6_2.mask_average_precision_report",
        "paired_bootstrap": True,
        "v62_tta": bool(v62.get("tta")),
        "yolo_weights": yolo.get("weights"),
        "yolo_detections": yolo.get("detections"),
        "point": {name: {metric: float(values[metric]) for metric in METRICS}
                  for name, values in point.items()},
        "interval": {},
        "difference": {},
    }
    for name in point:
        summary["interval"][name] = {
            metric: list(interval(np.asarray(samples[name][metric])))
            for metric in METRICS
        }
    for metric in METRICS:
        values = np.asarray(differences[metric])
        low, high = interval(values)
        summary["difference"][metric] = {
            "point": float(point["Model V6.2"][metric]
                           - point["YOLO11l-seg"][metric]),
            "ci95": [low, high],
            "excludes_zero": bool(low > 0.0 or high < 0.0),
            "fraction_v62_ahead": float((values > 0).mean()),
        }

    out = HERE / "results" / "step7_comparison.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'':<14}{'mAP50-95':>22}{'mAP50':>22}{'mAP75':>18}")
    for name in point:
        cells = ""
        for metric, width in zip(METRICS, (22, 22, 18)):
            low, high = summary["interval"][name][metric]
            cells += f"{point[name][metric]:.4f} [{low:.4f},{high:.4f}]".rjust(width)
        print(f"{name:<14}{cells}")
    print("-" * 78)
    cells = ""
    for metric, width in zip(METRICS, (22, 22, 18)):
        difference = summary["difference"][metric]
        low, high = difference["ci95"]
        cells += f"{difference['point']:+.4f} [{low:+.4f},{high:+.4f}]".rjust(width)
    print(f"{'difference':<14}{cells}")
    print("=" * 78)
    for metric in METRICS:
        difference = summary["difference"][metric]
        verdict = ("excludes zero -- a real difference"
                   if difference["excludes_zero"]
                   else "INCLUDES ZERO -- not beaten at this sample size")
        print(f"  {metric:<10} {verdict}   "
              f"(V6.2 ahead in {difference['fraction_v62_ahead']:.1%} of draws)")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
