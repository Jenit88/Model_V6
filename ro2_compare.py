#!/usr/bin/env python3
"""Compare phase C: put both models' output on these 12 boards side by side.

There are no labels here, so nothing in this file is an accuracy measurement
and no mAP is quoted. Three things can be said honestly on unlabelled data:

1. **How much each model emits**, per class and per confidence band. On real
   boards the low bands are where phantom duplicates live -- that is how the
   `falt` set's 54% duplicate rate was found.
2. **Whether the two models agree.** Two independently trained models putting a
   mask in the same place is weak evidence both are right; a detection only one
   of them makes is where to look first. Agreement is not correctness, but on
   unlabelled data it is the only cross-check available.
3. **What the pictures show.** The overlays are the deliverable; the numbers
   below only say which ones to open first.

A caution the numbers cannot carry themselves: **the two confidence scores are
not the same quantity.** V6.2's is sqrt(centre score) x mean semantic
probability; YOLO's is objectness x class probability. Thresholding both at
0.50 matches the number, not the operating point, so a count difference at a
shared threshold is not by itself evidence that one model over-detects.

    ./run_ro2.sh compare
"""
from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("PCB_RO2_OUT", str(HERE / "results" / "ro2")))

CLASS_NAMES = {1: "Rectangle", 2: "Rectangle_concave", 3: "circle", 4: "circle_full"}
BANDS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
MATCH_IOU = 0.50
MATCH_THRESHOLD = 0.50


def load(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def v62_masks(record, threshold):
    """(mask, class_id, confidence) for instances above a confidence floor."""
    instance_map = record["instance_map"]
    out = []
    for item in record["instances"]:
        if item["confidence"] < threshold:
            continue
        mask = instance_map == item["instance_id"]
        if not mask.any():
            continue
        out.append((mask, item["class_id"], item["confidence"]))
    return out


def yolo_masks(record, threshold):
    packed = record["masks"]
    if packed.size == 0:
        return []
    height, width = record["mask_shape"]
    flat = np.unpackbits(packed, axis=1, count=height * width).astype(bool)
    out = []
    for index in range(flat.shape[0]):
        score = float(record["scores"][index])
        if score < threshold:
            continue
        out.append((flat[index].reshape(height, width),
                    int(record["classes"][index]), score))
    return out


def band_counts(scores):
    return {f"{low:.1f}-{high:.1f}": int(sum(1 for s in scores if low <= s < high))
            for low, high in BANDS}


def print_bands(title, scores):
    print(f"\n  {title} ({len(scores):,} detections)")
    if not scores:
        print("    none")
        return
    for low, high in BANDS:
        n = sum(1 for s in scores if low <= s < high)
        bar = "#" * int(round(40 * n / len(scores)))
        print(f"    {low:.1f}-{high:<4.1f}{n:>6,}  {n / len(scores):>6.1%} {bar}")


def greedy_match(left, right, iou_threshold=MATCH_IOU):
    """Match same-class masks by IoU, best pair first."""
    pairs = []
    candidates = []
    for i, (mask_i, class_i, _) in enumerate(left):
        area_i = int(mask_i.sum())
        for j, (mask_j, class_j, _) in enumerate(right):
            if class_i != class_j:
                continue
            intersection = int(np.logical_and(mask_i, mask_j).sum())
            if intersection == 0:
                continue
            union = area_i + int(mask_j.sum()) - intersection
            iou = intersection / union if union else 0.0
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True)
    used_left, used_right = set(), set()
    for iou, i, j in candidates:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        pairs.append((i, j, iou))
    return pairs, used_left, used_right


def contact_sheets(meta, models):
    """One side-by-side PNG per image, plus a montage of all of them."""
    import cv2

    sheets = OUT / "side_by_side"
    sheets.mkdir(parents=True, exist_ok=True)
    for stale in sheets.glob("*.png"):
        stale.unlink()

    available = [(name, OUT / folder) for name, folder in models
                 if (OUT / folder).is_dir()]
    rows = []
    for record in meta:
        index = record["index"]
        tiles = []
        for name, folder in available:
            matches = (sorted(folder.glob(f"{index:03d}_*.png"))
                       or sorted(folder.glob(f"{index:03d}.png")))
            if not matches:
                continue
            image = cv2.imread(str(matches[0]))
            if image is None:
                continue
            header = np.full((26, image.shape[1], 3), 32, dtype=np.uint8)
            cv2.putText(header, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([header, image]))
        if not tiles:
            continue
        separator = np.full((tiles[0].shape[0], 4, 3), 90, dtype=np.uint8)
        joined = tiles[0]
        for tile in tiles[1:]:
            joined = np.hstack([joined, separator, tile])
        caption = np.full((24, joined.shape[1], 3), 16, dtype=np.uint8)
        cv2.putText(caption,
                    f"{index:03d}  {record['name']}  "
                    f"{record['original_width']}x{record['original_height']}  "
                    f"pad {record['padding_fraction']:.0%}",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (220, 220, 220), 1, cv2.LINE_AA)
        joined = np.vstack([caption, joined])
        stem = Path(record["name"]).stem[:44]
        cv2.imwrite(str(sheets / f"{index:03d}_{stem}.png"), joined)
        rows.append(joined)

    if rows:
        width = max(row.shape[1] for row in rows)
        padded = [np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)),
                         constant_values=16) for row in rows]
        cv2.imwrite(str(OUT / "all_images.png"), np.vstack(padded))
    return sheets


def main() -> None:
    meta = json.loads((OUT / "frames_meta.json").read_text())
    yolo = load(OUT / "yolo_records.pkl")
    v62 = load(OUT / "v62_records.pkl")
    deployment_path = OUT / "v62_records_deployment.pkl"
    v62_deployment = load(deployment_path) if deployment_path.is_file() else None

    print("=" * 74)
    print(f"{len(meta)} unlabelled boards from {Path(meta[0]['path']).parent}")
    print("identical 512x512 letterboxed frames to both models "
          "(letterbox_rgb, RGB->BGR for YOLO)")
    print(f"mean letterbox padding: "
          f"{np.mean([r['padding_fraction'] for r in meta]):.1%} of each frame")
    print("=" * 74)

    v62_scores = [i["confidence"] for r in v62["per_image"].values()
                  for i in r["instances"]]
    yolo_scores = [float(s) for r in yolo["per_image"].values()
                   for s in r["scores"]]
    print("\nRAW OUTPUT, no confidence floor on either model")
    print_bands("Model V6.2 (benchmark decode thresholds)", v62_scores)
    print_bands("YOLO11l-seg (conf=0.001, Ultralytics val defaults)", yolo_scores)
    print("\n  NOTE: the two scores are different quantities -- V6.2 is")
    print("  sqrt(centre) x semantic, YOLO is objectness x class. A shared")
    print("  threshold matches the number, not the operating point.")

    print("\n" + "-" * 74)
    print("DETECTIONS RETAINED AT A SHARED CONFIDENCE FLOOR")
    print("-" * 74)
    print(f"  {'floor':<8}{'V6.2':>10}{'YOLO':>10}   {'V6.2/img':>10}{'YOLO/img':>10}")
    sweep = {}
    for floor in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9):
        a = sum(1 for s in v62_scores if s >= floor)
        b = sum(1 for s in yolo_scores if s >= floor)
        sweep[f"{floor:.2f}"] = {"v62": a, "yolo": b}
        print(f"  {floor:<8.2f}{a:>10,}{b:>10,}   "
              f"{a / len(meta):>10.1f}{b / len(meta):>10.1f}")

    if v62_deployment is not None:
        shipped = [i["confidence"] for r in v62_deployment["per_image"].values()
                   for i in r["instances"] if i["confidence"] >= 0.5]
        print(f"\n  V6.2 at the shipped deployment profile "
              f"(fitted decode thresholds + 0.50 floor): {len(shipped):,}")
        print("  Not reproducible by filtering the benchmark run -- its decode")
        print("  thresholds change which instances exist at all.")

    print("\n" + "-" * 74)
    print(f"PER CLASS, both models floored at {MATCH_THRESHOLD:.2f}")
    print("-" * 74)
    per_class = {"v62": defaultdict(int), "yolo": defaultdict(int)}
    for record in v62["per_image"].values():
        for item in record["instances"]:
            if item["confidence"] >= MATCH_THRESHOLD:
                per_class["v62"][item["class_id"]] += 1
    for record in yolo["per_image"].values():
        for class_id, score in zip(record["classes"], record["scores"]):
            if score >= MATCH_THRESHOLD:
                per_class["yolo"][int(class_id)] += 1
    print(f"  {'class':<22}{'V6.2':>9}{'share':>8}{'YOLO':>9}{'share':>8}")
    total_a = max(sum(per_class["v62"].values()), 1)
    total_b = max(sum(per_class["yolo"].values()), 1)
    for class_id in sorted(CLASS_NAMES):
        a, b = per_class["v62"][class_id], per_class["yolo"][class_id]
        print(f"  {CLASS_NAMES[class_id]:<22}{a:>9,}{a / total_a:>8.1%}"
              f"{b:>9,}{b / total_b:>8.1%}")

    print("\n" + "-" * 74)
    print(f"AGREEMENT, same class, IoU >= {MATCH_IOU:.2f}, both floored at "
          f"{MATCH_THRESHOLD:.2f}")
    print("-" * 74)
    print(f"  {'#':<4}{'image':<28}{'V6.2':>6}{'YOLO':>6}{'both':>6}"
          f"{'only62':>8}{'onlyY':>7}{'meanIoU':>9}")
    per_image_rows = []
    all_ious = []
    total_matched = total_only_a = total_only_b = 0
    for record in meta:
        index = record["index"]
        left = v62_masks(v62["per_image"][index], MATCH_THRESHOLD)
        right = yolo_masks(yolo["per_image"][index], MATCH_THRESHOLD)
        pairs, used_left, used_right = greedy_match(left, right)
        ious = [p[2] for p in pairs]
        all_ious.extend(ious)
        only_a = len(left) - len(used_left)
        only_b = len(right) - len(used_right)
        total_matched += len(pairs)
        total_only_a += only_a
        total_only_b += only_b
        iou_text = f"{np.mean(ious):.3f}" if ious else "--"
        print(f"  {index:<4}{record['name'][:26]:<28}{len(left):>6}{len(right):>6}"
              f"{len(pairs):>6}{only_a:>8}{only_b:>7}{iou_text:>9}")
        per_image_rows.append({
            "index": index, "name": record["name"],
            "v62": len(left), "yolo": len(right), "matched": len(pairs),
            "only_v62": only_a, "only_yolo": only_b,
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None,
        })

    denominator = total_matched + total_only_a + total_only_b
    print(f"\n  matched pairs        : {total_matched:,}")
    print(f"  V6.2 only            : {total_only_a:,}")
    print(f"  YOLO only            : {total_only_b:,}")
    if denominator:
        print(f"  agreement            : "
              f"{total_matched / denominator:.1%} of all detections are matched")
    if all_ious:
        arr = np.asarray(all_ious)
        print(f"  matched mask IoU     : mean {arr.mean():.3f}, "
              f"median {np.median(arr):.3f}, min {arr.min():.3f}")
        print("\n  Where the two agree, mask IoU says how much of the remaining")
        print("  difference is geometry rather than detection.")

    sheets = contact_sheets(meta, [
        ("Model V6.2 (deployment profile)", "overlays_v62_deployment"),
        ("Model V6.2 (benchmark)", "overlays_v62"),
        ("YOLO11l-seg", "overlays_yolo"),
    ])

    (OUT / "comparison.json").write_text(json.dumps({
        "images": len(meta),
        "source": str(Path(meta[0]["path"]).parent),
        "mean_padding_fraction": float(
            np.mean([r["padding_fraction"] for r in meta])
        ),
        "match_iou": MATCH_IOU,
        "match_threshold": MATCH_THRESHOLD,
        "raw_bands": {"v62": band_counts(v62_scores),
                      "yolo": band_counts(yolo_scores)},
        "floor_sweep": sweep,
        "per_class_at_threshold": {
            "v62": {CLASS_NAMES[c]: per_class["v62"][c] for c in sorted(CLASS_NAMES)},
            "yolo": {CLASS_NAMES[c]: per_class["yolo"][c] for c in sorted(CLASS_NAMES)},
        },
        "agreement": {
            "matched": total_matched,
            "only_v62": total_only_a,
            "only_yolo": total_only_b,
            "mean_matched_iou": float(np.mean(all_ious)) if all_ious else None,
        },
        "per_image": per_image_rows,
        "seconds_per_image": {
            "v62": v62.get("seconds_per_image"),
            "yolo": yolo.get("seconds_per_image"),
        },
        "caveats": [
            "Unlabelled images: no ground truth, so no accuracy and no mAP.",
            "V6.2 and YOLO confidences are different quantities; a shared "
            "threshold matches the number, not the operating point.",
            "V6.2 runs single-pass here; predict() has no TTA path, so this is "
            "its deployment configuration, not the TTA setup behind 0.7710.",
            "Agreement is not correctness. Both models can be wrong together.",
        ],
    }, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"side-by-side : {sheets}")
    print(f"all in one   : {OUT / 'all_images.png'}")
    print(f"numbers      : {OUT / 'comparison.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
