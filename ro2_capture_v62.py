#!/usr/bin/env python3
"""Compare phase A: build the shared frames, then capture V6.2's detections.

These images are unlabelled, so there is no mAP to quote here and no ground
truth to match against. What can be compared honestly on unlabelled boards is
what each model *emits* on identical pixels: how many objects, of what class,
at what confidence, and whether the two models agree about where they are.

This phase owns the letterboxing for both models. Every frame is produced by
`letterbox_rgb` -- the same function V6.2's own prediction path calls -- and
written to disk, so phase B feeds YOLO the identical bytes rather than letting
Ultralytics re-letterbox the source PNGs into slightly different pixels. That
was the single decision that made step 7 a matched comparison.

Detections are captured with NO confidence floor (module defaults), so the
whole ranking is on disk and phase C can threshold both models at any matched
operating point. Applying deployment.py's 0.50 floor here would have thrown
away the distribution that is the point of the exercise.

Inference is single-pass. `predict()` has no TTA path, so this is V6.2's
deployment configuration -- not the TTA configuration that produced the
published 0.7710.

    PCB_RO2_SOURCE=/mnt/c/Users/u117134/Desktop/ro/ro_2 ./run_ro2.sh capture-v62
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

from ro2_overlay import draw_overlay, masks_from_instance_map

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
    import cv2
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    m.REQUIRE_GPU = False
    m.configure_runtime()

    source = Path(os.environ["PCB_RO2_SOURCE"])
    out_dir = Path(os.environ.get("PCB_RO2_OUT", str(HERE / "results" / "ro2")))
    out_dir.mkdir(parents=True, exist_ok=True)

    m.MODEL_FOR_INFERENCE = (
        Path(os.environ["PCB_MODEL_OUTPUT_DIR"]) / "best_model_v6_2_instance.keras"
    )

    # Two configurations are worth seeing on unlabelled boards. "benchmark" is
    # the module defaults that produced the published numbers, and leaves the
    # whole confidence range on disk. "deployment" is what predict_folder.py
    # actually ships: fitted decode thresholds plus a 0.50 floor. The floor can
    # be reapplied afterwards, but the decode thresholds cannot -- they change
    # which instances exist at all -- so that pass has to be captured, not
    # simulated.
    profile = os.environ.get("PCB_RO2_PROFILE", "benchmark").strip().lower()
    if profile not in ("benchmark", "deployment"):
        raise SystemExit(f"PCB_RO2_PROFILE must be benchmark or deployment: {profile}")
    if profile == "deployment":
        from deployment import apply_deployment_profile
        apply_deployment_profile(m)
    suffix = "" if profile == "benchmark" else "_deployment"

    paths = sorted(
        p for p in source.rglob("*")
        if p.suffix.lower() in m.IMAGE_EXTENSIONS and p.is_file()
    )
    if not paths:
        raise SystemExit(f"no images found under {source}")

    # Sampling lives here rather than in phase B because phase A writes the
    # frames both models read: whatever is chosen here is what YOLO sees, and a
    # fixed seed makes the choice reproducible across reruns of either model.
    sample = int(os.environ.get("PCB_RO2_SAMPLE", "0"))
    seed = int(os.environ.get("PCB_RO2_SEED", "20260908"))
    if sample and sample < len(paths):
        paths = sorted(random.Random(seed).sample(paths, sample))

    (out_dir / f"overlays_v62{suffix}").mkdir(parents=True, exist_ok=True)

    print(f"source : {source}")
    print(f"profile: {profile}")
    print(f"images : {len(paths)}")
    print(f"model  : {m.MODEL_FOR_INFERENCE.name}\n", flush=True)

    # ---- shared input, built once, consumed by both models -----------------
    frames = np.zeros((len(paths), m.IMG_SIZE, m.IMG_SIZE, 3), dtype=np.uint8)
    meta: list[dict] = []
    for index, path in enumerate(paths):
        rgb = m._as_rgb_uint8(m.read_rgb_image(path), name=f"image {path}")
        letterboxed, metadata = m.letterbox_rgb(rgb)
        frames[index] = letterboxed
        pad = 1.0 - (
            float(metadata["resized_width"]) * float(metadata["resized_height"])
        ) / float(m.IMG_SIZE * m.IMG_SIZE)
        meta.append({
            "index": index,
            "name": path.name,
            "path": str(path),
            "original_width": int(metadata["original_width"]),
            "original_height": int(metadata["original_height"]),
            "resized_width": int(metadata["resized_width"]),
            "resized_height": int(metadata["resized_height"]),
            "left": int(metadata["left"]),
            "top": int(metadata["top"]),
            "scale": float(metadata["scale"]),
            "padding_fraction": pad,
        })
    np.save(out_dir / "frames.npy", frames)
    (out_dir / "frames_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"frames : {out_dir / 'frames.npy'} {frames.shape}")
    print(f"padding: {np.mean([r['padding_fraction'] for r in meta]):.1%} "
          "of every frame is letterbox grey\n", flush=True)

    # ---- V6.2 ---------------------------------------------------------------
    model = tf.keras.models.load_model(m.MODEL_FOR_INFERENCE, compile=False)
    m.validate_model_output_shapes(model)
    m.unpack_model_outputs(
        model(tf.zeros((1, m.IMG_SIZE, m.IMG_SIZE, 3), tf.float32), training=False)
    )
    print("warm-up complete\n", flush=True)

    captured: dict[int, dict] = {}
    total = 0
    started = time.perf_counter()
    for index, path in enumerate(paths):
        normalized = np.asarray(
            m.normalize_image(frames[index]), dtype=np.float32
        )
        outputs = m.unpack_model_outputs(
            model(tf.convert_to_tensor(normalized[None, ...], tf.float32),
                  training=False)
        )
        (
            semantic_probabilities,
            center_probabilities_small,
            offset_vectors_small,
            boundary_probability,
        ) = m.output_probabilities(outputs, 0)
        (
            instance_map,
            class_by_instance,
            _,
            instance_center_scores,
        ) = m.decode_instances(
            semantic_probabilities,
            center_probabilities_small,
            offset_vectors_small,
            boundary_probability,
        )
        # Published summariser, so `confidence` here is the same quantity the
        # shipped instances.json reports rather than something recomputed.
        instances = m.summarise_instances(
            instance_map,
            class_by_instance,
            semantic_probabilities,
            instance_center_scores,
        )

        # Compact renderer rather than make_instance_overlay: at 40 detections
        # per board its label plates tile over the masks they describe. Same
        # renderer as phase B, so the panels differ only where the models do.
        masks, mask_classes, mask_scores = masks_from_instance_map(
            instance_map, instances
        )
        overlay = draw_overlay(frames[index], masks, mask_classes, mask_scores)
        cv2.imwrite(
            str(out_dir / f"overlays_v62{suffix}" / f"{index:03d}_{path.stem[:50]}.png"),
            overlay[..., ::-1],
        )

        captured[index] = {
            "name": path.name,
            "instance_map": instance_map.astype(np.uint16, copy=False),
            "instances": [
                {
                    "instance_id": int(item["instance_id"]),
                    "class_id": int(item["class_id"]),
                    "class_name": str(item["class_name"]),
                    "confidence": float(item["confidence"]),
                }
                for item in instances
            ],
        }
        total += len(instances)
        print(f"[{index + 1}/{len(paths)}] {path.name}: {len(instances)} instances",
              flush=True)

    elapsed = time.perf_counter() - started
    out = out_dir / f"v62_records{suffix}.pkl"
    with out.open("wb") as handle:
        pickle.dump(
            {
                "model": "Model V6.2",
                "profile": profile,
                "weights": str(m.MODEL_FOR_INFERENCE),
                "tta": False,
                "confidence_floor": float(m.DEPLOYMENT_MIN_CONFIDENCE),
                "class_names": dict(m.CLASS_NAMES),
                "per_image": captured,
                "seconds_per_image": elapsed / len(paths),
            },
            handle,
            protocol=4,
        )
    print(f"\n{total} detections over {len(paths)} images, "
          f"{elapsed / len(paths):.2f} s/image")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
