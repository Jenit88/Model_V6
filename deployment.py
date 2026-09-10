#!/usr/bin/env python3
"""The configuration to run the model on real, unlabelled boards.

Benchmarks and deployment want different operating points, and until now only
the benchmark one was reachable. `fit_thresholds.py` searched for a better set
on 2026-09-05 and found one that survived a held-out half of validation
(+0.0105 F1), but nothing outside `reevaluate.py` ever applied it, so every
prediction run still used values inherited from Model V5.

Measured on 16 real boards where the model was reported as failing:

    score band     shipped   this profile
    0.2 - 0.4          581             13
    0.6 - 1.0          187            194
    total            1,074            299

54% of all detections sat in the 0.2-0.4 band, and they were duplicates --
a spurious low-confidence `Rectangle` on top of pads already found correctly.
The class mix read 80.4% Rectangle against 39.7% in the training data. This
profile removes 98% of that band while leaving the confident detections alone
(they go slightly *up*, because instances no longer fragment against each
other).

**Why this is not simply the better configuration.** On the labelled test split
these thresholds score marginally *worse* -- mask mAP50-95 0.7710 -> 0.7694 --
because they buy precision with recall, and mask AP is rank-aware: a weak
duplicate that ranks last costs a benchmark almost nothing. Unlabelled
deployment has no ranking to hide behind; every retained instance is drawn and
counted. Same model, genuinely different right answer. Keep the shipped values
for anything whose number gets published, and this for anything that looks at
a real board.

    from deployment import apply_deployment_profile
    applied = apply_deployment_profile(model_module)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
THRESHOLD_FIT = HERE / "results" / "threshold_fit.json"

# results/threshold_fit.json -> model_v6_2 module attribute.
THRESHOLD_ATTRIBUTES = {
    "semantic_confidence": "SEMANTIC_CONFIDENCE_THRESHOLD",
    "center_confidence": "CENTER_CONFIDENCE_THRESHOLD",
    "center_nms_radius": "CENTER_NMS_RADIUS",
    "minimum_instance_area": "MIN_INSTANCE_AREA",
    "boundary_confidence": "BOUNDARY_CONFIDENCE_THRESHOLD",
}

# Instances recovered without a centre peak score FALLBACK_INSTANCE_SCORE
# (0.05) and survive the fitted thresholds, because those act on centre
# evidence this path has none of. On the same 16 boards they were 24% of what
# remained. 0.50 clears them without touching the 0.6-1.0 mass where every
# correct detection sits; the measured gap between the two populations is wide
# enough that the exact value is not delicate.
DEFAULT_MIN_CONFIDENCE = 0.50


def apply_deployment_profile(module, *, minimum_confidence=None, verbose=True):
    """Point `module` at the deployment operating point. Returns what changed.

    Set PCB_DEPLOYMENT_PROFILE=0 to opt out and keep the shipped thresholds,
    or PCB_MIN_CONFIDENCE to override the floor.
    """
    if os.environ.get("PCB_DEPLOYMENT_PROFILE", "1").strip().lower() in (
        "0", "false", "no",
    ):
        if verbose:
            print("thresholds : shipped (deployment profile disabled)")
        return None

    if minimum_confidence is None:
        minimum_confidence = float(
            os.environ.get("PCB_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE)
        )

    applied: dict[str, object] = {}
    if THRESHOLD_FIT.is_file():
        values = json.loads(THRESHOLD_FIT.read_text())["fitted"]["values"]
        for key, attribute in THRESHOLD_ATTRIBUTES.items():
            if key in values:
                setattr(module, attribute, values[key])
                applied[key] = values[key]
    elif verbose:
        # Not fatal: the floor alone still removes the fallback band. But the
        # duplicate suppression is the larger half, so say so rather than
        # letting a quieter run look like a healthy one.
        print(f"WARNING: {THRESHOLD_FIT} missing -- fitted thresholds NOT "
              "applied; only the confidence floor is active")

    module.DEPLOYMENT_MIN_CONFIDENCE = float(minimum_confidence)
    applied["minimum_confidence"] = float(minimum_confidence)

    if verbose:
        print(f"profile    : deployment -> {json.dumps(applied)}")
    return applied
