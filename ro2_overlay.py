#!/usr/bin/env python3
"""One compact overlay renderer, used for both models.

`make_instance_overlay` draws a filled label plate per instance at font scale
0.5 with the instance number and full class name. On a board carrying 40
detections in a small area those plates tile over each other and hide the masks
they describe -- the thing the picture exists to show.

This draws the same information small: two-letter class code, score, thin text
with a dark outline instead of a filled plate, and no instance number. Nothing
here changes what either model predicts; it is presentation only.

Kept in its own module with no dependency beyond numpy and cv2 so that the
TensorFlow venv and the torch venv can both import it, which is also what makes
the two overlay sets directly comparable -- same palette, same geometry, same
text size, so a visible difference between the panels is a difference between
the models.
"""
from __future__ import annotations

import numpy as np

# Same palette as CLASS_COLOURS_RGB in model_v6_2.py, so a colour means the
# same class in every picture this project produces.
CLASS_COLOURS_RGB = {
    1: (230, 65, 65), 2: (255, 165, 45), 3: (60, 180, 90), 4: (65, 135, 230),
}
CLASS_CODES = {1: "R", 2: "RC", 3: "c", 4: "cf"}

FONT_SCALE = 0.28
TEXT_THICKNESS = 1
MASK_ALPHA = 0.40
CONTOUR_THICKNESS = 1


def draw_overlay(
    frame_rgb,
    masks,
    classes,
    scores,
    *,
    alpha: float = MASK_ALPHA,
    font_scale: float = FONT_SCALE,
    draw_labels: bool = True,
    draw_boxes: bool = False,
):
    """Mask fill, outline, and a small class code + score per detection."""
    import cv2

    canvas = frame_rgb.astype(np.float32).copy()
    for mask, class_id in zip(masks, classes):
        colour = np.asarray(CLASS_COLOURS_RGB[int(class_id)], dtype=np.float32)
        canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * colour
    canvas = canvas.astype(np.uint8)

    for mask, class_id, score in zip(masks, classes, scores):
        colour = tuple(int(c) for c in CLASS_COLOURS_RGB[int(class_id)])
        mask_u8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, colour, CONTOUR_THICKNESS)
        if not (draw_labels or draw_boxes):
            continue
        ys, xs = np.nonzero(mask)
        if not xs.size:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        if draw_boxes:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 1)
        if not draw_labels:
            continue
        text = f"{CLASS_CODES[int(class_id)]}{float(score):.2f}".replace("0.", ".")
        anchor = (x0, max(8, y0 - 2))
        # Dark outline first, then the coloured glyphs: readable over both the
        # bright copper and the dark substrate without a plate to hide the mask.
        cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), TEXT_THICKNESS + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    colour, TEXT_THICKNESS, cv2.LINE_AA)
    return canvas


def masks_from_instance_map(instance_map, instances, threshold=0.0):
    """(masks, classes, scores) from V6.2's label map + summarise_instances."""
    masks, classes, scores = [], [], []
    for item in instances:
        confidence = float(item["confidence"])
        if confidence < threshold:
            continue
        mask = instance_map == int(item["instance_id"])
        if not mask.any():
            continue
        masks.append(mask)
        classes.append(int(item["class_id"]))
        scores.append(confidence)
    return masks, classes, scores
