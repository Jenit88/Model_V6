
# Model V6.2 — scratch training run `Model_v6_2_scratch_rtx`

Complete record of the 120-epoch from-random-init training run of
`model_v6_2.py` on the `Split_Data` PCB dataset, plus every evaluation that has
been run against its final checkpoint.

| | |
|---|---|
| Run directory (authoritative) | WSL Ubuntu: `/home/u117134c/Models/Model_v6_2_scratch_rtx` (from Windows Explorer: `//wsl$/Ubuntu/home/u117134c/Models/Model_v6_2_scratch_rtx`) |
| This directory | archived copy of that run — see [What is in this folder](#what-is-in-this-folder) |
| Model | `Model_v6_2_PCB_Scratch_BiFPN_DualContext_DualDecoder` |
| Task | 4-class instance segmentation, 512×512 letterboxed |
| Init | **from scratch** — no pretrained encoder, no transfer, no prior checkpoint |
| Epochs | 120 / 120 completed |
| Started / finished | 2026-09-01 23:13 → 2026-09-05 06:46 (**79 h 33 m**, ≈ 39.8 min/epoch) |
| Parameters | 6,417,683 (24.48 MB) — all trainable, 0 frozen |
| Hardware | NVIDIA RTX PRO 1000 Blackwell Laptop GPU, 8 GB, under WSL2 |
| Headline (as trained) | **test mask mAP50-95 0.7463**, val 0.8298 |
| Headline (shipped, + TTA) | **test mask mAP50-95 0.7711**, val 0.8450 |

---

## Contents

1. [What is in this folder](#what-is-in-this-folder)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Losses](#losses)
5. [Augmentation](#augmentation)
6. [Optimisation schedule](#optimisation-schedule)
7. [Runtime environment](#runtime-environment)
8. [How training went](#how-training-went)
9. [Checkpoint selection](#checkpoint-selection)
10. [Final results — instance segmentation](#final-results--instance-segmentation)
11. [Final results — semantic segmentation](#final-results--semantic-segmentation)
12. [Post-training improvement chain](#post-training-improvement-chain)
13. [Known issues and caveats](#known-issues-and-caveats)
14. [Reproducing and re-evaluating](#reproducing-and-re-evaluating)

---

## What is in this folder

| Path | What it is |
|---|---|
| `README.md` | this file |
| `TRAINING_REPORT.md` | the report the training script emitted at epoch 119 |
| `training_log.csv` | per-epoch Keras history, 120 rows × 27 columns |
| `instance_checkpoint_history.json` | the 25 periodic instance evaluations (256-image subset) |
| `final_model_selection.json` | the three-way candidate comparison over the complete val split |
| `best_model_v6_2_instance.keras` | **the shipped model** (223 MB) — a copy of `best_semantic_model_v6_2.keras` |
| `best_semantic_model_v6_2.keras` | best `val_semantic_foreground_miou` checkpoint — selection winner |
| `best_instance_model_v6_2.keras` | best periodic-instance checkpoint (epoch 110) |
| `last_model_v6_2.keras` | end-of-training weights (epoch 120) |
| `initial_random.weights.h5` | the random init the run started from (26 MB) — makes the run reproducible |
| `performance/val_final_evaluation/` | complete val split, default decoder, no TTA |
| `performance/test_final_evaluation/` | complete test split, default decoder, no TTA |
| `predictions/` | rendered validation predictions (6 sampled boards) + manifest |
| `validation_previews/` | one preview PNG every 10 epochs, `epoch_0001` … `epoch_0120` |
| `reports/` | mid-run markdown snapshots at epoch 50 and 100 |
| `tensorboard/` | `train/` and `validation/` event files |
| `dataset_and_augmentation_preview.png` | sanity render of the loader and augmentation pipeline |
| `fallback_measurement_val.json` | TP/FP of the centre-less fallback decoder path, per gate setting |
| `label_overlap_classification.json` | polygon-overlap audit for val and test |
| `2026-09-05-scratch-120ep/` | the snapshot taken the day the run finished (subset of the above) |

Checkpoints are 223 MB each because they carry the AdamW moments **and** the EMA
shadow weights, not just the parameters.

**Only in WSL, not copied here:**

- `model_summary.txt` (188 KB full Keras layer table) and `training_config.json`
  (the machine-readable config this README summarises).
- `performance/val_evaluation/` and `performance/test_evaluation/` — the
  **TTA / tuned-threshold** evaluations described in
  [Post-training improvement chain](#post-training-improvement-chain). The
  `*_final_evaluation` folders copied here are the *baseline* decoder only.
- `predictions/` in WSL holds 16 `ro_2` production boards; the copy here holds
  6 validation samples. Different content, same folder name.

---

## Dataset

`Split_Data` from `1000_images` (**not** the `45000 images` folder, which has no
test split). Full measured audit in `memory/dataset-split-data-audit.md`.

| split | images | instances (this run's eval) |
|---|---:|---:|
| train | 4,680 | 151,828 polygons, 32.4 / image |
| val | 1,048 | 48,388 |
| test | 432 | 33,472 |

Source images range 1024×1536 → 2588×1940, letterboxed to 512×512 with fill
value 114. Prepared arrays live on ext4 at `~/data/pcb_v62_arrays` (~9.7 GB) —
never under `/mnt/c`, because a 9p-mounted memmap starves the GPU.

**Classes** (id 0 is background):

| id | class | train polygons | share | class weight |
|---:|---|---:|---:|---:|
| 1 | `Rectangle` | 57,857 | 39.7 % | 3.442 |
| 2 | `Rectangle_concave` | 680 | 0.47 % | 7.741 |
| 3 | `circle` | 12,000 | 8.2 % | 6.099 |
| 4 | `circle_full` | 76,612 | 52.6 % | 4.639 |

Background weight is 1.0. Foreground is 14.9 % of pixels.

**Two properties of this dataset drive almost every result below:**

1. **`Rectangle_concave` is structurally starved** — 680 training polygons, and
   they are large (median 9,104 px). Large *and* rare is the hard combination.
   No architecture change substitutes for more labels.
2. **Test objects are much smaller and much denser than train objects.** Median
   equivalent diameter train→test: `Rectangle` 35.7→21.9 px (0.61×), `circle`
   33.8→16.1 px (**0.48×**), `circle_full` 21.0→16.0 px (0.76×). Objects per
   image: train 32.4, val 46.2, **test 77.5**. This is the single reason test
   mAP50-95 sits 0.084 below val while test *F1 is higher* — the model still
   finds everything, it just outlines small objects less tightly.

---

## Model

`Model_v6_2_PCB_Scratch_BiFPN_DualContext_DualDecoder` — 6,417,683 parameters,
all trainable.

**Input** `(None, 512, 512, 3)`. The stem runs two parallel branches, a learned
RGB detail conv and a **fixed Sobel** edge-feature layer, concatenated and fused
— so edge evidence is present from layer one rather than having to be learned.

**Four output heads:**

| head | shape | purpose |
|---|---|---|
| `semantic` | `(None, 512, 512, 5)` | per-pixel class, background + 4 |
| `center` | `(None, 256, 256, 4)` | per-class instance-centre heatmap |
| `offset` | `(None, 256, 256, 2)` | pixel → centre vector, normalised by head width/height |
| `boundary` | `(None, 512, 512, 1)` | instance boundary, used to split touching objects |

**Spatial contract:** full resolution 512², instance head 256², offsets stored as
`dx / 256, dy / 256`, instance ids `uint16` (max 65,535 per image).

The instance decoder consumes all four heads: threshold semantic → find centres
by NMS on the centre heatmap → assign foreground pixels to centres via the offset
field → cut with the boundary map → drop fragments. Default decoder thresholds:

```
semantic_confidence              0.30
center_confidence                0.10
center_nms_radius                2
maximum_assignment_distance      51
boundary_confidence              0.50
minimum_instance_area            13
minimum_component_area_fraction  0.10
minimum_boundary_core_area       6
maximum_centers_per_class        150
fallback_instance_score          0.05
```

---

## Losses

Weighted sum of four head losses:

| head | loss | weight |
|---|---|---:|
| `semantic` | `SemanticFocalDiceLovaszFromLogits` | 1.00 |
| `offset` | `MaskedOffsetPixelHuberLoss` (δ = 1.0 px) | 1.00 |
| `boundary` | `BoundaryBCEDiceFromLogits` | 0.30 |
| `center` | `CenterNetFocalFromLogits` | 0.25 |

The semantic loss is itself a blend: **focal 0.45 / dice 0.30 / Lovász 0.25**,
with the per-class weights listed in [Dataset](#dataset). No class reaches the
8.0 weight cap, so the cap never bound during this run.

---

## Augmentation

Train only — validation and test are **never** augmented.

- **1 original + 49 realistic variants + 8 dihedral transforms**, on a
  95-epoch cycle.
- **45 exact rotation angles**: 1°, 3°, 5° … 89°, applied in both directions.
  (This is the property that later distinguished V6.2 from the YOLO baseline,
  which trained at `degrees=0.0`.)
- **Zoom-out ladder** — the single most important augmentation for this dataset.
  Discrete steps 0.95 → 0.50 in 0.05 decrements, `p = 0.5`, padded with
  **black (0)**, deliberately *not* the 114 letterbox grey: 114 also occurs
  inside real board images, so padding with it would teach the model that a
  mid-grey region is sometimes background and sometimes not. Without this the
  model would literally never see a test-scale object during training.
- Extra zoom-in 1.06–1.12, zoom-to-fill with a 1.002 safety factor.
- Translation ≤ 4 % of the frame.
- Photometric: brightness ±0.10, contrast ±0.10, gamma 0.9–1.1, exposure gain
  0.95–1.05, saturation 0.92–1.08, hue ±3°, Gaussian noise σ 0.005–0.02.

---

## Optimisation schedule

| | |
|---|---|
| Optimiser | AdamW (β₁ 0.9, β₂ 0.999, ε 1e-7) wrapped in `LossScaleOptimizer` |
| Weight decay | 1e-4 |
| Batch size | **2**, with **2-step gradient accumulation** → effective batch 4 |
| Steps / epoch | 2,340 |
| LR schedule | linear warm-up 6 epochs → cosine decay |
| Peak LR | 3.0e-4 (reached at epoch 6) |
| Final LR | 1.0e-6 (floor) |
| EMA | on, momentum 0.999 |
| Global clipnorm | 5.0 |
| Loss scale | dynamic, initial 32,768, growth every 2,000 steps |
| Precision | `mixed_float16` for training; **evaluation runs in float32** |
| Seed | 42 |

Batch 2 is not a choice, it is the ceiling: batch 4, 6 and 8 all OOM on 8 GB
because Keras `GroupNormalization` materialises a `[N, G, HW, C/G]` moments
tensor. Batch 2 peaks at 3.7 GiB.

---

## Runtime environment

| | |
|---|---|
| TensorFlow | 2.21.0 |
| Keras | 3.15.1 |
| GPU | NVIDIA RTX PRO 1000 Blackwell Generation Laptop GPU (8 GB) |
| Host | Windows 11 + WSL2 Ubuntu, venv `~/envs/pcb62` |

Two environment faults had to be fixed before this GPU would train at all, both
now handled by `env.sh`:

1. **`opencv` overwrites `LD_LIBRARY_PATH` on import.** TensorFlow's `dlopen` of
   `libcusolver.so.11` then fails even though the wheel ships it, TF prints only
   `Cannot dlopen some GPU libraries`, and then **silently trains on the CPU**.
   Every `site-packages/nvidia/*/lib` is now exported up front.
2. **TF 2.21 ships no cubins for compute capability 12.0 (Blackwell)**, so
   kernels JIT from PTX on first use — TF's own warning says "30 minutes or
   longer". `CUDA_CACHE_MAXSIZE` is now 4 GiB; the 256 MB default cannot hold the
   cache, so the cost would otherwise be paid on *every* run. cuDNN 9.2.5 has
   native Blackwell kernels, so convolutions were never affected.

**Measured performance characteristics** (not estimates):

- Sustained fp16 matmul: **17.27 TFLOP/s** — the hardware is healthy.
- Model forward: 42.75 GMAC/image, **70.6 % of it at 256×256**.
- The training step achieves only ~0.66 TFLOP/s → it is **launch- and
  bandwidth-bound across 464 small kernels**, not FLOP-bound. This is why
  `steps_per_execution=8` (−2.5 %) and disabling per-class IoU metrics (−1.3 %)
  were the wins and no arithmetic optimisation helped.
- **XLA does not compile** this model: `ResizeBilinearGrad` mixes f32/f16 under
  `mixed_float16`.
- Data loader in steady state: 23 ms/image, against a ~376 ms/image step — the
  loader is never the bottleneck.
- GPU utilisation during the run: ~92 %.

Only one device on this laptop can train. The **Intel Arc Pro 140T** cannot run
a TF op (`intel-extension-for-tensorflow` is discontinued and never supported
Keras 3) and the **Intel AI Boost NPU** is inference-only with no device node in
WSL. Both were checked, not assumed. The Arc still contributes — it drives the
display, which keeps all 8 GB of the RTX free.

### Incident during the run

One crash at 06:17 on epoch 85:
`Check failed: GpuLaunchKernel(ColumnReduceKernel...) is OK (INTERNAL: unknown error)`
— a CUDA context loss, not a code fault (199k steps had already run clean, no
OOM, the WSL VM survived, no Windows driver-reset event was logged). TensorFlow
`CHECK()`s here, so it aborts with no Python exception and no chance to recover
in-process. It then sat idle for 105 minutes before anyone noticed. `watchdog.sh`
was added afterwards: it polls every 60 s and relaunches, recovery ~2 min. That
idle time is included in the 79 h 33 m wall clock above.

---

## How training went

Selected epochs from `training_log.csv` (loss columns are training unless
prefixed `val_`):

| epoch | loss | semantic | center | offset | boundary | boundary F1 | fg mIoU (tr) | fg mIoU (val) | LR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.719 | 0.732 | 3.125 | 0.0574 | 0.495 | 0.419 | 0.124 | 0.148 | 5.0e-5 |
| 5 | 0.659 | 0.301 | 1.000 | 0.0180 | 0.298 | 0.796 | 0.630 | – | 2.5e-4 |
| 6 | 0.600 | 0.273 | 0.906 | 0.0159 | 0.283 | 0.813 | 0.666 | – | 3.0e-4 |
| 10 | 0.427 | 0.192 | 0.604 | 0.0112 | 0.241 | 0.861 | 0.776 | – | 3.0e-4 |
| 20 | 0.301 | 0.133 | 0.394 | 0.0076 | 0.206 | 0.895 | 0.850 | – | 2.9e-4 |
| 40 | 0.220 | 0.095 | 0.262 | 0.0053 | 0.182 | 0.917 | 0.889 | – | 2.4e-4 |
| 60 | 0.176 | 0.074 | 0.192 | 0.0042 | 0.166 | 0.931 | 0.912 | – | 1.7e-4 |
| 80 | 0.153 | 0.065 | 0.152 | 0.0033 | 0.154 | 0.940 | 0.922 | – | 8.4e-5 |
| 100 | 0.138 | 0.059 | 0.126 | 0.0028 | 0.147 | 0.946 | 0.930 | 0.9081 | 2.4e-5 |
| 120 | **0.1324** | 0.0567 | 0.1167 | 0.0027 | 0.1460 | 0.9469 | **0.9319** | **0.9083** | 1.0e-6 |

Final validation: `val_loss` **0.2014**, `val_boundary_f1` 0.9525,
`val_semantic_foreground_miou` 0.9083, `val_semantic_pixel_accuracy` 0.9859.

Train fg mIoU 0.9319 vs val 0.9083 — a 0.024 gap after 120 epochs from scratch.
The gap is real but small and stopped widening after ~epoch 90; `val_loss` was
still flat-to-falling at the end (0.2042 → 0.2014 over the last 30 epochs), so
this run is **not overfit**.

### Periodic instance evaluation

Every 5 epochs the run decoded instances on a fixed 256-image validation subset.
All 25 evaluations are in `instance_checkpoint_history.json`; `saved` marks the
epochs that beat the running best on `mask_map50_95`.

| epoch | P | R | F1 | mAP50 | mAP50-95 | fg mIoU | saved |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | 0.0184 | 0.0684 | 0.0290 | 0.0134 | 0.0043 | 0.148 | ✓ |
| 5 | 0.7198 | 0.8738 | 0.7894 | 0.8595 | 0.5751 | 0.716 | ✓ |
| 10 | 0.8438 | 0.9383 | 0.8885 | 0.9354 | 0.7284 | 0.832 | ✓ |
| 20 | 0.8820 | 0.9567 | 0.9178 | 0.9513 | 0.7814 | 0.881 | ✓ |
| 30 | 0.8938 | 0.9620 | 0.9267 | 0.9554 | 0.7966 | 0.893 | ✓ |
| 40 | 0.9002 | 0.9633 | 0.9307 | 0.9562 | 0.8052 | 0.891 | ✓ |
| 50 | 0.9048 | 0.9678 | 0.9353 | 0.9600 | 0.8127 | 0.906 | ✓ |
| 55 | 0.9081 | 0.9679 | 0.9371 | 0.9575 | 0.8103 | 0.896 | |
| 60 | 0.9049 | 0.9688 | 0.9358 | 0.9611 | 0.8115 | 0.905 | |
| 65 | 0.9095 | 0.9692 | 0.9384 | 0.9666 | 0.8190 | 0.907 | ✓ |
| 70 | 0.9207 | 0.9719 | 0.9456 | 0.9683 | 0.8220 | 0.907 | ✓ |
| 80 | 0.9194 | 0.9719 | 0.9449 | 0.9701 | 0.8257 | 0.906 | ✓ |
| 90 | 0.9262 | 0.9726 | 0.9488 | 0.9718 | 0.8295 | 0.907 | ✓ |
| 95 | 0.9274 | 0.9730 | 0.9497 | 0.9718 | 0.8296 | 0.909 | ✓ |
| 100 | 0.9284 | 0.9729 | 0.9501 | 0.9725 | 0.8284 | 0.908 | |
| 105 | 0.9241 | 0.9723 | 0.9476 | 0.9710 | 0.8271 | 0.908 | |
| **110** | 0.9269 | 0.9737 | 0.9497 | **0.9740** | **0.8306** | 0.908 | ✓ |
| 115 | 0.9294 | 0.9740 | 0.9512 | 0.9733 | 0.8295 | 0.909 | |
| 120 | 0.9272 | 0.9738 | 0.9499 | 0.9730 | 0.8301 | 0.909 | |

Shape of the curve: **72 % of final mAP50-95 was reached by epoch 10**, 98 % by
epoch 50. The last 70 epochs bought +0.018 mAP50-95, almost all of it as
**precision** (0.9048 → 0.9269) at constant recall. Between epoch 95 and 120 the
metric moves inside a ±0.001 band — the run had converged, and the last 25
epochs were confirmation rather than progress.

---

## Checkpoint selection

Three candidates were re-evaluated on the **complete** 1,048-image validation
split (not the 256-image subset), in float32:

| rank | checkpoint | mAP50-95 | mAP50 | F1 | P | R | fg mIoU |
|---:|---|---:|---:|---:|---:|---:|---:|
| **1** | `best_semantic_model_v6_2.keras` | **0.82979** | 0.97476 | 0.94970 | 0.92522 | 0.97551 | 0.90847 |
| 2 | `best_instance_model_v6_2.keras` (ep 110) | 0.82949 | 0.97466 | 0.94937 | 0.92472 | 0.97537 | 0.90802 |
| 3 | `last_model_v6_2.keras` (ep 120) | 0.82906 | 0.97392 | 0.95026 | 0.92606 | 0.97576 | 0.90834 |

Spread across all three: **0.0007 mAP50-95**. The result is a stable plateau,
not a lucky epoch — which is the useful part of this table. The winner was
copied to `best_model_v6_2_instance.keras`, the file everything downstream loads.

> **Note on `final_model_selection.json`:** its `selection_rule` field says
> "highest complete-validation instance F1". That description is **stale** —
> selection actually ranked on `mask_map50_95`. You can verify this from the
> table: `last_model` has the best F1 (0.95026) and was correctly *not* chosen.
> The behaviour is right, the recorded description is wrong.

---

## Final results — instance segmentation

Complete splits, float32, default decoder thresholds, **no TTA**.
Matching is maximum-cardinality one-to-one same-class at IoU 0.50; mAP uses
confidence-ranked greedy matching at each of 10 IoU thresholds (0.50…0.95),
integrated on 101 recall points, max 300 detections/image. Detection confidence
is the mean predicted probability of the decoded instance's class over its own
pixels.

### Headline

| | val (1,048 img) | **test (432 img, untouched)** |
|---|---:|---:|
| mask mAP50 | 0.9748 | 0.9717 |
| mask mAP75 | 0.9434 | **0.8458** |
| mask mAP50-95 | **0.8298** | **0.7463** |
| F1 @ IoU 0.50 | 0.9497 | 0.9702 |
| Precision | 0.9252 | 0.9595 |
| Recall | 0.9755 | 0.9811 |
| TP / FP / FN | 47,203 / 3,815 / 1,185 | 32,841 / 1,386 / 631 |
| Throughput | 1.45 img/s | 1.34 img/s |

**Test mAP50-95 is 0.084 below val while test F1 is higher.** That is not a
contradiction. mAP50 barely moves (0.9748 → 0.9717) but mAP75 collapses
(0.9434 → 0.8458): the model still *finds* everything on test, its masks just
fit worse, because test objects are 0.48–0.76× the linear size of training
objects. This is exactly what the dataset audit predicted before the run.

### AP by IoU threshold

| IoU | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 0.9748 | 0.9722 | 0.9691 | 0.9640 | 0.9564 | 0.9434 | 0.9126 | 0.8214 | 0.6078 | 0.1763 |
| test | 0.9717 | 0.9704 | 0.9631 | 0.9375 | 0.9044 | 0.8458 | 0.7714 | 0.6576 | 0.4134 | 0.0279 |

The whole val↔test gap opens above IoU 0.65 and is worst at 0.95 — a pure
mask-tightness deficit at small object sizes.

### Per class — validation

| class | GT | pred | TP | FP | FN | P | R | F1 | AP50 | AP75 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Rectangle` | 12,816 | 13,724 | 12,597 | 1,127 | 219 | 0.9179 | 0.9829 | 0.9493 | 0.9882 | 0.9667 | 0.8454 |
| `Rectangle_concave` | 292 | 324 | 290 | 34 | 2 | 0.8951 | 0.9932 | 0.9416 | 0.9950 | 0.9950 | 0.9328 |
| `circle` | 6,712 | 6,894 | 6,534 | 360 | 178 | 0.9478 | 0.9735 | 0.9605 | 0.9838 | 0.9386 | 0.8244 |
| `circle_full` | 28,568 | 30,076 | 27,782 | 2,294 | 786 | 0.9237 | 0.9725 | 0.9475 | 0.9320 | 0.8732 | 0.7166 |

### Per class — test

| class | GT | pred | TP | FP | FN | P | R | F1 | AP50 | AP75 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Rectangle` | 15,140 | 15,735 | 14,872 | 863 | 268 | 0.9452 | 0.9823 | 0.9634 | 0.9892 | 0.9591 | 0.8159 |
| `Rectangle_concave` | 108 | 222 | 98 | 124 | 10 | **0.4414** | 0.9074 | 0.5939 | 0.9256 | 0.9104 | **0.8105** |
| `circle` | 3,272 | 3,351 | 3,190 | 161 | 82 | 0.9520 | 0.9749 | 0.9633 | 0.9821 | **0.5452** | **0.5630** |
| `circle_full` | 14,952 | 14,919 | 14,681 | 238 | 271 | 0.9840 | 0.9819 | 0.9830 | 0.9901 | 0.9686 | 0.7959 |

Two entries in this table are the whole story of what was wrong with the run's
default configuration:

**`Rectangle_concave` precision 0.4414 but AP50-95 0.8105.** A rank-aware metric
says the model is fine; a fixed-threshold metric says it is terrible. That
combination can only mean **the cut-off is wrong, not the model**. Rendering 6
test images and reading all 94 detections confirmed it exactly:

- every correct detection scores **0.78–0.98**
- every false positive scores **0.13–0.35**
- **every detection below 0.20 has `centre_score` of exactly 0.050**

0.050 is `FALLBACK_INSTANCE_SCORE`. So every one of those false positives is a
*centre-less fallback recovery* — the decoder's last-resort path minting small
slivers (median 1,035 px) from leftover semantic pixels — and
`center_confidence = 0.10` admits all of them. The `0.05` score was added
precisely so these would rank last, which worked; nobody had then moved the
threshold to exploit it. See
[Post-training improvement chain](#post-training-improvement-chain).

**`circle` AP50 0.9821 but AP50-95 0.5630.** It finds them; it cannot outline
them at a 16 px median diameter. This is the target that dihedral TTA was
aimed at, and it is where TTA paid.

---

## Final results — semantic segmentation

Per-pixel, complete splits, default configuration.

| | val | test |
|---|---:|---:|
| Overall pixel accuracy | 0.9858 | 0.9830 |
| Mean IoU incl. background | 0.9237 | 0.8593 |
| **Foreground mean IoU** | **0.9085** | **0.8284** |
| Frequency-weighted IoU | 0.9724 | 0.9675 |
| Foreground mean Dice | 0.9518 | 0.9049 |

### Per class IoU

| class | val IoU | val P | val R | test IoU | test P | test R |
|---|---:|---:|---:|---:|---:|---:|
| `background` | 0.9845 | 0.9908 | 0.9936 | 0.9828 | 0.9924 | 0.9902 |
| `Rectangle` | 0.9084 | 0.9545 | 0.9495 | 0.8962 | 0.9297 | 0.9614 |
| `Rectangle_concave` | 0.9447 | 0.9680 | 0.9751 | 0.7965 | 0.8792 | 0.8945 |
| `circle` | 0.9113 | 0.9675 | 0.9401 | **0.7434** | 0.9129 | **0.8002** |
| `circle_full` | 0.8695 | 0.9439 | 0.9169 | 0.8776 | 0.9181 | 0.9521 |

`circle` recall drops to 0.80 on test — pixel-level confirmation of the same
small-object story the instance metrics tell. The test confusion matrix shows
84,602 `circle` pixels predicted as `Rectangle_concave` and 78,184 the other
way; those two classes are the only pair that meaningfully confuse each other.

---

## Post-training improvement chain

Three changes were applied **without retraining**, each measured, in this order.
Full write-up in `memory/session/2026-09-05-improvement-chain.md`.

### 1. Gate the fallback path (measured before touching it)

`measure_fallback.py` → `fallback_measurement_val.json`. The centre-less
fallback recoveries are overwhelmingly false positives, **but not purely**:
deleting the path outright costs 2 TP on `circle_full` and adds 2 FN. So it is
**gated, not deleted**. Measured on validation:

| gate | class-1 FP | class-2 FP | class-3 FP | class-4 FP | TP lost |
|---|---:|---:|---:|---:|---:|
| none (shipped default) | 136 | 5 | 67 | 208 | – |
| drop fallback entirely | 102 | 0 | 49 | 190 | 2 |
| score ≥ 0.3 | 93 | 0 | 40 | 184 | 2 |
| score ≥ 0.5 | 57 | 0 | 21 | 164 | 6 |
| score ≥ 0.6 | 45 | 0 | 18 | 155 | 12 |

The same commit fixed the real defect underneath: instances were being minted
from offset-assignment residue.

### 2. Joint threshold search, split-validated

`fit_thresholds.py` → `threshold_fit.json`. 400 images, 150 random draws, fitted
on one half of validation and verified on the **other half, never used for
fitting**.

| config | fit F1 | held-out F1 | held-out P | held-out R |
|---|---:|---:|---:|---:|
| shipped (0.30 / 0.10 / 2 / 0.50) | 0.9487 | 0.9506 | 0.9281 | 0.9743 |
| fitted (0.396 / 0.444 / 1 / 0.691, min area 53) | 0.9587 | **0.9611** | 0.9605 | 0.9617 |

### 3. Dihedral TTA

`tta.py`. The largest single gain, and it landed exactly where it was aimed —
**mAP75, mask tightness: +0.0314 on test**. Cost: throughput falls from
1.34 img/s to 0.28 img/s (~5×).

### Cumulative result, no retraining

| | as trained | + decoder fix | **+ TTA** | + fitted thresholds |
|---|---:|---:|---:|---:|
| test mAP50-95 | 0.7463 | 0.7499 | **0.7710** | 0.7694 |
| test mAP75 | 0.8458 | 0.8498 | **0.8812** | 0.8785 |
| test precision | 0.9595 | 0.9639 | 0.9696 | **0.9809** |
| test recall | 0.9811 | 0.9812 | 0.9810 | 0.9675 |
| val mAP50-95 | 0.8298 | 0.8301 | 0.8445 | **0.8450** |

**The fitted thresholds won on val and lost on test** (−0.0016 mAP50-95), buying
precision with recall. That is the split validation working, not a failed
experiment — per the dataset audit, the test operating point genuinely sits
elsewhere. So **two configurations ship**:

- **TTA alone** — best masks. `performance/test_evaluation/` in WSL:
  test mAP50 0.9902, mAP75 0.8812, mAP50-95 **0.7711**, F1 0.9753,
  P 0.9696, R 0.9810, fg mIoU 0.8630.
- **TTA + fitted thresholds** — fewest false alarms.
  `performance/val_evaluation/` in WSL: val mAP50 0.9792, mAP75 0.9556,
  mAP50-95 **0.8450**, F1 0.9679, P 0.9696, R 0.9662, fg mIoU 0.9125.

The fitted values are deliberately **not** baked into `model_v6_2.py`;
`reevaluate.py:51` reads them from `results/threshold_fit.json` when present.

Per-class effect of the chain on test — this is where it actually landed:

| class | AP50-95 as trained | AP50-95 + TTA | Δ |
|---|---:|---:|---:|
| `Rectangle` | 0.8159 | 0.8226 | +0.0067 |
| `Rectangle_concave` | 0.8105 | 0.8697 | **+0.0592** |
| `circle` | 0.5630 | 0.5902 | +0.0272 |
| `circle_full` | 0.7959 | 0.8019 | +0.0060 |

And `Rectangle_concave` precision on test goes **0.4414 → 0.9153** (124 FP → 10
FP) with recall rising to 1.0000 — the fallback-gate fix, doing exactly what the
diagnosis said it would.

### Deployment note

`predict_folder.py` over 112 unseen images from `_Pictures` and `ro/ro_2`:
correct detections score **0.78–0.98**, errors **0.12–0.59**. A confidence cut
near **0.6** removes essentially every error at almost no cost. On real boards
this mattered a great deal — 54 % of raw detections were phantom duplicates, and
the `deployment.py` profile plus that confidence floor cut 1,074 detections to
209 while keeping all 194 confident ones.

---

## Known issues and caveats

1. **44 images are byte-identical between val and test** (10 % of test). Exact
   md5 duplicates across train/val and train/test: **none** — the train boundary
   is clean. But because decoder thresholds are tuned on val, that 10 % of test
   is contaminated for any val-tuned setting. Treat the fitted-threshold test
   numbers accordingly.
2. **`final_model_selection.json` records a stale selection rule** — see
   [Checkpoint selection](#checkpoint-selection). Behaviour correct, description
   wrong.
3. **Label overlap**: a later polygon overwrites an earlier one in 9.3 % of train
   images (292,656 px, **104 training instances erased outright**), 8.0 % of val
   and **19.4 % of test**. `label_overlap_classification.json` classifies the
   val/test cases: mostly harmless `bleed` (248 val pairs, 2,990 px total), plus
   16 val / 36 test `nested` pairs that account for most of the shared pixels.
   Zero cross-class duplicates in either split.
4. **`Rectangle_concave` has only 680 training polygons.** Its val numbers look
   excellent (AP50-95 0.9328) and its test numbers are far shakier (0.8105, and
   just 108 GT instances). Do not read either as a stable estimate — the class is
   too small for the confidence interval to be narrow.
5. **A near-duplicate scan needs a within-split control.** An 8×8 aHash reports
   42 % train-val matches here, but it also matches 71.8 % of train against
   *itself*, so it is not discriminative and that number is an artefact. A 16×16
   dHash gives 0.0 % train-val and train-test against 20.6 % within-train.
6. **Filenames collide across splits** (all `sample_val_NNNNNN`) only because
   each split was numbered from its own counter. It is not duplication.
7. **Evaluation runs float32, training ran mixed_float16.** Any timing comparison
   between the two is meaningless.
8. **XLA cannot be enabled** on this model as written (see
   [Runtime environment](#runtime-environment)).

---

## Reproducing and re-evaluating

```bash
# environment — env.sh handles the two CUDA faults described above
source ~/envs/pcb62/bin/activate && source env.sh

# rebuild the prepared arrays (~1.9 min, thread-pooled)
PCB_DATASET_ROOT=/mnt/c/Users/u117134/Desktop/Data_preprocessing/1000_images/Split_Data \
PCB_ARRAY_DIR=~/data/pcb_v62_arrays \
RUN_MODE=prepare python model_v6_2.py

# verify the pipeline before spending 80 hours
RUN_MODE=selftest python model_v6_2.py

# train from scratch, 120 epochs, ~3.3 days on this hardware
./session.sh start scratch          # tmux session `pcb`, dashboard on :8088

# re-evaluate the shipped checkpoint
python reevaluate.py                # reads results/threshold_fit.json if present
```

`PCB_ARRAY_DIR` overrides `ARRAY_DIR` independently of `DATASET_ROOT`
specifically so the arrays can live on ext4 while the dataset stays on `/mnt/c`.

Exact reproduction of *this* run also needs `initial_random.weights.h5` (in this
folder) and seed 42.

### If a future run needs to be faster

The FLOP analysis points at one place: the centre and offset head stacks
(`center_features_1/2`, `offset_features_1/2`) are **22.6 % of all MACs**, all of
it at 256². The uplift plan's P4 already argues that centre+offset is
over-engineered for data this regular.

---

## Related material

| | |
|---|---|
| `memory/session/2026-09-01-rtx-scratch-training-launch.md` | making the GPU train, and the launch |
| `memory/session/2026-09-05-scratch-run-complete.md` | the run finishing, and the fallback diagnosis |
| `memory/session/2026-09-05-improvement-chain.md` | the no-retraining chain |
| `memory/session/2026-09-08-step7-and-deployment-profile.md` | V6.2 vs YOLO11, and the deployment profile |
| `memory/dataset-split-data-audit.md` | the measured dataset audit quoted throughout |
| `memory/v6-2-uplift-plan.md` | the phased plan this run implements |
| `report_v62_vs_yolo11.html` | the V6.2 vs YOLO11 comparison report |
