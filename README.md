# Model_V6

**Model V6.2 — PCB instance segmentation**

Four-class instance segmentation of PCB features, trained **from random
initialisation** — no ImageNet weights, no warm start, nothing downloaded.
One file, `model_v6_2.py`, owns every stage that has to agree with every other
stage: dataset preparation, augmentation, targets, training, decoding and
evaluation.

| | |
|---|---|
| Task | 4-class instance segmentation (`Rectangle`, `Rectangle_concave`, `circle`, `circle_full`) |
| Input | 512×512, aspect-ratio-preserving letterbox |
| Outputs | full-res semantic map · half-res class centre heatmaps · half-res centre-offset vectors · full-res instance boundary |
| Parameters | 6,417,683 |
| Target hardware | RTX PRO 1000 Blackwell laptop, 8 GB, `sm_120` |
| Training cost | ~35 min/epoch × 120 epochs ≈ 2.8 days |

## Results

Measured on the held-out test split (432 images, 77.5 objects/image). Each row
is a real configuration you can reproduce with the commands below.

| Configuration | test mAP50-95 | test mAP50 | test F1 | val mAP50-95 |
|---|---:|---:|---:|---:|
| As trained (fallback path ungated) | 0.7463 | — | — | 0.8298 |
| Fallback gated — `reevaluation_smoke_plain` | 0.7679 | 0.9884 | 0.9710 | 0.8429 |
| **+ dihedral TTA — shipped operating point** | **0.7710** | 0.9902 | 0.9753 | 0.8445 |
| + fitted thresholds — `reevaluation_final` | 0.7694 | 0.9809 | 0.9742 | 0.8450 |

Against a compute-matched YOLO11l-seg baseline on the same 432 images, same
metric implementation, 1,000 paired bootstrap draws:

| | mAP50-95 | 95% interval |
|---|---:|---|
| Model V6.2 | 0.7711 | [0.7664, 0.7762] |
| YOLO11l-seg | 0.7231 | [0.7189, 0.7270] |
| **Difference** | **+0.0480** | [+0.0438, +0.0526], positive in 100% of draws |

mAP50 is a tie (0.9902 vs 0.9882) — the gap is entirely in mask tightness, not
in whether objects are found.

> **Quote these numbers with the caveat.** 44 images are byte-identical between
> val and test (10% of test), and decoder thresholds are fitted on val. Anything
> tuned on validation contaminates that 10%. See [Honest caveats](#honest-caveats).

---

## Install — from a fresh `git clone`

Every command below runs **inside WSL2 (Ubuntu)**, not in PowerShell.

```bash
# 1. Clone
git clone https://github.com/Jenit88/Model_V6.git
cd Model_V6

# 2. Create the dedicated venv (Python 3.11 — not 3.12, not 3.13)
sudo apt update && sudo apt install -y python3.11 python3.11-venv
python3.11 -m venv ~/envs/pcb62
~/envs/pcb62/bin/pip install --upgrade pip

# 3. TensorFlow with its bundled CUDA stack, then the rest
~/envs/pcb62/bin/pip install "tensorflow[and-cuda]==2.21.0"
~/envs/pcb62/bin/pip install \
    opencv-python-headless numpy scipy matplotlib pandas pillow psutil

# 4. tmux, for the detached long-running session
sudo apt install -y tmux

# 5. Restore the executable bit (a Windows checkout usually drops it)
chmod +x *.sh

# 6. Point the paths at your machine — edit these three lines in env.sh:
#      PCB_DATASET_ROOT   source PNGs + YOLO polygons
#      PCB_ARRAY_DIR      prepared memmaps  (MUST be on ext4, not /mnt/c)
#      PCB_MODEL_OUTPUT_DIR
nano env.sh
```

### Verify before you commit three days to it

```bash
source ./env.sh

# GPU must be listed. If this prints [] you are about to train on the CPU.
$PCB_PYTHON -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# expect: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

./run.sh selftest      # ~1 min, no dataset/GPU/checkpoint needed
```

`selftest` asserts the decode, target, ranking and augmentation contracts on
synthetic data. If it passes and the GPU is listed, the install is good.

### Exact versions this was built and measured on

| package | version |
|---|---|
| Python | 3.11.15 |
| tensorflow | 2.21.0 |
| keras | 3.15.1 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| opencv-python-headless | 5.0.0.93 |
| matplotlib | 3.11.1 |
| pandas | 3.0.5 |
| pillow | 12.3.0 |
| psutil | 7.2.2 |
| tensorboard | 2.21.0 |
| nvidia-cudnn-cu12 | 9.25.1.1 |
| nvidia-cublas-cu12 | 12.9.2.10 |

There is no `requirements.txt`; the `pip install` lines above are the
specification. `opencv-python-headless` is deliberate — the GUI build pulls in
Qt and is not needed.

---

## Quick start

```bash
cd Model_V6                    # wherever you cloned it

./run.sh selftest              # ~1 min. No dataset, GPU or checkpoint needed.
./session.sh start prepare     # one-off, ~2 min: PNG + polygons -> memmaps
./session.sh start scratch     # the 2.8-day training run
./session.sh attach            # watch it   (detach: Ctrl-b then d)
```

Everything long-running goes into a detached `tmux` session called `pcb`, so a
closed terminal or a disconnected VS Code window cannot kill a multi-day run.

---

## 1. Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability 12.0 (Blackwell), 8 GB. **Required for training.** |
| CPU | 16 threads (12 used by the loader, 16 by `prepare`) |
| RAM | 31.5 GB host; WSL given 20 GB via `C:\Users\<you>\.wslconfig` |
| Disk | 13 GB source dataset + 9.7 GB prepared arrays + ~1 GB per checkpoint set |
| OS | Windows 11 + WSL2 (Ubuntu) |
| Python | 3.11, TensorFlow 2.21.0, Keras 3.15.1, in `~/envs/pcb62` |

**Only the NVIDIA GPU can train this.** TensorFlow has no XPU backend —
`intel-extension-for-tensorflow` is discontinued and never supported a
Keras 3 / TF 2.21 stack — so an Intel Arc iGPU cannot run a single op of this
graph. That is not wasted capacity: the iGPU drives the desktop, which is why
the whole 8 GB of the RTX stays free for training.

### Two environment facts that silently break GPU training

Both were found by diagnosis on this machine, and both are handled by `env.sh`.
You do not need to do anything — but if you ever run Python directly instead of
through `./run.sh`, you will hit them.

1. **`opencv` overwrites `LD_LIBRARY_PATH`.** After that, TensorFlow's `dlopen`
   of `libcusolver.so.11` fails even though the file is present in
   `site-packages/nvidia/cusolver/lib`. TF prints only
   `Cannot dlopen some GPU libraries` and **silently trains on the CPU** — an
   entire run can complete without the GPU ever being used. `env.sh` exports
   every `site-packages/nvidia/*/lib` up front so load order stops mattering.

2. **TF 2.21 ships no cubins for `sm_120`.** Every kernel is JIT-compiled from
   PTX on first use; TF's own warning says this "could take 30 minutes or
   longer". The result is cached on disk, so it is a one-time cost per machine
   — *provided the cache is big enough*. The 256 MB default is not, so `env.sh`
   sets `CUDA_CACHE_MAXSIZE=4 GiB`. Convolutions are unaffected: the bundled
   cuDNN 9.25 has native Blackwell kernels.

**Always launch through `./run.sh` or `./session.sh`.** They source `env.sh`
first. Bare `python model_v6_2.py` is the documented way to get a silent
CPU-only run.

## 2. Dataset

Expected layout under `PCB_DATASET_ROOT`, YOLO polygon format:

```
Split_Data/
  images/{train,val,test}/*.png
  labels/{train,val,test}/*.txt      # YOLO polygons, class ids 1-4
```

Class id mapping is `YOLO_TO_SEMANTIC_ID = {1:1, 2:2, 3:3, 4:4}`; semantic id 0
is background.

### What is actually in it — measured, not assumed

6,160 images (train 4,680 / val 1,048 / test 432), 13 GB source. Source images
run 1024×1536 to 2588×1940, so letterbox bars occupy up to a third of the canvas.

| class | train polygons | share | class weight |
|---|---:|---:|---:|
| `circle_full` (4) | 76,612 | 52.6% | 4.64 |
| `Rectangle` (1) | 57,857 | 39.7% | 3.44 |
| `circle` (3) | 12,000 | 8.2% | 6.10 |
| `Rectangle_concave` (2) | **680** | **0.47%** | 7.74 |

Two properties drive most design decisions in this repo:

* **`Rectangle_concave` is the structural limit.** 680 training polygons, and
  they are large (median 9,104 px). Large *and* rare is the hard combination.
  No architecture change substitutes for more labels.
* **Object scale shifts hard across splits.** Median equivalent diameter for
  `Rectangle` falls 35.7 → 27.7 → 21.9 px from train → val → test, and objects
  per image rises 32.4 → 46.2 → **77.5**. The test split is denser and much more
  zoomed-out than train. V5's zoom-to-fill is always ≥ 1.0, so training never
  showed the model a test-scale object — which is why the zoom-out branch
  (`ZOOM_OUT_RANGE = (0.45, 1.00)`, p=0.5) exists.

Full audit: `memory/dataset-split-data-audit.md`.

## 3. The workflow, end to end

### Stage 0 — `selftest`

```bash
./run.sh selftest
```

Asserts the decode, target, ranking and augmentation contracts on synthetic
data. **No dataset, GPU or checkpoint needed**, ~1 min. Run this after any edit
to `model_v6_2.py`; it is the cheapest way to find that you broke the target
generator or the augmentation cycle.

### Stage 1 — `prepare` (one-off, ~2 min)

```bash
./session.sh start prepare
```

Reads the PNGs and YOLO polygons once and writes letterboxed, memmapped arrays
to `PCB_ARRAY_DIR` (9.7 GB). Also writes
`dataset_and_augmentation_preview.png` so you can see labels and augmentation
before committing three days to them.

> **The arrays must live on ext4, not `/mnt/c`.** Every training step
> random-reads that memmap, and the 9p mount is roughly an order of magnitude
> slower — it will starve the GPU. The source dataset can stay on the Windows
> drive; it is read once.

### Stage 2 — `benchmark` (optional)

```bash
./session.sh start benchmark          # sweeps batch 2,4,6,8
PCB_BENCH_BATCHES=2,3 ./run.sh benchmark
```

Times real training steps and reports peak VRAM per batch size, each in its own
subprocess so an OOM is survivable. Writes `benchmark_results.json`.

What it already measured on this machine, and why the settings are what they are:

| setting | measurement |
|---|---|
| batch 2 | 752 ms/step, 3.7 GiB peak |
| batch 4, 6, 8 | **OOM** — Keras `GroupNormalization` materialises `[N, groups, HW, C/groups]` moments |
| `steps_per_execution 8` | −2.5% |
| per-class IoU during training | −1.3% |
| gradient accumulation 2 | −4.2% per image, but doubles effective batch — fewer AdamW+EMA applies over 6.4M params more than pays for it |
| XLA | does not compile: `ResizeBilinearGrad` mixes f32/f16 under `mixed_float16` |

Batch 2 costs nothing statistically here: the model uses `GroupNormalization`,
not BatchNorm, so normalisation is per-sample either way.

### Stage 3 — Start the training run

```bash
./session.sh start scratch          # detached: survives a closed terminal
```

or, to watch it in the foreground and lose it when the terminal closes:

```bash
./run.sh scratch
```

**Always use `session.sh` for a real run.** It puts the job in a detached tmux
session, so a closed terminal, a disconnected VS Code window, or a dropped SSH
connection cannot kill 2.8 days of work.

To aim a run at a directory other than `env.sh`'s default — which is what you
want for every run after the first:

```bash
PCB_ARRAY_DIR=$HOME/data/pcb_v62_arrays_repaired \
PCB_MODEL_OUTPUT_DIR=$HOME/Models/my_new_run \
  ./session.sh start scratch
```

`session.sh` forwards those two variables into the tmux pane explicitly and
prints them back — read that line and confirm it says what you meant.

#### Check these four things in the first ten minutes

```bash
./session.sh attach            # window 'work' is the run; Ctrl-b then d to detach
nvidia-smi                     # ~4 GB used, 60-95% util once stepping
tail -f logs/scratch.log
```

| check | healthy |
|---|---|
| `overrides :` line at launch | names the array + output dir you intended |
| GPU memory | ~4.2 GB, **not** 0 MiB (0 means it fell back to CPU) |
| step time | settles near **0.9 s/step**; the first minutes read ~2 s/step because Keras' running average is dominated by PTX JIT warmup — do not size the run from it |
| epoch line | `Epoch 1/120` (or the resumed epoch), 2,340 steps |

#### What the schedule is, and what you must not change mid-run

120 epochs, ~35 min/epoch, ≈ 2.8 days. Configuration comes from
`train_rtx.py::apply_laptop_configuration` — batch 2 × 2 accumulation,
`mixed_float16`, cosine LR 3e-4 → 1e-6 over exactly `EPOCHS`, 6 warmup epochs,
weight decay 1e-4, 6 loader threads.

Two constraints on `EPOCHS` that are easy to violate:

* the cosine schedule anneals over exactly `EPOCHS`, so **shortening a run
  mid-flight leaves the model un-annealed**;
* it must exceed `AUGMENTATION_CYCLE_LENGTH` (95) for every image to see every
  V5 augmentation mode.

Override for a short experiment, never for a real run:

```bash
PCB_EPOCHS=10 PCB_MODEL_OUTPUT_DIR=$HOME/Models/smoke ./session.sh start scratch
```

#### What lands in `PCB_MODEL_OUTPUT_DIR`

| file | what |
|---|---|
| `best_instance_model_v6_2.keras` | best on `INSTANCE_SELECTION_METRIC` (`mask_map50_95`) |
| `best_semantic_model_v6_2.keras` | best on semantic mIoU |
| `best_model_v6_2_instance.keras` | the authoritative copy, chosen at the end |
| `fit_backup/` | Keras `BackupAndRestore` state — this is what makes a resume free |
| `training_log.csv` | per-epoch metrics |
| `instance_checkpoint_history.json` | one record per instance evaluation (every 5 epochs) |
| `TRAINING_REPORT.md` | rewritten every epoch |
| `reports/epoch_NNNN.md` | snapshot every 50 epochs |
| `final_model_selection.json` | which candidate won, and on what |

Selection is on **mAP50-95**, not instance F1: F1 at IoU 0.50 cannot tell a
barely-passing mask from an exact one. At the end, every saved candidate is
re-scored on the *complete* validation split; test is reported separately and
never selected on.

#### Stopping a run

```bash
./watchdog.sh stop            # tell the supervisors this was deliberate
./session.sh stop             # kill the tmux session and everything in it
```

Stop the watchdog **first**. Killing the session alone leaves a supervisor that
will faithfully restart the job you just stopped.

### Stage 3b — Fine-tuning a finished checkpoint

`run.sh` deliberately offers scratch only — `train_rtx.py` says so explicitly,
because this project's claim is scratch training. Fine-tuning goes through
`model_v6_2.train(fine_tune=True)`, which uses a fresh optimizer, a much lower
LR, and its own output directory.

The defaults it reads (`model_v6_2.py:234-251`):

| constant | default | note |
|---|---|---|
| `FINE_TUNE_SOURCE_MODEL` | `MODEL_FOR_INFERENCE` | **points at a V5-transfer run that does not exist in a fresh clone — you must override it** |
| `FINE_TUNE_OUTPUT_DIR` | `~/Models/Model_v6_2_fine_tuned` | never the source run's directory |
| `FINE_TUNE_ARRAY_DIR` | `ARRAY_DIR` | point elsewhere for domain adaptation |
| `FINE_TUNE_EPOCHS` | 160 | |
| `FINE_TUNE_LEARNING_RATE` | 3e-5 → 3e-7 | 10× below the scratch LR |
| `FINE_TUNE_WARMUP_EPOCHS` | 2 | |
| `FINE_TUNE_WEIGHT_DECAY` | 5e-5 | |
| `FINE_TUNE_INSTANCE_CHECKPOINT_EVERY_N_EPOCHS` | 5 | |
| `FINE_TUNE_INSTANCE_CHECKPOINT_MAX_IMAGES` | 0 | 0 = evaluate every validation image |
| `FINE_TUNE_INSTANCE_EARLY_STOPPING_PATIENCE_EVALUATIONS` | 8 | |

Create `fine_tune_rtx.py` next to `train_rtx.py`:

```python
#!/usr/bin/env python3
"""Fine-tune a finished V6.2 checkpoint. Launch through ./run.sh-style env."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("model_v6_2", HERE / "model_v6_2.py")
m = importlib.util.module_from_spec(spec)
sys.modules["model_v6_2"] = m
spec.loader.exec_module(m)

# The same measured laptop configuration the scratch run uses.
m.REQUIRE_GPU = True
m.USE_MIXED_PRECISION = True
m.BATCH_SIZE = 2
m.GRADIENT_ACCUMULATION_STEPS = 2
m.STEPS_PER_EXECUTION = 8
m.PER_CLASS_IOU_METRICS_DURING_TRAINING = False
m.USE_XLA_JIT = False
m.DATA_LOADER_WORKERS = 6

# What to refine, and where it lands. The default source points at a V5
# transfer run; override it or the load fails.
m.FINE_TUNE_SOURCE_MODEL = (
    Path.home() / "Models/Model_v6_2_scratch_rtx/best_model_v6_2_instance.keras"
)
m.FINE_TUNE_OUTPUT_DIR = Path.home() / "Models/Model_v6_2_fine_tuned"
m.FINE_TUNE_ARRAY_DIR = m.ARRAY_DIR      # or another V6-compatible prepared set
m.FINE_TUNE_EPOCHS = 160

m.validate_spatial_configuration()
m.validate_augmentation_configuration()
m.train(fine_tune=True)
```

Run it inside the session so it survives a closed terminal:

```bash
source ./env.sh
$PCB_PYTHON -u fine_tune_rtx.py 2>&1 | tee -a logs/fine_tune.log

# or detached:
tmux new-session -d -s ft "source ./env.sh && \
  $PCB_PYTHON -u fine_tune_rtx.py 2>&1 | tee -a logs/fine_tune.log"
```

The fine-tuned result is **not automatically authoritative**: it is included in
the same final candidate comparison as every other checkpoint, so a degraded
fine-tuned epoch cannot win merely by being trained last.

### Stage 3c — the other entry point: `RUN_MODE`

`model_v6_2.py` has its own dispatcher with modes `run.sh` does not expose:
`selftest`, `report`, `prepare`, `preview`, `train`, `transfer`, `fine_tune`,
`predict`, `evaluate`. It is selected by editing `RUN_MODE` at
`model_v6_2.py:173` and running the file directly.

> **Trap:** `RUN_MODE` is currently `"transfer"`. Running
> `python model_v6_2.py` in a fresh clone therefore attempts a Model V5 warm
> start against a checkpoint you do not have — and, without `env.sh` sourced,
> does it on the CPU. Prefer `./run.sh` for everything it covers.

### Stage 4 — watching it

```bash
./session.sh attach       # tmux: work / dash / tb / gpu / dog windows
./session.sh status       # one screen, no attach
./run.sh report           # progress summary, safe while training
```

* **Dashboard** — <http://localhost:8088>. Live curves, full parameter set,
  per-epoch instance metrics, GPU utilisation/power/clocks.
* **TensorBoard** — <http://localhost:6006>.
* **Files** — `TRAINING_REPORT.md` in the output dir is rewritten every epoch;
  `reports/epoch_NNNN.md` snapshots every 50; `training_log.csv` has per-epoch
  metrics; `instance_checkpoint_history.json` has one record per instance
  evaluation (every 5 epochs).

> Read per-epoch mAP from **`instance_checkpoint_history.json`**, not from
> `training_log.csv`. Eval epochs log extra keys, so the CSV is ragged — a fixed
> column index reads a *different metric* on non-eval rows.

### Stage 5 — surviving a crash

This project has lost runs to two distinct failures, which need two different
mechanisms. Both are now automatic.

| layer | lives in | catches |
|---|---|---|
| `watchdog.sh` (tmux window `dog`) | inside WSL | training dies, machine stays up (CUDA context loss — WSL2's paravirtualised GPU is prone to it) |
| `pcb_supervisor.sh` (Windows logon task) | outside WSL | host crash/reboot, which takes the WSL VM and the watchdog with it |

```bash
./pcb_supervisor.sh status     # report, change nothing
./pcb_supervisor.sh            # resume/ensure the run and its watchdog
./watchdog.sh stop             # stand down deliberately (creates logs/.watchdog-stop)

# install / remove the Windows logon task (no admin rights needed)
powershell -ExecutionPolicy Bypass -File register_pcb_watchdog.ps1
powershell -ExecutionPolicy Bypass -File register_pcb_watchdog.ps1 -Remove
```

The logon task is scaffolding for one run, not a permanent fixture — remove it
once the run is scored. Firing on every logon is safe because
`pcb_supervisor.sh` is idempotent: if training is already running, already
finished, or deliberately stopped, it does nothing, so it can never stack two
jobs onto the same 8 GB GPU.

Resuming is safe and near-free: `BackupAndRestore` continues from the last
completed epoch and `AugmentationEpochSyncCallback` restores the deterministic
augmentation cycle position, so **a resumed run is not a different experiment**.
A crash typically costs ~25 minutes and no epochs.

Both supervisors refuse to act when they should: schedule complete, stop file
present, or already running. `watchdog.sh` gives up after 12 restarts, because a
crash loop is a bug to read rather than paper over.

**Which run is supervised is set in one place**, `active_run.env`:

```bash
PCB_ARRAY_DIR=/home/u117134c/data/pcb_v62_arrays_repaired
PCB_MODEL_OUTPUT_DIR=/home/u117134c/Models/Model_v6_2_tier1_rtx
```

This exists because `env.sh` defaults `PCB_MODEL_OUTPUT_DIR` to a run that has
already *finished*. Anything that starts training without overriding it aims at
that finished directory.

### Stage 6 — evaluating a finished checkpoint

```bash
PCB_EVAL_TAG=mytag ./run.sh reevaluate           # both splits, current decoder
PCB_USE_TTA=1 PCB_EVAL_TAG=tta ./run.sh reevaluate
```

Writes to `performance/<split>_<tag>/` so the original results stay intact and
two decoders can be compared directly. This is how every "no-retraining"
improvement below was measured.

### Stage 7 — improving without retraining

```bash
./run.sh tta-selfcheck                            # verify the TTA algebra first
PCB_FIT_IMAGES=400 PCB_FIT_DRAWS=150 ./run.sh fit-thresholds
./improve.sh                                      # the whole chain, unattended
./improve.sh status
```

* **Dihedral TTA** (`tta.py`). Every class here is symmetric under the full
  dihedral group, so all eight transforms are exactly label-preserving. The trap
  this file exists for: **the offset head is a vector field, not a raster** —
  un-rotating the image grid is only half the inverse, every `(dx, dy)` must be
  rotated too, or the averaged field points in eight directions and the decoder
  groups nothing. `self_check()` verifies the inverse against the project's own
  target generator, not against itself. Worth **+0.0314 mAP75**.
* **Threshold fitting** (`fit_thresholds.py`). The five decoder thresholds were
  still V5's. The search halves validation by image index, scores candidates on
  one half and re-scores the winner on the **held-out** half; `improve.sh` reads
  the held-out gain back and refuses to apply a fit that did not survive.
  Predictions are cached, so 200 candidates cost one forward pass per image, not
  200.

### Stage 8 — running on real, unlabelled boards

```bash
source ./env.sh

PCB_SOURCE=/mnt/c/.../_Pictures PCB_SAMPLE=40 PCB_PREDICT_OUT=results/out \
  $PCB_PYTHON predict_folder.py

$PCB_PYTHON predict_pictures.py <source_dir> <output_dir>   # overlays only, resumable
```

Both apply the **deployment profile** automatically. With no ground truth there
is no precision to quote, so the honest signal is the score distribution: a
healthy run puts detections at 0.8–1.0 and leaves the 0.1–0.3 band empty.

## Operating points — and why there are two

`deployment.py` exists because benchmarks and real boards want genuinely
different answers from the same model.

Measured on 16 real boards where the model was reported as failing:

| score band | shipped | deployment profile |
|---|---:|---:|
| 0.2 – 0.4 | 581 | 13 |
| 0.6 – 1.0 | 187 | 194 |
| **total** | **1,074** | **299** |

54% of all detections sat in the 0.2–0.4 band and were **duplicates** — a
spurious low-confidence `Rectangle` on top of a pad already found correctly. The
class mix read 80.4% `Rectangle` against 39.7% in training. The profile removes
98% of that band while the confident detections go slightly *up*, because
instances stop fragmenting against each other.

**This is not simply the better configuration.** On the labelled test split
these thresholds score marginally *worse* — 0.7710 → 0.7694 — because they buy
precision with recall, and mask AP is rank-aware: a weak duplicate that ranks
last costs a benchmark almost nothing. Unlabelled deployment has no ranking to
hide behind; every retained instance is drawn and counted.

| use | operating point |
|---|---|
| anything whose number gets published | **shipped** thresholds |
| anything that looks at a real board | **deployment** profile |

```python
from deployment import apply_deployment_profile
applied = apply_deployment_profile(model_module)
```

`PCB_DEPLOYMENT_PROFILE=0` opts out; `PCB_MIN_CONFIDENCE` overrides the 0.50
confidence floor. That floor exists to clear centre-less fallback recoveries,
which score a flat `FALLBACK_INSTANCE_SCORE = 0.05` and survive the fitted
thresholds because those act on centre evidence this path has none of.

| threshold | shipped | fitted |
|---|---:|---:|
| `semantic_confidence` | 0.30 | 0.396 |
| `center_confidence` | 0.10 | 0.444 |
| `center_nms_radius` | 2 | 1 |
| `minimum_instance_area` | 13 | 53 |
| `boundary_confidence` | 0.50 | 0.691 |

## Configuration

Every path is an environment variable, so the same file runs on the training box
and anywhere the dataset has been copied without editing paths back and forth.

| variable | default | what |
|---|---|---|
| `PCB_DATASET_ROOT` | `/mnt/c/.../Split_Data` | source PNGs + polygons, read once |
| `PCB_ARRAY_DIR` | `~/data/pcb_v62_arrays` | prepared memmaps — **must be ext4** |
| `PCB_MODEL_OUTPUT_DIR` | `~/Models/Model_v6_2_scratch_rtx` | checkpoints, logs, reports |
| `PCB_ENV` / `PCB_PYTHON` | `~/envs/pcb62` | the venv |
| `PCB_TMUX_SESSION` | `pcb` | session name |
| `PCB_EPOCHS`, `PCB_BATCH_SIZE`, `PCB_LEARNING_RATE`, `PCB_STEPS_PER_EXECUTION`, `PCB_GRADIENT_ACCUMULATION_STEPS`, `PCB_USE_XLA_JIT` | see `train_rtx.py` | override for sweeps |
| `PCB_BENCH_BATCHES` | `2,4,6,8` | batch sizes the benchmark sweeps |
| `PCB_DEPLOYMENT_PROFILE`, `PCB_MIN_CONFIDENCE` | `1`, `0.50` | deployment operating point |
| `PCB_USE_TTA`, `PCB_EVAL_TAG` | — | evaluation |
| `PCB_APPLY_FITTED_THRESHOLDS` | — | set by `improve.sh` only when the fit survived its held-out half |
| `PCB_FIT_IMAGES`, `PCB_FIT_DRAWS` | — | threshold search size |
| `PCB_SOURCE`, `PCB_SAMPLE`, `PCB_PREDICT_OUT`, `PCB_SAMPLE_SEED` | — | folder prediction |

> A tmux pane inherits the **tmux server's** environment, captured whenever that
> server first started — possibly days ago, for an unrelated session. Exporting a
> variable in the calling shell does **not** reach the pane. `session.sh` and
> `watchdog.sh` therefore forward the three path variables explicitly and print
> them. This is not theoretical: it once pointed a fresh run at a finished run's
> output directory, and once made a watchdog read the wrong `training_log.csv`,
> conclude 120/120 one minute after starting, and stand down — leaving a
> 2.8-day run unsupervised while reporting itself healthy.

## Repository map

```
model_v6_2.py           the model — every stage lives here

env.sh                  environment; MUST be sourced before any Python
run.sh                  single entry point for every mode
session.sh              detached tmux session holding long-running jobs
watchdog.sh             restart on process death (inside WSL)
pcb_supervisor.sh       restart after host reboot (Windows logon task)
register_pcb_watchdog.ps1
active_run.env          which run the supervisors keep alive
train_rtx.py            measured laptop configuration + benchmark harness
train_wsl.py, train_mac.py

reevaluate.py           re-score a checkpoint with the current decoder
fit_thresholds.py       split-validated decoder threshold search
tta.py                  dihedral TTA (+ algebra self-check)
improve.sh              the whole no-retraining chain, unattended
deployment.py           the real-board operating point
predict_folder.py       sample a folder, report the score distribution
predict_pictures.py     overlays only, resumable
dashboard.py            live dashboard on :8088
```

## Command reference

| goal | command |
|---|---|
| verify the install | `./run.sh selftest` |
| build the arrays | `./session.sh start prepare` |
| **start training** | `./session.sh start scratch` |
| start training elsewhere | `PCB_MODEL_OUTPUT_DIR=... ./session.sh start scratch` |
| **fine-tune** | `$PCB_PYTHON -u fine_tune_rtx.py` (see Stage 3b) |
| resume after any crash | `./pcb_supervisor.sh` |
| watch live | `./session.sh attach` · :8088 · :6006 |
| progress, without attaching | `./run.sh report` · `./session.sh status` |
| stop deliberately | `./watchdog.sh stop` then `./session.sh stop` |
| measure step time / VRAM | `./session.sh start benchmark` |
| re-score a checkpoint | `PCB_EVAL_TAG=t ./run.sh reevaluate` |
| fit decoder thresholds | `./run.sh fit-thresholds` |
| whole no-retrain chain | `./improve.sh` |
| predict a folder | `$PCB_PYTHON predict_folder.py` |

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `tf.config.list_physical_devices('GPU')` prints `[]` | `env.sh` not sourced, or CUDA packages missing | `source ./env.sh`; reinstall with `tensorflow[and-cuda]==2.21.0` |
| Training runs but GPU is idle | `opencv` clobbered `LD_LIBRARY_PATH`; TF fell back to CPU | launch via `./run.sh`, never bare `python` |
| First step takes 30+ minutes | PTX JIT for `sm_120`, cache too small | `env.sh` sets `CUDA_CACHE_MAXSIZE=4 GiB`; it is a one-time cost |
| `Permission denied: ./run.sh` | executable bit lost in a Windows checkout | `chmod +x *.sh` |
| OOM above batch 2 | `GroupNormalization` moment tensors | keep batch 2 + accumulation 2 |
| Epoch time far above ~35 min | arrays on `/mnt/c` instead of ext4 | move `PCB_ARRAY_DIR` to `~` |
| Run stopped, nothing restarted it | no logon task registered | `register_pcb_watchdog.ps1`, then `./pcb_supervisor.sh` |
| Resume starts from epoch 1 | no `fit_backup/` in the output dir | investigate before starting — the supervisor refuses this on purpose |
| Watchdog says "120/120, standing down" on a fresh run | it is reading a *different* run's `training_log.csv` | check `active_run.env`; the watchdog log names the directory it watches |
| Keras reports ~2 s/step early | running average dominated by PTX JIT warmup | time a real interval instead; do not size a run from it |
| `can't find window: work` in the supervisor log | tmux matches a target session by **prefix** — `-t pcb` also matches `pcbsup` | use exact match (`=pcb`); fixed in `session.sh`, `watchdog.sh`, `pcb_supervisor.sh` |
| Fine-tune fails loading the source model | `FINE_TUNE_SOURCE_MODEL` defaults to a V5-transfer run | override it — see Stage 3b |
| A stopped run restarts itself | the watchdog is still up | `./watchdog.sh stop` **before** `./session.sh stop` |
| Real boards show ~54% duplicate detections | benchmark operating point on unlabelled data | apply the deployment profile |

## Honest caveats

Carry these with any number from this repo.

1. **44 images are byte-identical between val and test** — 10% of test. Decoder
   thresholds are fitted on val, so that 10% of test is contaminated. Exact
   duplicates across *train*/val and train/test: none.
2. **Test is not train.** Objects are ~0.6× the diameter and 2.4× as dense.
   A model that looks fine on val can still be out of distribution on test.
3. **`Rectangle_concave` has 680 training polygons.** Its metrics move a lot
   between runs, and no architecture change fixes that.
4. **The published 0.7463 / 0.7710 were measured against damaged labels.**
   19.4% of test had overlapping polygons before the Tier-4 repair. "Old vs new"
   is not like-for-like until the old checkpoint is re-scored on repaired GT.
5. **A near-duplicate scan with an 8×8 aHash reports 42% train-val overlap and
   is an artifact** — it matches 71.8% of train against train itself. A 16×16
   dHash gives 0.0%. Always report the within-split control alongside.
6. **Measure a path before you threshold it.** The false positives that looked
   like a decoder bug turned out to be centre-less fallback recoveries all
   scoring exactly 0.05 — one measurement, not a threshold sweep, settled it.
