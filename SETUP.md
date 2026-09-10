# Running model_v6.2 on this laptop

Everything below was measured on this machine, not assumed.

## Hardware, and what can actually train

| | |
|---|---|
| CUDA GPU | NVIDIA RTX PRO 1000 Blackwell Laptop, **8 GB**, compute capability **12.0**, 50 W |
| Second GPU | Intel Arc Pro 140T (integrated) |
| CPU | Core Ultra 7 265H, 16 threads |
| Host RAM | 31.5 GB (WSL is given 20 GB by `C:\Users\u117134\.wslconfig`) |

**Only the NVIDIA GPU can train this model.** TensorFlow has no XPU backend —
Intel's `intel-extension-for-tensorflow` is discontinued and never supported a
Keras 3 / TF 2.21 stack — so the Arc iGPU cannot run a single op of this graph.
It is not idle capacity being wasted: it drives the desktop, which is why the
whole 8 GB of the RTX stays available for training. The 16 CPU threads are used,
by the data loader (12 threads) and by dataset preparation (16).

## Two environment facts that silently break GPU training

Both were found by diagnosis here, and both are handled by `env.sh`.

1. **`opencv` overwrites `LD_LIBRARY_PATH`**, after which TensorFlow's `dlopen`
   of `libcusolver.so.11` fails even though the file is right there in
   `site-packages/nvidia/cusolver/lib`. TensorFlow then prints only
   `Cannot dlopen some GPU libraries` and **silently trains on the CPU**. This
   is why the pre-existing `~/miniconda3/envs/tf` environment reported no GPU.

2. **TensorFlow 2.21 ships no CUDA binaries for compute capability 12.0.**
   Every kernel is JIT-compiled from PTX on first use — TF's own warning says
   this "could take 30 minutes or longer". The result is cached on disk, so it
   is a one-time cost per machine, provided the cache is large enough; the
   256 MB default is not. `env.sh` sets `CUDA_CACHE_MAXSIZE=4 GiB`.
   Convolutions are unaffected: cuDNN 9.2.5 has native Blackwell kernels.

## Layout

| path | what |
|---|---|
| `~/envs/pcb62` | dedicated venv: Python 3.11, TF 2.21.0, Keras 3.15.1 |
| `/mnt/c/.../1000_images/Split_Data` | source PNGs + YOLO polygons, read once |
| `~/data/pcb_v62_arrays` | prepared memmaps, 9.7 GB, **on ext4** |
| `~/Models/Model_v6_2_scratch_rtx` | checkpoints, logs, reports, previews |

The arrays must not live under `/mnt/c`: every training step random-reads that
memmap, and the 9p mount is roughly an order of magnitude slower than ext4.

## Commands

```bash
cd /mnt/c/Users/u117134/Desktop/dev/model-v6-2

./session.sh start selftest    # ~1 min, no dataset or GPU needed
./session.sh start prepare     # one-off, ~2 min
./session.sh start benchmark   # step time and VRAM per batch size
./session.sh start scratch     # the training run

./session.sh attach            # watch live (detach: Ctrl-b then d)
./session.sh status            # one screen, without attaching
./run.sh report                # progress summary, safe while training
```

Everything long-running lives in a detached `tmux` session called `pcb`, so a
closed terminal or a disconnected VS Code window cannot kill a multi-day run.

## Watching it

* **Dashboard** — <http://localhost:8088> from Windows. Live curves, the full
  parameter set, per-epoch instance metrics, GPU utilisation/power/clocks.
* **tmux** — `./session.sh attach`; window `work` is the run, `dash` the
  dashboard, `gpu` a live `nvidia-smi`.
* **Files** — `TRAINING_REPORT.md` in the output directory is rewritten every
  epoch; `reports/epoch_NNNN.md` snapshots every 50.
* **TensorBoard** — `tensorboard --logdir ~/Models/Model_v6_2_scratch_rtx/tensorboard`

## Resuming

Re-run `./session.sh start scratch`. `BackupAndRestore` continues from the last
completed epoch, and `AugmentationEpochSyncCallback` restores the deterministic
augmentation cycle position, so a resumed run is not a different experiment.
