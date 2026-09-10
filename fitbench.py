#!/usr/bin/env python3
"""Measure the real fit() step rate, and where the per-sample loader time goes.

train_on_batch ignores steps_per_execution and pays full Python overhead per
call, so the batch sweep in train_rtx.py understates fit(). This runs the
actual Keras training loop for a fixed number of steps under each candidate
configuration.

    PCB_FITBENCH_STEPS=40 ~/envs/pcb62/bin/python fitbench.py
"""
from __future__ import annotations

import cProfile
import importlib.util
import io
import json
import os
import pstats
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = int(os.environ.get("PCB_FITBENCH_STEPS", "40"))


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v6_2", HERE / "model_v6_2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v6_2"] = module
    spec.loader.exec_module(module)
    return module


class StepTimer:
    """Time only the steps after warm-up, from inside the fit loop."""

    def __init__(self, m, warmup: int):
        import tensorflow as tf

        self.warmup = warmup
        self.count = 0
        self.started = None
        self.elapsed = 0.0

        class _Callback(tf.keras.callbacks.Callback):
            def on_train_batch_end(inner, batch, logs=None):
                # With steps_per_execution > 1 this fires once per executed
                # group and `batch` is the index of the group's last step, so
                # the step count must come from `batch`, not from a per-call
                # counter, and the warm-up test must be >= not ==.
                self.count = int(batch) + 1
                if self.started is None and self.count >= self.warmup:
                    self.started = time.perf_counter()
                    self.start_count = self.count

        self.start_count = 0
        self.callback = _Callback()

    def finish(self):
        if self.started is None or self.count <= self.start_count:
            return float("nan")
        self.elapsed = time.perf_counter() - self.started
        return self.elapsed / (self.count - self.start_count)


def profile_loader(m, sequence, samples: int = 6) -> str:
    """Attribute the per-sample loader cost to functions."""
    profiler = cProfile.Profile()
    profiler.enable()
    for index in range(samples):
        sequence[index]
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(14)
    return "\n".join(stream.getvalue().splitlines()[4:22])


def run_case(m, label: str, *, batch_size: int, accumulation: int,
             steps_per_execution: int, jit: bool, per_class_iou: bool) -> dict:
    import tensorflow as tf

    m.BATCH_SIZE = batch_size
    m.GRADIENT_ACCUMULATION_STEPS = accumulation
    m.STEPS_PER_EXECUTION = steps_per_execution
    m.USE_XLA_JIT = jit
    m.PER_CLASS_IOU_METRICS_DURING_TRAINING = per_class_iou

    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    arrays = m.load_split_arrays("train")
    sequence = m.InstanceArraySequence(arrays, batch_size=batch_size,
                                       training=True)
    model = m.build_model_v6_2_instance()
    weights = m.compute_class_weights(arrays["semantic"])
    m.compile_model(model, weights, learning_rate=m.LEARNING_RATE)

    warmup = max(steps_per_execution * 2, 8)
    total = STEPS + warmup
    timer = StepTimer(m, warmup)
    tf.config.experimental.reset_memory_stats("GPU:0")

    started = time.perf_counter()
    try:
        model.fit(sequence, epochs=1, steps_per_epoch=total, verbose=0,
                  callbacks=[timer.callback])
    except Exception as error:
        return {"label": label, "failed": type(error).__name__,
                "detail": str(error).splitlines()[0][:180]}
    wall = time.perf_counter() - started

    per_step = timer.finish()
    peak = tf.config.experimental.get_memory_info("GPU:0")["peak"] / (1024 ** 3)
    images = len(arrays["images"])
    return {
        "label": label,
        "batch_size": batch_size,
        "effective_batch": batch_size * accumulation,
        "step_seconds": per_step,
        "ms_per_image": per_step / batch_size * 1000.0,
        "peak_gib": peak,
        "epoch_minutes": per_step * (images / batch_size) / 60.0,
        "compile_seconds": wall - timer.elapsed,
    }


def main() -> None:
    m = load_model_module()
    m.REQUIRE_GPU = True
    m.DATA_LOADER_WORKERS = int(os.environ.get("PCB_LOADER_WORKERS", "12"))
    m.configure_runtime()

    print(f"\n### loader profile (single-threaded, {m.BATCH_SIZE} per batch)")
    arrays = m.load_split_arrays("train")
    sequence = m.InstanceArraySequence(arrays, batch_size=2, training=True)
    sequence[0]  # build the copy-paste donor index outside the profile
    print(profile_loader(m, sequence))

    cases = [
        ("baseline: spe=1, no xla",   dict(batch_size=2, accumulation=1,
                                          steps_per_execution=1, jit=False,
                                          per_class_iou=True)),
        ("spe=8",                     dict(batch_size=2, accumulation=1,
                                          steps_per_execution=8, jit=False,
                                          per_class_iou=True)),
        ("spe=8, no per-class IoU",   dict(batch_size=2, accumulation=1,
                                          steps_per_execution=8, jit=False,
                                          per_class_iou=False)),
        ("spe=8, no per-class, XLA",  dict(batch_size=2, accumulation=1,
                                          steps_per_execution=8, jit=True,
                                          per_class_iou=False)),
        ("+ accumulation 2",          dict(batch_size=2, accumulation=2,
                                          steps_per_execution=8, jit=False,
                                          per_class_iou=False)),
    ]

    results = []
    for label, kwargs in cases:
        print(f"\n--- {label} ---", flush=True)
        try:
            row = run_case(m, label, **kwargs)
        except Exception as error:  # keep going; one failure is information
            row = {"label": label, "failed": type(error).__name__,
                   "detail": str(error).splitlines()[0][:180]}
        if row.get("failed"):
            print(f"  {row['failed']}: {row['detail']}")
        else:
            print(f"  {row['step_seconds']*1000:7.0f} ms/step  "
                  f"{row['ms_per_image']:6.0f} ms/image  "
                  f"peak {row['peak_gib']:.2f} GiB  "
                  f"{row['epoch_minutes']:5.1f} min/epoch  "
                  f"(compile {row['compile_seconds']:.0f}s)")
        results.append(row)

    print("\n" + "=" * 86)
    print(f"{'configuration':<30} {'ms/step':>8} {'ms/image':>9} "
          f"{'peak GiB':>9} {'min/epoch':>10}")
    for row in results:
        if row.get("failed"):
            print(f"{row['label']:<30} {row['failed']:>8}")
            continue
        print(f"{row['label']:<30} {row['step_seconds']*1000:>8.0f} "
              f"{row['ms_per_image']:>9.0f} {row['peak_gib']:>9.2f} "
              f"{row['epoch_minutes']:>10.1f}")
    print("=" * 86)
    (HERE / "fitbench_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
