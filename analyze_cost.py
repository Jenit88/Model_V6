#!/usr/bin/env python3
"""Where the forward pass actually spends its multiply-accumulates.

Analytic MACs per layer from the built graph, grouped by the spatial
resolution the layer runs at and by the architectural stage it belongs to.
Run through ./run.sh so env.sh has been sourced:

    ~/envs/pcb62/bin/python analyze_cost.py
"""
from __future__ import annotations

import importlib.util
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "model_v6_2", HERE / "model_v6_2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_v6_2"] = module
    spec.loader.exec_module(module)
    return module


def layer_macs(layer) -> int:
    """MACs for the layer types this model is built from."""
    import tensorflow as tf

    shape = getattr(layer, "output", None)
    if shape is None:
        return 0
    shape = getattr(shape, "shape", None)
    if shape is None or len(shape) != 4:
        # Dense layers in the channel-attention blocks.
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            return int(weights[0].size) if weights else 0
        return 0
    _, height, width, out_channels = shape
    if height is None or width is None:
        return 0
    positions = int(height) * int(width)

    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        k = layer.kernel_size[0] * layer.kernel_size[1]
        return positions * int(out_channels) * k
    if isinstance(layer, tf.keras.layers.Conv2D):
        k = layer.kernel_size[0] * layer.kernel_size[1]
        in_channels = layer.input.shape[-1]
        if in_channels is None:
            return 0
        groups = getattr(layer, "groups", 1) or 1
        return positions * int(out_channels) * int(in_channels) * k // groups
    return 0


def stage_of(name: str) -> str:
    for prefix, stage in (
        ("rgb_detail", "stem 512"), ("edge_detail", "stem 512"),
        ("detail_fusion", "stem 512"), ("fixed_sobel", "stem 512"),
        ("detail_half", "stem 512"),
        ("backbone_p1", "encoder"), ("backbone_p2", "encoder"),
        ("backbone_p3", "encoder"), ("backbone_p4", "encoder"),
        ("backbone_p5", "encoder"), ("sppf", "encoder"),
        ("deep_attention", "encoder"),
        ("pyramid_", "bifpn"), ("bifpn", "bifpn"),
        ("semantic_context", "context"), ("instance_context", "context"),
        ("semantic_", "semantic decoder"), ("boundary_", "semantic decoder"),
        ("instance_", "instance decoder"),
        ("center_", "heads"), ("offset_", "heads"),
    ):
        if name.startswith(prefix):
            return stage
    return "other"


def main() -> None:
    m = load_model_module()
    model = m.build_model_v6_2_instance()

    by_resolution: dict[int, int] = defaultdict(int)
    by_stage: dict[str, int] = defaultdict(int)
    rows = []
    total = 0
    for layer in model.layers:
        macs = layer_macs(layer)
        if not macs:
            continue
        total += macs
        shape = layer.output.shape
        resolution = int(shape[1]) if len(shape) == 4 and shape[1] else 0
        by_resolution[resolution] += macs
        by_stage[stage_of(layer.name)] += macs
        rows.append((macs, layer.name, resolution))

    print(f"total {total / 1e9:.2f} GMAC per image "
          f"({model.count_params():,} parameters)\n")

    print("by input resolution the layer runs at")
    for resolution, macs in sorted(by_resolution.items(), reverse=True):
        bar = "#" * int(round(40 * macs / total))
        print(f"  {resolution:>4} px  {macs / 1e9:6.2f} GMAC  "
              f"{macs / total * 100:5.1f}%  {bar}")

    print("\nby stage")
    for stage, macs in sorted(by_stage.items(), key=lambda kv: -kv[1]):
        bar = "#" * int(round(40 * macs / total))
        print(f"  {stage:<18} {macs / 1e9:6.2f} GMAC  "
              f"{macs / total * 100:5.1f}%  {bar}")

    print("\n15 most expensive layers")
    for macs, name, resolution in sorted(rows, reverse=True)[:15]:
        print(f"  {macs / 1e9:6.3f} GMAC  {macs / total * 100:5.1f}%  "
              f"{resolution:>4}px  {name}")


if __name__ == "__main__":
    main()
