#!/bin/bash
# Compare Model V6.2 and YOLO11l-seg on a folder of unlabelled real boards.
#
#   ./run_ro2.sh capture-v62         V6.2, benchmark decode thresholds
#   ./run_ro2.sh capture-v62-deploy  V6.2, the shipped deployment profile
#   ./run_ro2.sh capture-yolo        YOLO on the frames phase A wrote
#   ./run_ro2.sh compare             counts, agreement, side-by-side pictures
#   ./run_ro2.sh all                 the four above, in order
#
# Staged through files on disk for the same reason run_step7.sh is: model_v6_2
# needs TensorFlow, Ultralytics needs torch, and the torch build here is cu128
# for sm_120, so the two cannot share a process.
#
# Phase A owns the letterboxing for BOTH models and writes frames.npy, so YOLO
# is fed the identical pixels rather than re-letterboxing the source PNGs.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null

export PCB_RO2_SOURCE="${PCB_RO2_SOURCE:-/mnt/c/Users/u117134/Desktop/ro/ro_2}"
export PCB_YOLO_ENV="${PCB_YOLO_ENV:-$HOME/envs/yolo}"
export PCB_YOLO_PYTHON="$PCB_YOLO_ENV/bin/python"
export PCB_YOLO_BEST="${PCB_YOLO_BEST:-$HOME/Models/yolo11_fair/seg/weights/best.pt}"
mkdir -p logs results/ro2

case "${1:-all}" in
  capture-v62)
    exec "$PCB_PYTHON" -u ro2_capture_v62.py
    ;;
  capture-v62-deploy)
    PCB_RO2_PROFILE=deployment exec "$PCB_PYTHON" -u ro2_capture_v62.py
    ;;
  capture-yolo)
    exec "$PCB_YOLO_PYTHON" -u ro2_capture_yolo.py
    ;;
  compare)
    exec "$PCB_PYTHON" -u ro2_compare.py
    ;;
  all)
    "$PCB_PYTHON" -u ro2_capture_v62.py 2>&1 | tee logs/ro2_v62.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "phase A failed" >&2; exit 1; }
    PCB_RO2_PROFILE=deployment "$PCB_PYTHON" -u ro2_capture_v62.py 2>&1 \
      | tee logs/ro2_v62_deploy.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "phase A (deploy) failed" >&2; exit 1; }
    "$PCB_YOLO_PYTHON" -u ro2_capture_yolo.py 2>&1 | tee logs/ro2_yolo.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "phase B failed" >&2; exit 1; }
    "$PCB_PYTHON" -u ro2_compare.py 2>&1 | tee logs/ro2_compare.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "phase C failed" >&2; exit 1; }
    ;;
  rotation)
    # Controlled test for the orientation hypothesis: same board, same scale,
    # same field of view, only the angle changes. Captures land in their own
    # directory so the 12-board run above is untouched.
    export PCB_RO2_OUT="$PWD/results/ro2_rotation"
    mkdir -p "$PCB_RO2_OUT"
    "$PCB_PYTHON" -u ro2_rotation_test.py build "$PCB_RO2_SOURCE" \
      2>&1 | tee logs/ro2_rotation_build.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "rotation build failed" >&2; exit 1; }
    PCB_RO2_SOURCE="$PCB_RO2_OUT/images" "$PCB_PYTHON" -u ro2_capture_v62.py \
      2>&1 | tee logs/ro2_rotation_v62.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "rotation phase A failed" >&2; exit 1; }
    "$PCB_YOLO_PYTHON" -u ro2_capture_yolo.py 2>&1 | tee logs/ro2_rotation_yolo.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "rotation phase B failed" >&2; exit 1; }
    "$PCB_PYTHON" -u ro2_rotation_test.py report 2>&1 \
      | tee logs/ro2_rotation_report.log
    ;;
  *)
    echo "usage: $0 {capture-v62|capture-v62-deploy|capture-yolo|compare|rotation|all}" >&2
    exit 2
    ;;
esac
