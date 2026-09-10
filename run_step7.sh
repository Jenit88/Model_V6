#!/bin/bash
# Step 7: score Model V6.2 and YOLO11l-seg under one protocol, with intervals.
#
#   ./run_step7.sh capture-v62    V6.2 predictions on test -> per-image records
#   ./run_step7.sh capture-yolo   YOLO predictions on the same frames
#   ./run_step7.sh score [draws]  one metric over both, paired bootstrap
#   ./run_step7.sh all            the three above, in order
#
# The two capture phases run in different virtual environments -- model_v6_2
# needs TensorFlow, Ultralytics needs torch, and the torch build here is cu128
# for sm_120 -- which is why this is staged through files on disk rather than
# done in one process.
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh >/dev/null

export PCB_YOLO_ENV="${PCB_YOLO_ENV:-$HOME/envs/yolo}"
export PCB_YOLO_PYTHON="$PCB_YOLO_ENV/bin/python"
export PCB_YOLO_BEST="${PCB_YOLO_BEST:-$HOME/Models/yolo11_fair/seg/weights/best.pt}"
# TTA is the shipped best-mask configuration and the source of the published
# 0.7710; the fitted-threshold variant trades mAP for precision and is not the
# headline number.
export PCB_USE_TTA="${PCB_USE_TTA:-1}"
mkdir -p logs results

case "${1:-all}" in
  capture-v62)
    exec "$PCB_PYTHON" -u step7_capture_v62.py
    ;;
  capture-yolo)
    exec "$PCB_YOLO_PYTHON" -u step7_capture_yolo.py
    ;;
  score)
    exec "$PCB_PYTHON" -u step7_score.py "${2:-1000}"
    ;;
  all)
    "$PCB_PYTHON" -u step7_capture_v62.py 2>&1 | tee logs/step7_v62.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "phase A failed" >&2; exit 1; }
    "$PCB_YOLO_PYTHON" -u step7_capture_yolo.py 2>&1 | tee logs/step7_yolo.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "phase B failed" >&2; exit 1; }
    "$PCB_PYTHON" -u step7_score.py "${2:-1000}" 2>&1 | tee logs/step7_score.log
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "phase C failed" >&2; exit 1; }
    ;;
  *)
    echo "usage: $0 {capture-v62|capture-yolo|score [draws]|all}" >&2
    exit 2
    ;;
esac
