#!/bin/bash
# YOLO11l-seg baseline, matched to Model V6.2's data and protocol.
#
#   ./run_yolo.sh build     rebuild the dataset from Split_Data (fixes the
#                           off-by-one class map in the shipped data.yaml)
#   ./run_yolo.sh train     train to a completed PCB_YOLO_EPOCHS schedule
#   ./run_yolo.sh status    progress without attaching
#
# A separate venv from the TensorFlow one: this needs torch built for sm_120
# (cu128). The pre-existing conda env has torch 2.13 against CUDA 13 sitting on
# CUDA 12 wheels and cannot import at all.
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

export PCB_YOLO_ENV="${PCB_YOLO_ENV:-$HOME/envs/yolo}"
export PCB_YOLO_PYTHON="$PCB_YOLO_ENV/bin/python"
export PCB_YOLO_DATASET="${PCB_YOLO_DATASET:-$HOME/data/yolo_fair}"
export PCB_DATASET_ROOT="${PCB_DATASET_ROOT:-/mnt/c/Users/u117134/Desktop/Data_preprocessing/1000_images/Split_Data}"
export PCB_YOLO_EPOCHS="${PCB_YOLO_EPOCHS:-300}"   # completed schedule; see train_yolo_fair.py
mkdir -p logs

case "${1:-status}" in
  build)
    exec python3 build_yolo_dataset.py
    ;;
  train)
    exec "$PCB_YOLO_PYTHON" -u train_yolo_fair.py
    ;;
  status)
    RUN="$HOME/Models/yolo11_fair/seg"
    if [ -f "$RUN/results.csv" ]; then
      echo "epochs completed: $(( $(wc -l < "$RUN/results.csv") - 1 ))"
      echo "latest:"
      head -1 "$RUN/results.csv" | tr ',' '\n' | grep -n "metrics/.*(M)" \
        | while IFS=: read -r i name; do
            printf "  %-28s %s\n" "$name" \
              "$(tail -1 "$RUN/results.csv" | cut -d, -f"$i")"
          done
    else
      echo "no results.csv yet at $RUN"
    fi
    pgrep -fa "train_yolo_fair.py" >/dev/null && echo "training: RUNNING" \
      || echo "training: not running"
    ;;
  *)
    echo "usage: $0 {build|train|status}" >&2; exit 2 ;;
esac
