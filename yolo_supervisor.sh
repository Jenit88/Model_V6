#!/bin/bash
# Keep the YOLO baseline alive until its schedule completes.
#
# Two different failures have each cost this project a run:
#
#   04-09  the training process died while the machine stayed up
#          (CUDA context loss, TensorFlow CHECK-failed with no Python
#          exception). watchdog.sh was written for this and handles it.
#   05-09  the host crashed. That takes the WSL VM with it, so a poller
#          living inside WSL -- watchdog.sh included -- dies alongside the
#          job it was supposed to restart. 27 hours were lost before anyone
#          noticed.
#
# This script handles the first by relaunching in a loop. The second needs a
# trigger outside WSL: register_yolo_watchdog.ps1 registers a Windows logon
# task that calls this script, which is why every path below is idempotent --
# a logon trigger must never stack a second job onto the same 8 GB GPU.
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

RUN="${PCB_YOLO_RUN:-$HOME/Models/yolo11_fair/seg}"
LOG="$HERE/logs/yolo_fair.log"
STATE="$HOME/Models/yolo11_fair/supervisor"
mkdir -p "$HERE/logs" "$STATE"

note() { echo "$(date '+%F %T')  $*" | tee -a "$STATE/supervisor.log"; }
# Already training: do nothing. This is what makes a logon trigger safe.
#
# Match the interpreter, not the filename. `pgrep -f train_yolo_fair.py` also
# matches any shell whose command line merely mentions the script -- the
# launcher itself, a status check, a logon wrapper -- and on 07-09 that made
# the supervisor stand down against the very command that was starting it.
if pgrep -f "bin/python -u train_yolo_fair" >/dev/null 2>&1; then
  note "training already running (pid $(pgrep -f 'bin/python -u train_yolo_fair' | tr '\n' ' ')); nothing to do"
  exit 0
fi

if [ -f "$STATE/COMPLETE" ]; then
  note "run already marked complete; nothing to do"
  exit 0
fi

attempt=0
while true; do
  attempt=$((attempt + 1))

  if [ -f "$RUN/weights/last.pt" ]; then
    # An epoch-based schedule resumes cleanly: Ultralytics restores the run's
    # args from the checkpoint and continues to the same epoch total, so a
    # crash costs the epochs since the last save and nothing else. (This was
    # not true of the earlier `time` budget, which restarted the wall clock on
    # every resume and quietly inflated the baseline's compute.)
    note "attempt $attempt: RESUMING from $RUN/weights/last.pt -- continues to the checkpoint's own epoch total"
  else
    note "attempt $attempt: fresh start, ${PCB_YOLO_EPOCHS:-300} epochs to completion"
  fi

  started=$(date +%s)
  ./run_yolo.sh train 2>&1 | tee -a "$LOG"
  code=${PIPESTATUS[0]}
  elapsed=$(( $(date +%s) - started ))
  echo "$elapsed" >> "$STATE/elapsed_seconds"
  note "attempt $attempt exited $code after $((elapsed / 60)) min"

  if [ "$code" -eq 0 ]; then
    touch "$STATE/COMPLETE"
    total=$(awk '{s+=$1} END {printf "%.1f", s/3600}' "$STATE/elapsed_seconds")
    note "COMPLETE -- ${total} h of training across $attempt launch(es)"
    exit 0
  fi

  # A launch that dies in under two minutes is failing to start, not crashing
  # mid-run: a bad path, a missing venv, an occupied GPU. Retrying that on a
  # 60 s loop just fills the log. Stop and leave the reason in supervisor.log.
  if [ "$elapsed" -lt 120 ]; then
    note "died in under 2 min -- treating as a startup failure, not a crash. Stopping so the cause is visible. Last 20 log lines:"
    tail -20 "$LOG" | tr '\r' '\n' | tail -20 | tee -a "$STATE/supervisor.log"
    exit 1
  fi

  note "relaunching in 60 s"
  sleep 60
done
