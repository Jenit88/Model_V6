#!/bin/bash
# Keep the model_v6_2 training run alive across a host crash.
#
# Two failures have each cost this project a run, and they need different
# machinery:
#
#   04-09  the training process died while the machine stayed up (CUDA context
#          loss). watchdog.sh polls from inside WSL and handles this.
#   05-09  the host crashed, which takes the WSL VM with it -- and every
#          poller living inside it, watchdog.sh included. 27 hours were lost.
#   09-09  the host crashed again, at epoch 32/120 of the Tier-1 run. YOLO had
#          a Windows logon task by then and stood down correctly; this run had
#          none, so nothing restarted it.
#
# This script is the trigger that survives a reboot: register_pcb_watchdog.ps1
# calls it from a Windows logon task. Every path below is idempotent, because a
# logon trigger must never stack a second job onto the same 8 GB GPU.
#
#   ./pcb_supervisor.sh          resume/ensure the run and its watchdog
#   ./pcb_supervisor.sh status   report without changing anything
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

# The run to supervise. Kept in one file so a new run is one edit, not three.
# These must be exported before env.sh is sourced anywhere downstream: its
# defaults still point at the run that finished on 05-09.
[ -f "$HERE/active_run.env" ] && set -a && . "$HERE/active_run.env" && set +a

SESSION="${PCB_TMUX_SESSION:-pcb}"
# tmux resolves a target session by *prefix*, so `-t pcb` also matches the
# supervisor's own `pcbsup` session (created by the Windows logon task). On
# 10-09 that made the resume a no-op: has-session matched `pcbsup` and returned
# early, so the `work` window was never created, `respawn-pane -t pcb:work`
# failed with "can't find window: work", and only the aux windows came up --
# against the default output dir. `=` forces an exact name match.
SESSION_T="=$SESSION"
OUT="${PCB_MODEL_OUTPUT_DIR:?active_run.env must set PCB_MODEL_OUTPUT_DIR}"
STOP_FILE="$HERE/logs/.watchdog-stop"
SUP_LOG="$HERE/logs/pcb_supervisor.log"
mkdir -p "$HERE/logs"

note() { echo "$(date '+%F %T')  $*" | tee -a "$SUP_LOG"; }

epochs_target() { grep -oE 'm\.EPOCHS = [0-9]+' train_rtx.py | grep -oE '[0-9]+' | tail -1; }

epochs_done() {
  local f="$OUT/training_log.csv"
  [ -s "$f" ] || { echo 0; return; }
  # Subtract the header; tolerate a header repeated by an appending resume.
  awk -F, 'NR>1 && $1 != "epoch" && $1 != "" {n++} END {print n+0}' "$f"
}

# Match the interpreter, not the filename. `pgrep -f train_rtx.py` also matches
# the tmux pane command, a status check, or this script's own logon wrapper --
# on 07-09 that exact mistake made the YOLO supervisor stand down against the
# command that was starting it.
alive() { pgrep -f "bin/python -u train_rtx.py scratch" >/dev/null 2>&1; }

watchdog_up() {
  tmux list-windows -t "$SESSION_T" -F '#{window_name}' 2>/dev/null | grep -qx dog
}

target="$(epochs_target)"
done_count="$(epochs_done)"

if [ "${1:-run}" = "status" ]; then
  echo "run       : $OUT"
  echo "epochs    : ${done_count}/${target}"
  echo "training  : $(alive && echo running || echo 'NOT running')"
  echo "watchdog  : $(watchdog_up && echo running || echo 'NOT running')"
  [ -f "$STOP_FILE" ] && echo "stop file : present (supervisor would stand down)"
  exit 0
fi

# Stopped on purpose. Honour it -- a logon must not undo a deliberate stop.
if [ -f "$STOP_FILE" ]; then
  note "stop file present; nothing to do"
  exit 0
fi

if [ "$done_count" -ge "$target" ] 2>/dev/null; then
  note "schedule complete (${done_count}/${target}) for $OUT; nothing to do"
  exit 0
fi

# Already training: only make sure the in-VM watchdog is beside it. This is
# what makes firing on every logon safe.
if alive; then
  note "training already running at epoch ${done_count}/${target} (pid $(pgrep -f 'bin/python -u train_rtx.py scratch' | tr '\n' ' '))"
  if watchdog_up; then
    note "  watchdog already up; nothing to do"
  else
    note "  watchdog missing; starting it"
    ./watchdog.sh start >>"$SUP_LOG" 2>&1
  fi
  exit 0
fi

note "training is NOT running at epoch ${done_count}/${target}; resuming $OUT"
if [ -s "$OUT/fit_backup/training_metadata.json" ]; then
  note "  BackupAndRestore present: $(cat "$OUT/fit_backup/training_metadata.json")"
else
  note "  WARNING: no fit_backup -- this would start from scratch, not resume."
  note "  Refusing. Remove $OUT or investigate before starting a fresh run."
  exit 1
fi

# Keep the crash that caused this readable instead of appending onto it.
[ -f "$HERE/logs/scratch.log" ] && \
  mv "$HERE/logs/scratch.log" "$HERE/logs/scratch.crash-$(date +%m%d-%H%M).log"

./session.sh start scratch >>"$SUP_LOG" 2>&1
note "  launched; giving it 240 s to load arrays, build and compile"
sleep 240

if alive; then
  note "  training is up"
else
  note "  FAILED to come up -- read logs/scratch.log"
  tail -20 "$HERE/logs/scratch.log" 2>/dev/null | tr '\r' '\n' | tail -20 | tee -a "$SUP_LOG"
  exit 1
fi

watchdog_up || ./watchdog.sh start >>"$SUP_LOG" 2>&1
note "  watchdog up; supervisor done"
