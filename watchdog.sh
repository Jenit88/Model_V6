#!/bin/bash
# Restart the training run if it dies before finishing its schedule.
#
# On 2026-09-04 the run was killed mid-epoch by a CUDA kernel launch failure
# ("Check failed: GpuLaunchKernel(ColumnReduceKernel...) is OK (INTERNAL:
# unknown error)") -- a GPU driver reset, which WSL2's paravirtualised GPU is
# prone to on a laptop. It cost 105 minutes of idle GPU before anyone noticed.
# BackupAndRestore makes recovery safe and near-free, so nobody should have to
# notice.
#
#   ./watchdog.sh start    supervise, in its own tmux window
#   ./watchdog.sh stop     stand down (also stops it restarting anything)
#   ./watchdog.sh status
#
# It deliberately does NOT restart when:
#   * the schedule is complete (EPOCHS rows in training_log.csv)
#   * logs/.watchdog-stop exists, i.e. the run was stopped on purpose
#   * it has already restarted MAX_RESTARTS times -- a crash loop is a bug to
#     read, not to paper over.
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
SESSION="${PCB_TMUX_SESSION:-pcb}"
# tmux resolves a target session by *prefix*, so `-t pcb` also matches the
# supervisor's own `pcbsup` session (created by the Windows logon task). On
# 10-09 that made the resume a no-op: has-session matched `pcbsup` and returned
# early, so the `work` window was never created, `respawn-pane -t pcb:work`
# failed with "can't find window: work", and only the aux windows came up --
# against the default output dir. `=` forces an exact name match.
SESSION_T="=$SESSION"
STOP_FILE="$HERE/logs/.watchdog-stop"
WATCH_LOG="$HERE/logs/watchdog.log"
POLL_SECONDS=60
MAX_RESTARTS=12

epochs_target() { grep -oE 'm\.EPOCHS = [0-9]+' train_rtx.py | grep -oE '[0-9]+' | tail -1; }

epochs_done() {
  local f
  f="$(bash -c 'source ./env.sh >/dev/null 2>&1; echo "$PCB_MODEL_OUTPUT_DIR"')/training_log.csv"
  [ -s "$f" ] || { echo 0; return; }
  # Subtract the header; tolerate a header repeated by an appending resume.
  awk -F, 'NR>1 && $1 != "epoch" && $1 != "" {n++} END {print n+0}' "$f"
}

alive() { pgrep -f "train_rtx.py scratch" >/dev/null 2>&1; }

note() { echo "$(date '+%F %T')  $*" | tee -a "$WATCH_LOG"; }

supervise() {
  mkdir -p "$HERE/logs"
  local target restarts=0 done_count
  target="$(epochs_target)"
  # Name the directory being watched. The 08-09 failure was invisible because
  # this line did not: a watchdog pointed at a finished run logs "standing
  # down" and looks like correct behaviour.
  note "watchdog up; target ${target} epochs; polling every ${POLL_SECONDS}s"
  note "  watching $(bash -c 'source ./env.sh >/dev/null 2>&1; echo "$PCB_MODEL_OUTPUT_DIR"')" \
       "($(epochs_done) epochs done)"
  while true; do
    sleep "$POLL_SECONDS"
    [ -f "$STOP_FILE" ] && { note "stop file present; standing down"; return 0; }

    done_count="$(epochs_done)"
    if [ "$done_count" -ge "$target" ] 2>/dev/null; then
      note "schedule complete (${done_count}/${target}); standing down"
      return 0
    fi

    if alive; then
      continue
    fi

    if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
      note "died again after ${restarts} restarts -- not restarting. Read logs/scratch.log."
      return 1
    fi

    restarts=$((restarts + 1))
    note "training is not running at epoch ${done_count}/${target}; restart ${restarts}/${MAX_RESTARTS}"
    # Roll the log so the crash that caused this stays readable.
    [ -f "$HERE/logs/scratch.log" ] && \
      mv "$HERE/logs/scratch.log" "$HERE/logs/scratch.crash-$(date +%m%d-%H%M).log"
    ./session.sh start scratch >>"$WATCH_LOG" 2>&1
    # Give it time to load arrays, build and compile before judging it again.
    sleep 240
  done
}

case "${1:-status}" in
  start)
    rm -f "$STOP_FILE"
    tmux has-session -t "$SESSION_T" 2>/dev/null || { echo "no tmux session '$SESSION'" >&2; exit 1; }
    tmux list-windows -t "$SESSION_T" -F '#{window_name}' | grep -qx dog && \
      tmux kill-window -t "$SESSION_T:dog"
    # Forward the path overrides into the window, for the same reason
    # session.sh does: a tmux window inherits the *server's* environment,
    # captured whenever that server first started, not the environment of the
    # shell running this script. On 08-09 that made the watchdog read the
    # finished run's training_log.csv, conclude 120/120 one minute after
    # starting, and stand down -- leaving a fresh 2.8-day run unsupervised
    # while reporting itself healthy. A watchdog that silently supervises the
    # wrong directory is worse than none, because it is believed.
    ENV_PREFIX=""
    for name in PCB_ARRAY_DIR PCB_MODEL_OUTPUT_DIR PCB_DATASET_ROOT; do
      value="$(eval "printf '%s' \"\${$name-}\"")"
      [ -n "$value" ] && ENV_PREFIX="$ENV_PREFIX $name='$value'"
    done
    tmux new-window -d -t "$SESSION_T" -n dog -c "$HERE" \
      "env$ENV_PREFIX ./watchdog.sh _run; exec bash"
    echo "watchdog started in tmux window 'dog'  (log: logs/watchdog.log)"
    [ -n "$ENV_PREFIX" ] && echo "  overrides   :$ENV_PREFIX"
    ;;
  _run)    supervise ;;
  stop)    touch "$STOP_FILE"; echo "watchdog will stand down within ${POLL_SECONDS}s" ;;
  status)
    if [ -f "$STOP_FILE" ]; then echo "watchdog: stopped (stop file present)"
    elif tmux list-windows -t "$SESSION_T" -F '#{window_name}' 2>/dev/null | grep -qx dog
    then echo "watchdog: running"; tail -5 "$WATCH_LOG" 2>/dev/null
    else echo "watchdog: not running"; fi
    ;;
  *) echo "usage: $0 {start|stop|status}" >&2; exit 2 ;;
esac
