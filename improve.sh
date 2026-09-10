#!/bin/bash
# Run every remaining no-retraining improvement, in order, unattended.
#
# Each stage waits for the GPU to be free of the previous one, so this can be
# launched once and left. Every stage writes its own result file under
# results/ and its own log under logs/, so a stage that fails does not lose the
# stages before it.
#
#   ./improve.sh            run the whole chain
#   ./improve.sh status     where it is
set -uo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"
source ./env.sh >/dev/null
mkdir -p logs results

STATE="$HERE/logs/improve.state"

note() { echo "$(date '+%F %T')  $*" | tee -a "$HERE/logs/improve.log"; }

wait_for_gpu_free() {
  # Wait until no evaluation/fitting process of ours is running.
  while pgrep -f "reevaluate.py|fit_thresholds.py" >/dev/null 2>&1; do
    sleep 60
  done
}

if [ "${1:-run}" = "status" ]; then
  echo "stage: $(cat "$STATE" 2>/dev/null || echo 'not started')"
  tail -12 "$HERE/logs/improve.log" 2>/dev/null
  exit 0
fi

note "=== improvement chain starting ==="

# --- 1. wait out the TTA evaluation already in flight -----------------------
echo "waiting-for-tta" > "$STATE"
note "waiting for the in-flight TTA evaluation to finish"
wait_for_gpu_free
note "GPU free"

# --- 2. fit the decoder thresholds ------------------------------------------
echo "fitting-thresholds" > "$STATE"
note "fitting decoder thresholds (400 images, 150 draws, split-validated)"
PCB_FIT_IMAGES=400 PCB_FIT_DRAWS=150 \
  "$PCB_PYTHON" -u fit_thresholds.py > logs/fit_thresholds.log 2>&1
note "threshold fit exit $?"
grep -E "held-out gain|shipped |fitted " logs/fit_thresholds.log | tail -4 \
  | tee -a "$HERE/logs/improve.log"

# --- 3. final combined evaluation -------------------------------------------
# Applied only if the fit survived its held-out half; fit_thresholds.py prints
# a negative gain and this reads it back rather than trusting the search.
echo "final-evaluation" > "$STATE"
GAIN=$("$PCB_PYTHON" - <<'PY'
import json, pathlib
p = pathlib.Path("results/threshold_fit.json")
print(json.loads(p.read_text())["held_out_gain_f1"] if p.exists() else 0.0)
PY
)
note "held-out threshold gain: $GAIN"

APPLY=$("$PCB_PYTHON" -c "print('1' if float('$GAIN') > 0 else '0')")
if [ "$APPLY" = "1" ]; then
  note "fitted thresholds survived the held-out half; applying them"
  export PCB_APPLY_FITTED_THRESHOLDS=1
else
  note "fitted thresholds did NOT survive; keeping the shipped values"
fi

note "final evaluation with TTA on both complete splits"
PCB_USE_TTA=1 PCB_EVAL_TAG=final \
  "$PCB_PYTHON" -u reevaluate.py > logs/final_eval.log 2>&1
note "final evaluation exit $?"
tail -8 logs/final_eval.log | tee -a "$HERE/logs/improve.log"

echo "done" > "$STATE"
note "=== improvement chain complete ==="
