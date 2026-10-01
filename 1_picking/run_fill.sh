#!/usr/bin/env bash
# v5 gap fill with option (a) (the v3 model set; V5_PLAN §9-10), resumable.
#
#   setsid nohup bash 1_picking/run_fill.sh > $OUT/driver.log 2>&1 < /dev/null &
#
# 1. waits for $WAIT_FOR (the 0.5 Hz amplitude rerun) to write DONE, so the two runs do not
#    share the machine; stops if it writes FAILED;
# 2. scaling test as part of the fill: $STAGE_MIN minutes at each worker count in $WORKER_STEPS,
#    rate counted over each stage after a 5-minute warm-up;
# 3. finishes the task list with the fastest worker count.
# Status: $OUT/STATUS; $OUT/DONE at the end. Stop: kill -- -<PGID in $OUT/driver.pid>.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
OUT="${OUT:-/wd1/mdenolle_data/picks_v5_fill}"
TASKS="${TASKS:-$OUT/tasks_offshore.csv}"
WAIT_FOR="${WAIT_FOR:-4_relocation/magnitude/rerun_hp05}"
WORKER_STEPS="${WORKER_STEPS:-30 60 90}"
STAGE_MIN="${STAGE_MIN:-20}"
ENV="${ENV:-$PWD/.pixi/envs/default}"
export LD_LIBRARY_PATH="$ENV/lib" PYTHONWARNINGS=ignore
export PYTHONPATH="${PYEXT:-/wd1/mdenolle_data/pyext}"   # pnwstore (the internal env's copy is broken)
mkdir -p "$OUT"
echo "$(ps -o pgid= $$ | tr -d ' ')" > "$OUT/driver.pid"
status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/STATUS"; }
pick() {  # $1 workers; runs in its own process group so a stage can be stopped cleanly
  setsid "$ENV/bin/python" 1_picking/dryrun_models.py --tasks "$TASKS" --out "$OUT" \
    --configs a --workers "$1" >> "$OUT/picker.log" 2>&1 &
  PID=$!
}
n_done_since() { awk -F, -v t="$1" 'NR>1 && $NF>=t' "$OUT/log.csv" 2>/dev/null | wc -l; }

status "waiting for $WAIT_FOR/DONE"
until [ -f "$WAIT_FOR/DONE" ]; do
  [ -f "$WAIT_FOR/FAILED" ] && { status "$WAIT_FOR FAILED: not starting"; exit 1; }
  sleep 120
done
status "start: $(($(wc -l < "$TASKS") - 1)) station-days in $TASKS"

best_w=0; best_rate=0
for w in $WORKER_STEPS; do
  [ -f "$OUT/DONE" ] && break
  pick "$w"
  sleep 300; t0=$(date '+%Y-%m-%dT%H:%M:%S')
  sleep $(( (STAGE_MIN - 5) * 60 ))
  n=$(n_done_since "$t0")
  rate=$(awk -v n="$n" -v m="$STAGE_MIN" 'BEGIN{printf "%.1f", n/(m-5)}')   # station-days / min
  kill -- -"$PID" 2>/dev/null; sleep 20; kill -9 -- -"$PID" 2>/dev/null
  status "scaling: $w workers -> $rate station-days/min"
  if awk -v a="$rate" -v b="$best_rate" 'BEGIN{exit !(a>b)}'; then best_w=$w; best_rate=$rate; fi
done
[ -f "$OUT/DONE" ] && { status "DONE during the scaling test"; exit 0; }

left=$(( $(wc -l < "$TASKS") - $(wc -l < "$OUT/log.csv") ))
status "running with $best_w workers ($best_rate/min): $left left, ~$(awk -v l="$left" -v r="$best_rate" 'BEGIN{printf "%.1f", l/r/1440}') days"
pick "$best_w"; wait "$PID"; rc=$?
if [ -f "$OUT/DONE" ]; then
  status "DONE: $(grep -c ',ok,' "$OUT/log.csv") ok of $(($(wc -l < "$OUT/log.csv") - 1))"
else
  status "picker exited ($rc) without DONE: rerun this script to resume"; touch "$OUT/FAILED"
fi
