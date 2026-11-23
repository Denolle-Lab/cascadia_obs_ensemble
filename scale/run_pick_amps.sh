#!/usr/bin/env bash
# v6 gap fill: picks + Route A amplitudes per station-day (workflow/01_picking/pick_and_amps.py),
# resumable, with a hang watchdog.
#
#   OUT=/wd1/mdenolle_data/picks_v6_east setsid nohup bash scale/run_pick_amps.sh \
#       > $OUT/driver.log 2>&1 < /dev/null &
#
# Needs $OUT/tasks.csv and $OUT/station_inventory.xml (pick_and_amps.py tasks / inventory).
# The v5 fill hung for two days after out-of-memory kills (multiprocessing.Pool waits forever
# for a killed task). Here, if log.csv has not grown for $STALL_MIN minutes, the picker is
# killed and restarted; it resumes from log.csv. At most $MAX_RESTARTS restarts.
# Status: $OUT/STATUS; $OUT/DONE at the end, $OUT/FAILED otherwise.
# Stop: kill -- -<PGID in $OUT/driver.pid>.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/.."
OUT="${OUT:?set OUT}"
WORKERS="${WORKERS:-40}"            # <= 40: 60 sat at the memory limit in the v5 fill
STALL_MIN="${STALL_MIN:-30}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
ENV="${ENV:-$PWD/.pixi/envs/default}"
export LD_LIBRARY_PATH="$ENV/lib" PYTHONWARNINGS=ignore
export PYTHONPATH="${PYEXT:-/wd1/mdenolle_data/pyext}"   # pnwstore (the internal env's copy is broken)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "$(ps -o pgid= $$ | tr -d ' ')" > "$OUT/driver.pid"
status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/STATUS"; }
nlog() { [ -f "$OUT/log.csv" ] && wc -l < "$OUT/log.csv" || echo 0; }
ntask=$(( $(wc -l < "$OUT/tasks.csv") - 1 ))

status "start: $ntask station-days, $WORKERS workers, commit $(git rev-parse --short HEAD)"
restarts=0
while [ ! -f "$OUT/DONE" ]; do
  setsid "$ENV/bin/python" workflow/01_picking/pick_and_amps.py run --out "$OUT" \
    --workers "$WORKERS" >> "$OUT/picker.log" 2>&1 &
  PID=$!
  last=$(nlog); t_last=$(date +%s); t_status=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    n=$(nlog)
    if [ "$n" != "$last" ]; then last=$n; t_last=$(date +%s); fi
    if [ $(( $(date +%s) - t_status )) -ge 600 ]; then
      done_n=$(( n - 1 ))
      ok=$(grep -c ',ok,' "$OUT/log.csv" 2>/dev/null || echo 0)
      status "$done_n/$ntask station-days ($ok ok), $(free -g | awk '/Mem/{print $7}') GB free"
      t_status=$(date +%s)
    fi
    if [ $(( $(date +%s) - t_last )) -ge $(( STALL_MIN * 60 )) ]; then
      status "no progress for $STALL_MIN min: killing the picker (restart $((restarts + 1)))"
      kill -- -"$PID" 2>/dev/null; sleep 30; kill -9 -- -"$PID" 2>/dev/null
      break
    fi
  done
  wait "$PID" 2>/dev/null
  [ -f "$OUT/DONE" ] && break
  restarts=$((restarts + 1))
  if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
    status "too many restarts: stopping"; touch "$OUT/FAILED"; exit 1
  fi
  status "picker stopped without DONE: resuming"
  sleep 30
done
status "DONE: $(grep -c ',ok,' "$OUT/log.csv") ok of $(( $(nlog) - 1 )); \
$(awk -F, 'NR>1{s[$4]++} END{for(k in s) printf "%s=%d ", k, s[k]}' "$OUT/log.csv")"
