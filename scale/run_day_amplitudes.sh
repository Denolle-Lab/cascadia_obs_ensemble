#!/usr/bin/env bash
# Route A amplitudes for every pick of ver5, one station-day at a time
# (workflow/05_magnitude/route_a_day_amplitudes.py): plan -> shards -> consolidate -> validate.
#
# Detach it from the terminal so it survives logout:
#   setsid nohup bash scale/run_day_amplitudes.sh > /wd1/mdenolle_data/amps_v5/driver.log 2>&1 < /dev/null &
#
# Re-running the same command resumes: a shard skips the station-days in its log
# (logs/shard_NNN.csv), so an OOM-killed or crashed shard loses at most one chunk
# (--chunk-tasks station-days). Each shard is a single process (no Pool to hang).
# Status: $OUT/STATUS (a progress line every 10 min), then $OUT/DONE or $OUT/FAILED.
# Stop everything: kill -- -<PGID> (the PGID is in $OUT/driver.pid).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HERE="$ROOT/workflow/05_magnitude"
cd "$HERE"
ENV="${ENV:-$ROOT/.pixi/envs/default}"     # pyarrow lives here
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ENV/lib"
# pnwstore is installed in the amplitude env only; /wd1/mdenolle_data/pyext links it
export PYTHONPATH="${PYEXT:-/wd1/mdenolle_data/pyext}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

PICKS="${PICKS:-/wd1/mdenolle_data/picks_v5/all_picks_all_regions_2010_2016_ver5.csv.gz}"
OUT="${OUT:-/wd1/mdenolle_data/amps_v5}"
INV="${INV:-$HERE/station_inventory_v2_slim.xml}"
NSHARD="${NSHARD:-48}"                     # one process each; ~1-3 GB RSS per shard
TRIES="${TRIES:-4}"
RUN_ARGS="${RUN_ARGS:-}"                   # e.g. --prefer-complete for a replan run
ARRIVALS="${ARRIVALS:-$ROOT/data/zenodo/cascadia-obs-ensemble-catalog-v3/catalog/arrivals_v3.csv}"
RAW_V2="${RAW_V2:-$HERE/rerun_v2/raw_wa_amplitudes_v2.csv}"
mkdir -p "$OUT"
echo "$(ps -o pgid= $$ | tr -d ' ')" > "$OUT/driver.pid"
rm -f "$OUT/DONE" "$OUT/FAILED"

status() { echo "$(date '+%F %T') $*" | tee -a "$OUT/STATUS"; }
fail()   { status "FAILED: $*"; touch "$OUT/FAILED"; exit 1; }

# ---- plan (once) ---------------------------------------------------------------
if [ ! -s "$OUT/plan.json" ]; then
    status "plan: $PICKS -> $NSHARD shards"
    "$PY" route_a_day_amplitudes.py plan --picks "$PICKS" --out "$OUT" --nshard "$NSHARD" \
        > "$OUT/plan.log" 2>&1 || fail "plan (see plan.log)"
fi
NSHARD=$("$PY" -c "import json;print(json.load(open('$OUT/plan.json'))['nshard'])")
NTASK=$("$PY" -c "import json;print(json.load(open('$OUT/plan.json'))['n_tasks'])")
status "start: $NTASK station-days, $NSHARD shards, inventory $(basename "$INV")"

done_tasks() { cat "$OUT"/logs/shard_*.csv 2>/dev/null | grep -vc '^task,' ; }
shard_todo() {   # station-days in the plan for shard $1 minus the ones logged
    local s; s=$(printf %03d "$1")
    local want; want=$(awk -F, -v s="$1" 'NR>1 && $NF==s' "$OUT/tasks.csv" | wc -l)
    local have=0
    [ -s "$OUT/logs/shard_$s.csv" ] && have=$(( $(wc -l < "$OUT/logs/shard_$s.csv") - 1 ))
    echo $(( want - have ))
}

# progress line every 10 min while the shards run
( while sleep 600; do
      [ -e "$OUT/DONE" ] || [ -e "$OUT/FAILED" ] && exit 0
      n=$(done_tasks); status "progress: $n / $NTASK station-days"
  done ) &
PROG=$!
trap 'kill $PROG 2>/dev/null' EXIT

# ---- shards, with retries -------------------------------------------------------
mkdir -p "$OUT/logs"
for try in $(seq 1 "$TRIES"); do
    pids=()
    for i in $(seq 0 $(( NSHARD - 1 ))); do
        left=$(shard_todo "$i")
        [ "$left" -le 0 ] && continue
        s=$(printf %03d "$i")
        "$PY" route_a_day_amplitudes.py run --out "$OUT" --shard "$i" --inventory "$INV" $RUN_ARGS \
            >> "$OUT/logs/shard_$s.out" 2>&1 &
        pids+=($!)
    done
    [ ${#pids[@]} -eq 0 ] && break
    status "try $try: ${#pids[@]} shards running"
    for p in "${pids[@]}"; do wait "$p"; done
    status "try $try ended: $(done_tasks) / $NTASK station-days"
done
left=$(( NTASK - $(done_tasks) ))
[ "$left" -gt 0 ] && fail "$left station-days still missing after $TRIES tries"

# ---- consolidate + validate ------------------------------------------------------
status "consolidate"
"$PY" route_a_day_amplitudes.py consolidate --out "$OUT" > "$OUT/consolidate.log" 2>&1 \
    || fail "consolidate (see consolidate.log)"
status "validate against the per-pick v2 amplitudes (arrivals_v3.csv)"
"$PY" route_a_day_amplitudes.py validate --amps "$OUT/amps" --arrivals "$ARRIVALS" \
    --raw "$RAW_V2" --out "$OUT/validation_v2_vs_day.csv" > "$OUT/validation.txt" 2>&1 \
    || status "validate failed (see validation.txt); amplitudes are still complete"
( cd "$OUT/amps" && md5sum amps_*.parquet > MD5SUMS )
status "DONE: $(du -sh "$OUT/amps" | cut -f1) in $OUT/amps"
touch "$OUT/DONE"
