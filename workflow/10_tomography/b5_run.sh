#!/bin/bash
# D-B048 driver: data, tomoDD-SE (relocation only) and GrowClust3D runs, bootstraps and comparison, per case.
# Resumable: every step skips what is already done (reloc.csv / summary.json present). Run detached from the worktree:
#   setsid nohup bash workflow/10_tomography/b5_run.sh [CASE ...] >> /wd1/mdenolle_data/b5_test/run.log 2>&1 < /dev/null &
# Default cases: f1803 synth synth_tdd synth_s025 f0399 f1051 (b5_test.cases; synth* = b5_synth.py).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY="env LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python"
OUT=/wd1/mdenolle_data/b5_test
NBOOT_TDD=${NBOOT_TDD:-30}
PAR=${PAR:-12}
CASES=${*:-"f1803 synth synth_tdd synth_s025 f0399 f1051"}
cd "$HERE"
log() { echo "[b5 $(date '+%F %T')] $*"; }

for c in $CASES; do
  log "case $c"
  if [ ! -f "$OUT/$c/data/summary.json" ]; then
    if [ "$c" = synth ]; then $PY b5_synth.py || { log "synth failed"; continue; }
    elif [ "$c" = synth_tdd ]; then $PY b5_synth.py --case synth_tdd --source tomodd || { log "synth_tdd failed"; continue; }
    elif [ "$c" = synth_s025 ]; then $PY b5_synth.py --case synth_s025 --start-scale 0.25 || { log "synth_s025 failed"; continue; }
    else $PY b5_prep.py "$c" || { log "prep $c failed"; continue; }; fi
  fi
  for v in hier dd; do $PY b5_tomodd.py "$c" "$v" || log "tomodd $c $v failed"; done
  for v in catalog default; do $PY b5_growclust.py "$c" "$v" || log "growclust $c $v failed"; done
  for v in hier dd; do
    seq 1 "$NBOOT_TDD" | xargs -P "$PAR" -I{} sh -c "$PY b5_tomodd.py $c $v --boot {} > /dev/null 2>&1 || echo 'boot {} failed'"
    log "tomodd $c $v bootstraps: $(ls -d $OUT/$c/tomodd/${v}_b*/reloc.csv 2>/dev/null | wc -l) of $NBOOT_TDD"
  done
  $PY b5_compare.py "$c" > "$OUT/$c/compare.txt" 2>&1 || log "compare $c failed"
  log "case $c done"
done
log "all done"
