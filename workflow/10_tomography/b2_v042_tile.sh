#!/bin/bash
# v0.4.2 (D-B099): one re-inverted tile, the v0.4.1 chain's steps (b2_v041_chain.py flr_/ver_/chk_/run_) on the rebuilt
# inputs of NEW (made by b2_tomodd_inputs.py NEW [+ b2_v040_noise.py NEW for a noise-floor copy]):
#   floor write + check -> inputs check against OLD (b2_v042_verify_inputs.py) -> bounds-checked check01 run with the
#   floor (main stages only) -> joint run TAG with v0.4.1's damping and smoothing -> back-transform.
# usage (repo root): workflow/10_tomography/b2_v042_tile.sh NEW OLD TAG SMOOTH [nocheck]
#   e.g. b2_v042_tile.sh washington_m0g washington_m0f d200_s480 480
# nice 10; binaries from params tomodd.regions.NEW.bins (the 2026-11-26 freeze).
set -u
NEW=$1; OLD=$2; TAG=$3; SM=$4; CHK=${5:-check}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY="env LD_LIBRARY_PATH=$ENV/lib nice -n 10 $ENV/bin/python"
W=workflow/10_tomography
RUNS=$($PY -c "import sys; sys.path.insert(0, '$W'); from manifest import load_params; print(load_params()['tomodd']['runs'])")
log() { echo "$(date '+%F %T') [v042 $NEW] $*"; }
[ -e "$RUNS/$NEW/inputs/summary.json" ] || { log "no inputs"; exit 1; }
log "floor"
$PY $W/b2_tomodd_floor.py write "$NEW" > "$RUNS/$NEW/floor_write.log" 2>&1 && \
  $PY $W/b2_tomodd_floor.py check "$NEW" > "$RUNS/$NEW/floor_check.log" 2>&1 || { log "FAILED floor"; exit 1; }
log "verify inputs against $OLD"
$PY $W/b2_v042_verify_inputs.py "$NEW" "$OLD" > "$RUNS/$NEW/verify_vs_$OLD.log" 2>&1 || { log "FAILED verify"; exit 1; }
if [ "$CHK" = check ]; then
  log "check01 (bounds-checked, floored)"
  rm -rf "${RUNS:?}/$NEW/check01"
  $PY $W/b2_tomodd_run.py "$NEW" check01 --check > "$RUNS/$NEW/check01.log" 2>&1
  grep -q 'exit 0' "$RUNS/$NEW/check01.log" || { log "FAILED check01"; exit 1; }
fi
RD="$RUNS/$NEW/$TAG"
[ -d "$RD" ] && mv "$RD" "$RD.partial.$(date +%s)"
log "run $TAG (damp 200, smooth $SM $SM $((SM / 2)))"
$PY $W/b2_tomodd_run.py "$NEW" "$TAG" --damp 200 --smooth "$SM" "$SM" $((SM / 2)) > "$RD.log" 2>&1 && \
  $PY $W/b2_tomodd_backtransform.py "$NEW" "$TAG" >> "$RD.log" 2>&1 || { log "FAILED run"; exit 1; }
log "done ($(tail -c 200 "$RD/time.txt" | tail -1))"
