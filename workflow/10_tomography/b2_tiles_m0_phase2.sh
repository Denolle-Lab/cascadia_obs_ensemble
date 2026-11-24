#!/bin/bash
# Track B, D-B046 phase 2: resolution tests of every tile restarted from M0, at damping 200 and the smoothing chosen
# in phase 1 (data/tomo_v1/b2_merge/tiles_m0/choice.csv, written by b2_tiles_m0.sh). Waits for phase 1 to finish,
# then starts b2_res_pipeline.sh <tile>_m0 with SMOOTH and DAMP=200 for the five tiles at once (each publishes to
# PR #33 when done).
#
#   cd ~/wt_tomo_res
#   setsid nohup bash workflow/10_tomography/b2_tiles_m0_phase2.sh 9>&- >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/tiles_m0_phase2.log 2>&1 < /dev/null &
set -uo pipefail
# bash reads a script while running it, and this one can wait hours for phase 1: run from a private copy so the repo
# file can be edited during the wait
if [ -z "${B2_P2_COPY:-}" ]; then
  cp "$0" /wd1/mdenolle_data/tomo_v1/b2_tomodd/.tiles_m0_phase2_running.sh
  B2_P2_COPY=1 CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash /wd1/mdenolle_data/tomo_v1/b2_tomodd/.tiles_m0_phase2_running.sh "$@"
fi
CODE=${CODE:?}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
export LD_LIBRARY_PATH=$ENV/lib     # pandas needs the env's libstdc++; a clean shell does not set it
RUNS=/wd1/mdenolle_data/tomo_v1/b2_tomodd
W=$CODE/workflow/10_tomography
CHOICE=$CODE/data/tomo_v1/b2_merge/tiles_m0/choice.csv
MAXJ=${MAXJ:-8}
log() { echo "$(date '+%F %T') [tiles_m0_phase2] $*"; }

log "waiting for phase 1 (stage analyse in $RUNS/tiles_m0.state)"
until grep -qx analyse $RUNS/tiles_m0.state 2>/dev/null; do
  # tiles_m0.log is append-only: only a FAILED after the last start counts, so a resumed phase 1 is not blocked
  awk '/\[tiles_m0\] start/ {f = 0} /\[tiles_m0\] stage .* FAILED/ {f = 1} END {exit !f}' $RUNS/tiles_m0.log 2>/dev/null &&
    { log "phase 1 failed, not starting"; exit 1; }
  sleep 60
done
[ -s $CHOICE ] || { log "no $CHOICE"; exit 1; }
log "choices: $(tr '\n' ' ' < $CHOICE)"
# read all choices first and stop on a parse error: an empty list would start nothing and still report success
rows=$($ENV/bin/python -c "
import ast, pandas as pd
for r in pd.read_csv('$CHOICE').itertuples():
    s = ast.literal_eval(r.smooth); print(r.region, *s, sep='\t')") || { log "could not read $CHOICE"; exit 1; }
[ -n "$rows" ] || { log "no tile in $CHOICE"; exit 1; }
n=0
while IFS=$'\t' read -r region w1 w2 w3; do
  log "start $region: damping 200, smoothing $w1 $w2 $w3"
  ( SMOOTH="$w1 $w2 $w3" DAMP=200 MAXJ=$MAXJ CODE=$CODE \
      setsid nohup bash $W/b2_res_pipeline.sh $region "prep queue run analyse publish" \
      >> $RUNS/$region/pipeline.log 2>&1 < /dev/null & )
  n=$((n + 1)); sleep 5
done <<< "$rows"
log "all $n started; progress in $RUNS/<tile>_m0/pipeline.log and res_tests/queue_d200_s<W1>/runner.log"
