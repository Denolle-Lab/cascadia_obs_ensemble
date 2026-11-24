#!/bin/bash
# Track B, D-B046 phase 2: resolution tests of every tile restarted from M0, at damping 200 and the smoothing chosen
# in phase 1 (data/tomo_v1/b2_merge/tiles_m0/choice.csv, written by b2_tiles_m0.sh). Waits for phase 1 to finish,
# then starts b2_res_pipeline.sh <tile>_m0 with SMOOTH and DAMP=200 for the five tiles at once (each publishes to
# PR #33 when done).
#
#   cd ~/wt_tomo_res
#   setsid nohup bash workflow/10_tomography/b2_tiles_m0_phase2.sh 9>&- >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/tiles_m0_phase2.log 2>&1 < /dev/null &
set -uo pipefail
CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
RUNS=/wd1/mdenolle_data/tomo_v1/b2_tomodd
W=$CODE/workflow/10_tomography
CHOICE=$CODE/data/tomo_v1/b2_merge/tiles_m0/choice.csv
MAXJ=${MAXJ:-8}
log() { echo "$(date '+%F %T') [tiles_m0_phase2] $*"; }

log "waiting for phase 1 (stage analyse in $RUNS/tiles_m0.state)"
until grep -qx analyse $RUNS/tiles_m0.state 2>/dev/null; do
  grep -q "stage .* FAILED" $RUNS/tiles_m0.log 2>/dev/null && { log "phase 1 failed, not starting"; exit 1; }
  sleep 60
done
[ -s $CHOICE ] || { log "no $CHOICE"; exit 1; }
log "choices: $(tr '\n' ' ' < $CHOICE)"
while IFS=$'\t' read -r region w1 w2 w3; do
  log "start $region: damping 200, smoothing $w1 $w2 $w3"
  ( SMOOTH="$w1 $w2 $w3" DAMP=200 MAXJ=$MAXJ CODE=$CODE \
      setsid nohup bash $W/b2_res_pipeline.sh $region "prep queue run analyse publish" \
      >> $RUNS/$region/pipeline.log 2>&1 < /dev/null & )
  sleep 5
done < <($ENV/bin/python -c "
import ast, pandas as pd
for r in pd.read_csv('$CHOICE').itertuples():
    s = ast.literal_eval(r.smooth); print(r.region, *s, sep='\t')")
log "all five started; progress in $RUNS/<tile>_m0/pipeline.log and res_tests/queue_d200_s<W1>/runner.log"
