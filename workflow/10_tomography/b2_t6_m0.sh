#!/bin/bash
# Track B: T6 joint-vs-stitched test (blanco + oregon) and M0 coarse whole-margin inversion (D-B044), damping 200
# (D-B045), detached; then compare and publish to PR #33. PI go 2026-11-23.
#
#   cd ~/wt_tomo_res
#   setsid nohup bash workflow/10_tomography/b2_t6_m0.sh >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/t6_m0.log 2>&1 < /dev/null &
#
# Stages (recorded in <runs>/t6_m0.state, skipped when done; rerun the same command to resume):
#   build    oregon_v2g inputs (dataset weight scale; oregon_v2's P-1 reference and statics), and the build stage of
#            b2_res_pipeline.sh for blanco_oregon_v2 and margin_m0_v2 (first build, P-1, statics, inputs, check01)
#   run      blanco_oregon_v2 d200_s20, oregon_v2g d200_s20, margin_m0_v2 d200_s20 and d1000_s20 (CND record),
#            each followed by the back-transform
#   analyse  b2_t6_joint.py (T6), b2_tomodd_summary.py for the three regions, b2_m0_report.py (M0 maps)
#   publish  commit data/tomo_v1/b2_merge/{t6,m0} and the run summaries, push, comment on PR #33
set -uo pipefail
if [ -z "${B2_T6_COPY:-}" ]; then
  cp "$0" /wd1/mdenolle_data/tomo_v1/b2_tomodd/.t6_m0_running.sh
  B2_T6_COPY=1 CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash /wd1/mdenolle_data/tomo_v1/b2_tomodd/.t6_m0_running.sh "$@"
fi
CODE=${CODE:?}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib
RUNS=/wd1/mdenolle_data/tomo_v1/b2_tomodd
W=$CODE/workflow/10_tomography
STATE=$RUNS/t6_m0.state
LOCK=/wd1/mdenolle_data/tomo_v1/publish.lock
OUTM=$CODE/data/tomo_v1/b2_merge

log() { echo "$(date '+%F %T') [t6_m0] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }
py() { (cd $CODE && $PY "$@"); }
finished() { grep -q "exit 0" "$RUNS/$1/$2.log" 2>/dev/null && [ -s "$RUNS/$1/$2/Output_Files/Vp_model.dat" ]; }

stage_build() {
  local rc=0
  ( [ -s $RUNS/oregon_v2g/inputs/summary.json ] || { mkdir -p $RUNS/oregon_v2g;
      log "inputs oregon_v2g"; py $W/b2_tomodd_inputs.py oregon_v2g > $RUNS/oregon_v2g/build.log 2>&1; } ) &
  local p1=$!
  for r in blanco_oregon_v2 margin_m0_v2; do
    mkdir -p $RUNS/$r
    ( CODE=$CODE bash $W/b2_res_pipeline.sh $r build >> $RUNS/$r/pipeline.log 2>&1 ) &
  done
  wait $p1 || rc=1
  wait || true
  [ -s $RUNS/oregon_v2g/inputs/summary.json ] || { log "oregon_v2g inputs failed"; rc=1; }
  for r in blanco_oregon_v2 margin_m0_v2; do
    grep -qx build $RUNS/$r/pipeline.state 2>/dev/null || { log "$r build failed (see $RUNS/$r/pipeline.log)"; rc=1; }
  done
  return $rc
}

stage_run() {
  local jobs=("blanco_oregon_v2 d200_s20 200" "oregon_v2g d200_s20 200" "margin_m0_v2 d200_s20 200"
              "margin_m0_v2 d1000_s20 1000")
  for j in "${jobs[@]}"; do set -- $j
    finished $1 $2 && continue
    [ -d $RUNS/$1/$2 ] && mv $RUNS/$1/$2 $RUNS/$1/$2.partial.$(date +%s)
    ( cd $CODE; $PY $W/b2_tomodd_run.py $1 $2 --damp $3 --smooth 20 20 10 > $RUNS/$1/$2.log 2>&1
      $PY $W/b2_tomodd_backtransform.py $1 $2 >> $RUNS/$1/$2.log 2>&1
      log "run done: $1 $2" ) &
    log "run started: $1 $2"; sleep 2
  done
  wait
  local bad=0
  for j in "${jobs[@]}"; do set -- $j; finished $1 $2 || { bad=1; log "not finished: $1 $2"; }; done
  [ $bad -eq 0 ]
}

stage_analyse() {
  log "T6 comparison"; py $W/b2_t6_joint.py > $RUNS/t6_compare.log 2>&1 || { log "b2_t6_joint.py failed"; return 1; }
  for r in blanco_oregon_v2 oregon_v2g margin_m0_v2; do
    py $W/b2_tomodd_summary.py $r 'd*_s*' >> $RUNS/t6_compare.log 2>&1 || return 1; done
  mkdir -p $OUTM/t6 $OUTM/m0
  cp $RUNS/blanco_oregon_v2/summary_runs.csv $OUTM/t6/blanco_oregon_v2_scan_summary.csv
  cp $RUNS/oregon_v2g/summary_runs.csv $OUTM/t6/oregon_v2g_scan_summary.csv
  cp $RUNS/margin_m0_v2/summary_runs.csv $OUTM/m0/scan_summary.csv
  for r in blanco_oregon_v2 margin_m0_v2; do cp $RUNS/$r/inputs/summary.json $OUTM/$( [ $r = margin_m0_v2 ] && echo m0 || echo t6 )/${r}_inputs_summary.json; done
  cp $RUNS/oregon_v2g/inputs/summary.json $OUTM/t6/oregon_v2g_inputs_summary.json
  log "M0 report"; py $W/b2_m0_report.py >> $RUNS/t6_compare.log 2>&1 || { log "b2_m0_report.py failed"; return 1; }
}

stage_publish() {
  exec 9> $LOCK; flock 9
  cd $CODE || return 1
  local br; br=$(git branch --show-current)
  git add $(find data/tomo_v1/b2_merge/t6 data/tomo_v1/b2_merge/m0 -type f -size -5M)
  if git diff --cached --quiet; then log "nothing new to commit"; else
    git commit -q -F - <<EOF || return 1
Track B: T6 joint vs stitched (blanco + oregon) and M0 coarse whole-margin inversion, damping 200 (D-B044, D-B045)

Runs on $RUNS: blanco_oregon_v2 d200_s20, oregon_v2g d200_s20, margin_m0_v2 d200_s20 / d1000_s20.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  git push -q origin $br || return 1
  local body
  body="**T6 joint vs stitched (blanco + oregon, damping 200, dataset weight scale) and M0 whole-margin inversion: done** ($(date '+%F %H:%M'))

$(sed -n '1,40p' data/tomo_v1/b2_merge/t6/blanco_oregon_v2_d200_s20/summary.md)

M0 scan:
\`\`\`
$(cut -d, -f1,2,6,8,9,10,13,14,15,16,21,23 data/tomo_v1/b2_merge/m0/scan_summary.csv | column -t -s,)
\`\`\`
Files: data/tomo_v1/b2_merge/t6/, data/tomo_v1/b2_merge/m0/. Not yet reviewed by Claude in the session."
  gh pr comment 33 --body "$body" > /dev/null && log "commented on PR #33"
}

log "start (code $CODE)"
for s in build run analyse publish; do
  done_stage $s && { log "stage $s already done"; continue; }
  log "stage $s"
  stage_$s || { log "stage $s FAILED"; exit 1; }
  mark $s
done
log "finished"
