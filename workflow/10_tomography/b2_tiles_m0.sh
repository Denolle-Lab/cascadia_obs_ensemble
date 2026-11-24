#!/bin/bash
# Track B, D-B046 phase 1: every tile restarted from M0 (margin_m0_v2 d200_s480), damping 200, dataset weight scale;
# smoothing scan per tile (params tomodd.merge.tile_smooth_scan); smoothing chosen by the P-M11 rule; publish.
# Phase 2 (resolution tests at the chosen smoothing) follows separately.
#
#   cd ~/wt_tomo_res
#   setsid nohup bash workflow/10_tomography/b2_tiles_m0.sh >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/tiles_m0.log 2>&1 < /dev/null &
#
# Stages (recorded in <runs>/tiles_m0.state; rerun the same command to resume):
#   build    per tile: link <tile>_m0/p1_reference -> <tile>_v2/p1_reference (P-1 does not depend on the start
#            model), then the build stage of b2_res_pipeline.sh <tile>_m0 (inputs with the M0 start and P-4
#            station corrections recomputed in it, bounds-checked check01); five tiles in parallel
#   scan     5 tiles x 5 smoothings at damping 200, all in parallel, each followed by the back-transform
#   analyse  b2_tomodd_summary.py per tile, b2_tiles_m0_choose.py (rule), b2_m0_report.py maps per tile
#   publish  commit data/tomo_v1/b2_merge/tiles_m0, push, comment on PR #33
set -uo pipefail
if [ -z "${B2_TM0_COPY:-}" ]; then
  cp "$0" /wd1/mdenolle_data/tomo_v1/b2_tomodd/.tiles_m0_running.sh
  B2_TM0_COPY=1 CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash /wd1/mdenolle_data/tomo_v1/b2_tomodd/.tiles_m0_running.sh "$@"
fi
CODE=${CODE:?}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib
RUNS=/wd1/mdenolle_data/tomo_v1/b2_tomodd
W=$CODE/workflow/10_tomography
STATE=$RUNS/tiles_m0.state
LOCK=/wd1/mdenolle_data/tomo_v1/publish.lock
OUT=$CODE/data/tomo_v1/b2_merge/tiles_m0
TILES="mendocino_gorda_c blanco oregon washington ridges_north"

log() { echo "$(date '+%F %T') [tiles_m0] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }
finished() { grep -q "exit 0" "$RUNS/$1/$2.log" 2>/dev/null && [ -s "$RUNS/$1/$2/Output_Files/Vp_model.dat" ]; }
scans() { (cd $W && $PY -c "import yaml; [print(*s) for s in yaml.safe_load(open('params.yaml'))['tomodd']['merge']['tile_smooth_scan']]"); }

stage_build() {
  for t in $TILES; do
    mkdir -p $RUNS/${t}_m0
    [ -e $RUNS/${t}_m0/p1_reference ] || ln -s ../${t}_v2/p1_reference $RUNS/${t}_m0/p1_reference
    ( CODE=$CODE bash $W/b2_res_pipeline.sh ${t}_m0 build >> $RUNS/${t}_m0/pipeline.log 2>&1 ) &
    sleep 2
  done
  wait
  local rc=0
  for t in $TILES; do
    grep -qx build $RUNS/${t}_m0/pipeline.state 2>/dev/null || { log "${t}_m0 build failed (see $RUNS/${t}_m0/pipeline.log)"; rc=1; }
  done
  return $rc
}

stage_scan() {
  local t s
  while read -r w1 w2 w3; do
    for t in $TILES; do
      r=${t}_m0; tag=d200_s$w1
      finished $r $tag && continue
      [ -d $RUNS/$r/$tag ] && mv $RUNS/$r/$tag $RUNS/$r/$tag.partial.$(date +%s)
      ( cd $CODE; $PY $W/b2_tomodd_run.py $r $tag --damp 200 --smooth $w1 $w2 $w3 > $RUNS/$r/$tag.log 2>&1
        $PY $W/b2_tomodd_backtransform.py $r $tag >> $RUNS/$r/$tag.log 2>&1
        log "run done: $r $tag" ) < /dev/null &
      log "run started: $r $tag"; sleep 2
    done
  done < <(scans)
  wait
  local bad=0
  while read -r w1 w2 w3; do for t in $TILES; do finished ${t}_m0 d200_s$w1 || { bad=1; log "not finished: ${t}_m0 d200_s$w1"; }; done; done < <(scans)
  [ $bad -eq 0 ]
}

stage_analyse() {
  mkdir -p $OUT
  for t in $TILES; do
    (cd $CODE && $PY $W/b2_tomodd_summary.py ${t}_m0 'd*_s*' >> $RUNS/tiles_m0_analyse.log 2>&1) || return 1
    cp $RUNS/${t}_m0/summary_runs.csv $OUT/${t}_scan_summary.csv
    cp $RUNS/${t}_m0/inputs/summary.json $OUT/${t}_inputs_summary.json
    (cd $CODE && $PY $W/b2_m0_report.py ${t}_m0 $OUT/$t >> $RUNS/tiles_m0_analyse.log 2>&1) || return 1
  done
  (cd $CODE && $PY $W/b2_tiles_m0_choose.py >> $RUNS/tiles_m0_analyse.log 2>&1) || return 1
}

stage_publish() {
  exec 9> $LOCK; flock 9
  cd $CODE || return 1
  local br; br=$(git branch --show-current)
  git add $(find data/tomo_v1/b2_merge/tiles_m0 -type f -size -5M)
  if git diff --cached --quiet; then log "nothing new to commit"; else
    git commit -q -F - <<EOF || return 1
Track B: tiles restarted from M0 (D-B046), smoothing scan at damping 200 and smoothing chosen by rule

Runs on $RUNS/<tile>_m0/d200_s*. Choice in data/tomo_v1/b2_merge/tiles_m0/choice.csv.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  git push -q origin $br || return 1
  gh pr comment 33 --body "**Tiles restarted from M0 (D-B046): smoothing scan at damping 200 done** ($(date '+%F %H:%M'))

Chosen smoothing per tile (largest within $(cd $W && $PY -c "import yaml; print(yaml.safe_load(open('params.yaml'))['tomodd']['merge']['smooth_rule_tol'])") of the best weighted RMS):
\`\`\`
$(cat $OUT/choice.csv)
\`\`\`
Full scans: data/tomo_v1/b2_merge/tiles_m0/summary.md; maps per tile in data/tomo_v1/b2_merge/tiles_m0/<tile>/. Not yet reviewed by Claude in the session; resolution tests (phase 2) not started." > /dev/null && log "commented on PR #33"
}

log "start (code $CODE)"
for s in build scan analyse publish; do
  done_stage $s && { log "stage $s already done"; continue; }
  log "stage $s"
  stage_$s || { log "stage $s FAILED"; exit 1; }
  mark $s
done
log "finished"
