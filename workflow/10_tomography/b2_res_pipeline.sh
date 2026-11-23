#!/bin/bash
# Track B, B2: build a tomoDD tile and run its resolution tests (checkerboards + spikes, D-B018) detached, then
# analyse and publish the results to GitHub (commit, push, PR). One process per tile; tiles can run at once.
#
#   cd ~/wt_tomo_res
#   setsid nohup bash workflow/10_tomography/b2_res_pipeline.sh TILE_v2 [STAGES] \
#       >> /wd1/mdenolle_data/tomo_v1/b2_tomodd/TILE_v2/pipeline.log 2>&1 < /dev/null &
#
# TILE_v2  a region of params tomodd.regions with a forward block (e.g. blanco_v2); its first build is TILE
#          (same name without _v2), whose picks give the P-1 reference times (Mendocino-Gorda pattern, D-B028).
# STAGES   default "build prep queue run analyse publish". Each stage is recorded in <tile>/pipeline.state when
#          it finishes and skipped afterwards; inside a stage, finished pieces are skipped too. After a crash,
#          a reboot or a lost session, rerun the same command and it resumes. Remove a line from pipeline.state
#          to force that stage again.
#
#   build    inputs TILE; P-1 reference (b2_fwd_p1.py); OBS S statics (b2_fwd_statics.py); inputs TILE_v2
#            (forward fixes incl. the P-4 station corrections); bounds-checked run check01 (D-B026)
#   prep     spike and checkerboard true models, synthetics and inversion inputs (TOMO_RES_REGION=TILE_v2)
#   queue    <tile>/res_tests/queue/queue.txt: real-data damping/smoothing scan (5 runs, tile root) + spikes at
#            smoothing 10/20/40 + checkerboards (Mendocino-Gorda v2 set: 48 runs)
#   run      the queue, at most MAXJ (default 10) inversions at a time for this tile
#   analyse  b2_res_checker.py analyse, b2_smooth_spike_analyse.py per smoothing, b2_tomodd_summary.py (scan)
#   publish  commit data/tomo_v1/b2_res_tests/TILE_v2 (files < 5 MB) + the queue/runner logs + the scan summary,
#            push, open the PR (or comment on it). Git steps hold a lock (several tiles publish to one branch).
#
# Code and outputs live in this worktree (CODE); the main checkout is never touched. Large files stay on /wd1.
set -uo pipefail
# bash reads a script while running it: run from a private copy so the repo file can be edited during a run
if [ -z "${B2_PIPELINE_COPY:-}" ]; then
  mkdir -p /wd1/mdenolle_data/tomo_v1/b2_tomodd/${1:?tile}
  cp "$0" /wd1/mdenolle_data/tomo_v1/b2_tomodd/$1/.pipeline_running.sh
  B2_PIPELINE_COPY=1 CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash /wd1/mdenolle_data/tomo_v1/b2_tomodd/$1/.pipeline_running.sh "$@"
fi

REGION=${1:?tile, e.g. blanco_v2}
STAGES=${2:-"build prep queue run analyse publish"}
CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib TOMO_RES_REGION=$REGION
MAXJ=${MAXJ:-10}
BASE_REGION=${REGION%_v2}
RUNS=/wd1/mdenolle_data/tomo_v1/b2_tomodd
RT=$RUNS/$REGION/res_tests
Q=$RT/queue
W=$CODE/workflow/10_tomography
OUT=$CODE/data/tomo_v1/b2_res_tests/$REGION
STATE=$RUNS/$REGION/pipeline.state
LOCK=/wd1/mdenolle_data/tomo_v1/publish.lock
BASE_BRANCH=${BASE_BRANCH:-tomo/res-tests-mg_v2}
mkdir -p $RUNS/$REGION

log() { echo "$(date '+%F %T') [$REGION] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }
py() { (cd $CODE && $PY "$@"); }
finished() { grep -q "exit 0" "$RUNS/$1/$2.log" 2>/dev/null && [ -s "$RUNS/$1/$2/Output_Files/Vp_model.dat" ]; }
param() { (cd $W && $PY -c "import yaml; T=yaml.safe_load(open('params.yaml'))['tomodd']; print($1)"); }

stage_build() {
  if [ ! -s $RUNS/$BASE_REGION/inputs/summary.json ]; then
    log "inputs $BASE_REGION"; py $W/b2_tomodd_inputs.py $BASE_REGION || return 1; fi
  if [ ! -s $RUNS/$REGION/p1_reference/region_picks_p1.parquet ]; then
    log "P-1 reference times"; py $W/b2_fwd_p1.py $BASE_REGION --out-region $REGION --nproc 16 --keep-grids || return 1; fi
  sf=$(param "T['regions']['$REGION']['forward']['statics_file']")
  if [ ! -s $CODE/$sf ]; then log "OBS S statics -> $sf"; py $W/b2_fwd_statics.py $REGION || return 1; fi
  if [ ! -s $RUNS/$REGION/inputs/summary.json ]; then
    log "inputs $REGION (forward fixes)"; py $W/b2_tomodd_inputs.py $REGION || return 1; fi
  if ! check01_ok; then
    rm -rf $RUNS/$REGION/check01
    log "bounds-checked run check01"; py $W/b2_tomodd_run.py $REGION check01 --check > $RUNS/$REGION/check01.log 2>&1
    check01_ok || { log "check01 failed (see $RUNS/$REGION/check01/stdout.txt)"; return 1; }
    log "check01 passed: no bounds error through ray tracing and LSQR"; fi
}

check01_ok() {   # notes/forward_accuracy.md 7.4: the -fcheck=bounds build stops after LSQR at the malformed FORMAT of
  # tomoDD-SE.f line 2324 (production builds do not); pass = exit 0, or that stop with no array-bound error before it
  local d=$RUNS/$REGION/check01
  grep -q "exit 0" $RUNS/$REGION/check01.log 2>/dev/null && return 0
  # the Fortran runtime error goes to stderr (time.txt); stdout.txt holds binary bytes, hence grep -a
  local f="$d/stdout.txt $d/time.txt"
  cat $f 2>/dev/null | command grep -aq "At line 2324 of file tomoDD-SE.f" &&
    cat $f | command grep -aq "Missing comma between descriptors" && cat $f | command grep -aq "~ lsqr" &&
    ! cat $f | command grep -aqi "out of bounds\|Index '"
}

parallel_py() {   # run "args" lines (one python call each) with at most $1 at a time
  local n=$1; shift
  printf '%s\n' "$@" | xargs -P $n -I{} bash -c "cd $CODE && $PY {} || exit 255"
}

stage_prep() {
  [ -d $RT/spike_null ] || { log "spike prep"; py $W/b2_smooth_spike_test.py --cfg res_tests prep || return 1; }
  local jobs=()
  for d in $RT/spike_*/; do t=$(basename $d); t=${t#spike_}
    [ -s $d/inputs/absolute.dat ] || jobs+=("$W/b2_smooth_spike_test.py --cfg res_tests syn $t > $d/syn.log 2>&1"); done
  [ ${#jobs[@]} -eq 0 ] || { log "spike synthetics: ${#jobs[@]}"; parallel_py 8 "${jobs[@]}" || return 1; }
  ls -d $RT/chk_L* > /dev/null 2>&1 || { log "checkerboard prep"; py $W/b2_res_checker.py prep || return 1; }
  jobs=()
  for d in $RT/chk_L???_p?/; do p=$(basename $d); p=${p#chk_}
    grep -q "forward done" $d/syn.log 2>/dev/null && [ -s $d/syn/absolute.syn ] && continue
    jobs+=("$W/b2_res_checker.py syn $p > $d/syn.log 2>&1"); done
  [ ${#jobs[@]} -eq 0 ] || { log "checkerboard synthetics: ${#jobs[@]}"; parallel_py 6 "${jobs[@]}" || return 1; }
  jobs=()
  for L in 040 060 100; do
    for pv in p0:clean p0:noise p0:noise_ploc p1:noise; do p=L${L}_${pv%%:*}; v=${pv##*:}
      [ -s $RT/chk_${p}_${v}/inputs/absolute.dat ] || jobs+=("$W/b2_res_checker.py inputs $p $v > $RT/inputs_${p}_${v}.log 2>&1"); done; done
  [ ${#jobs[@]} -eq 0 ] || { log "checkerboard inversion inputs: ${#jobs[@]}"; parallel_py 6 "${jobs[@]}" || return 1; }
}

stage_queue() {
  mkdir -p $Q; [ -s $Q/queue.txt ] && { log "queue exists"; return 0; }
  local R=$REGION/res_tests
  {
    for ds in "500 20 20 10" "1000 10 10 5" "1000 20 20 10" "1000 40 40 20" "2000 20 20 10"; do set -- $ds
      echo "$REGION d$1_s$2 $1 $2 $3 $4"; done
    for sm in "20 20 10" "10 10 5" "40 40 20"; do set -- $sm
      for d in $RT/spike_*/; do echo "$R/$(basename $d) inv_d1000_s$1 1000 $1 $2 $3"; done
      for L in 060 040 100; do
        for v in p0_noise p0_clean $( [ $1 = 20 ] && echo p1_noise p0_noise_ploc ); do echo "$R/chk_L${L}_$v inv_d1000_s$1 1000 $1 $2 $3"; done; done
    done
    echo END
  } > $Q/queue.txt
  log "queue: $(grep -vc END $Q/queue.txt) runs"
}

stage_run() {
  local n=0 started=0
  while read -r line; do
    [ "$line" = "END" ] && break
    set -- $line; local d=$1 tag=$2
    if finished $d $tag; then continue; fi
    [ -d $RUNS/$d/$tag ] && mv $RUNS/$d/$tag $RUNS/$d/$tag.partial.$(date +%s)   # left by a crash
    while [ $(jobs -rp | wc -l) -ge $MAXJ ]; do sleep 15; done
    ( cd $CODE; $PY $W/b2_tomodd_run.py $d $tag --damp $3 --smooth $4 $5 $6 > $RUNS/$d/$tag.log 2>&1; rc=$?
      $PY $W/b2_tomodd_backtransform.py $d $tag >> $RUNS/$d/$tag.log 2>&1
      echo "$(date +%T) done rc=$rc $d $tag" >> $Q/runner.log ) &
    echo "$(date +%T) start $d $tag" >> $Q/runner.log; started=$((started + 1)); sleep 2
  done < $Q/queue.txt
  wait
  echo "ALL DONE $(date +%T)" >> $Q/runner.log
  local bad=0
  while read -r line; do [ "$line" = "END" ] && break; set -- $line; finished $1 $2 || { bad=$((bad + 1)); log "not finished: $1 $2"; }; done < $Q/queue.txt
  log "queue: started $started this time, $bad not finished"
  [ $bad -eq 0 ]
}

stage_analyse() {
  log "checkerboard analysis"; py $W/b2_res_checker.py analyse || return 1
  local w
  for w in $(param "' '.join(str(s[0]) for s in T['res_tests']['smooth'])"); do
    log "spike analysis, smoothing $w"; py $W/b2_smooth_spike_analyse.py --cfg res_tests --inv inv_d1000_s$w || return 1; done
  log "damping/smoothing scan summary"; py $W/b2_tomodd_summary.py $REGION 'd*_s*' || return 1
  mkdir -p $OUT; cp $RUNS/$REGION/summary_runs.csv $OUT/scan_summary.csv 2>/dev/null || true
}

stage_publish() {
  exec 9> $LOCK; flock 9
  cd $CODE || return 1
  local br; br=$(git branch --show-current)
  mkdir -p $OUT/run_log; cp $Q/queue.txt $Q/runner.log $OUT/run_log/
  local big; big=$(cd $OUT && find . -type f -size +5M -printf '%p (%s B)\n')
  (cd $OUT && find . -type f -size +5M) | while read -r f; do echo "${OUT#$CODE/}/${f#./}"; done > $OUT/run_log/not_in_git.txt
  git add $(cd $OUT && find . -type f -size -5M -printf "${OUT#$CODE/}/%P\n")
  if git diff --cached --quiet; then log "nothing new to commit"; else
    git commit -q -F - <<EOF || return 1
Track B B2: $REGION resolution tests analysed $(date +%F) (checkerboards + spikes, D-B018; damping/smoothing scan)

$(grep -c ' done rc=0' $Q/runner.log) inversions done (queue and log in data/tomo_v1/b2_res_tests/$REGION/run_log/).
Inversion directories stay on $RUNS/$REGION.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  git push -q -u origin $br || return 1
  local summ; summ=$(sed -n '1,/^## Vs/p' $OUT/summary_tables.md | head -n -1)
  local note="**$REGION**: resolution tests analysed $(date '+%F %H:%M'), $(grep -c ' done rc=0' $Q/runner.log) inversions. Results in \`data/tomo_v1/b2_res_tests/$REGION/\`.${big:+ Files over 5 MB not in git: see \`run_log/not_in_git.txt\`.}

$summ"
  if gh pr view $br > /dev/null 2>&1; then
    gh pr comment $br --body "$note" > /dev/null && log "commented on $(gh pr view $br --json url -q .url)"
  else
    gh pr create --base $BASE_BRANCH --head $br --title "Track B: four more tomoDD tiles (D-B033) and their resolution tests" --body "$note

🤖 Generated with [Claude Code](https://claude.com/claude-code)" || return 1
  fi
}

log "start, stages: $STAGES (code $CODE, MAXJ $MAXJ)"
for s in $STAGES; do
  if done_stage $s; then log "stage $s already done, skipped"; continue; fi
  log "stage $s"
  if stage_$s; then mark $s; else log "stage $s FAILED, stopping (rerun the same command to resume)"; exit 1; fi
done
log "finished"
