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
#   publish  commit data/tomo_v1/b2_res_tests/TILE_v2 (files < 5 MB) + the queue/runner logs (run_log[_spikec][_s<W1>])
#            + the scan summary, push, open the PR (or comment on it). Git steps hold a lock (several tiles publish to
#            one branch); only these files are committed, whatever else is staged in the worktree.
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
log() { echo "$(date '+%F %T') [$REGION] $*"; }
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
# DAMP=<d>: resolution tests (and real-data runs at smoothing 10/20/40) at damping d instead of res_tests.damp;
# own queue (queue_d<d>), own stage records (queue_d<d> ...), results in data/tomo_v1/b2_res_tests/<tile>/d<d>/
DAMP=${DAMP:-}
# the Python scripts read TOMO_RES_DAMP, TOMO_RES_SMOOTH and TOMO_SPIKE_PLACEMENT: set them only from DAMP, SMOOTH and
# SPIKES_ONLY below, so a value inherited from the calling shell cannot mix runs. Exits before the stages log
# "stage setup FAILED: <reason>", which b4_relocate.sh (stage wait) reads as a failed pipeline
unset TOMO_RES_DAMP TOMO_RES_SMOOTH TOMO_SPIKE_PLACEMENT
[ -n "$DAMP" ] && export TOMO_RES_DAMP=$DAMP
# the damping as Python spells it in run and output names (res_cfg: float, int if whole, format g; a per-tile
# res_tests.regions.<tile>.damp when DAMP is unset), so queue tags, OUT and the checker's run glob always agree
DTEST=$(cd $W && $PY -c "import b2_smooth_spike_test as s; print(format(s.res_cfg()['damp'], 'g'))") ||
  { log "stage setup FAILED: cannot read the test damping (DAMP='$DAMP')"; exit 1; }
TAGD=${DAMP:+_d$DTEST}
# SPIKES_ONLY=1: spikes at the event clusters (TOMO_SPIKE_PLACEMENT=clusters, dirs spikec_*) with the DWS mask; no
# checkerboards, no real-data scan; own stage records (prep_spikec, queue_spikec[_d<d>] ...)
SPIKES_ONLY=${SPIKES_ONLY:-}
TAGM=${SPIKES_ONLY:+_spikec}
SPFX=spike_; [ -n "$SPIKES_ONLY" ] && { SPFX=spikec_; export TOMO_SPIKE_PLACEMENT=clusters; }
[ -n "$DAMP" ] && OUT=$OUT/d$DTEST
# SMOOTH="W1 W2 W3" (D-B046 phase 2): spikes and checkerboards at this one smoothing (all variants, mask from it), no
# real-data scan (done by b2_tiles_m0.sh); own queue and stage records (..._s<W1>)
SMOOTH=${SMOOTH:-}
if [ -n "$SMOOTH" ]; then   # tags carry W1 only (inv_d<D>_s<W1>, and the checker reads int(W1) back from them): another
  # W2 or W3 at the same W1 would reuse or overwrite the runs of the standard triple, so accept only W1 W1 W1/2
  read -r w1 w2 w3 x <<< "$SMOOTH"
  [[ "$w1 $w2 $w3" =~ ^[1-9][0-9]*\ [1-9][0-9]*\ [1-9][0-9]*$ && -z $x && "$w2" = "$w1" && $((2 * w3)) = "$w1" ]] ||
    { log "stage setup FAILED: SMOOTH must be \"W1 W1 W1/2\" in whole numbers (run tags carry W1 only), not \"$SMOOTH\""; exit 1; }
  SMOOTH="$w1 $w2 $w3"
fi
TAGS=${SMOOTH:+_s${SMOOTH%% *}}
MS=${SMOOTH%% *}; MS=${MS:-20}                      # smoothing (weight1) of the resolution mask
[ -n "$SMOOTH" ] && export TOMO_RES_SMOOTH="$SMOOTH"
Q=$RT/queue$TAGM$TAGD$TAGS
# published queue and runner logs, one folder per mode: a spikes-only or one-smoothing run at a damping shares OUT
# with the full run at that damping and overwrote its run_log/
RL=$OUT/run_log$TAGM$TAGS
mkdir -p $RUNS/$REGION

skey() { case $1 in build) echo $1 ;; prep) echo "prep$TAGM" ;; *) echo "$1$TAGM$TAGD$TAGS" ;; esac; }   # build shared by all
done_stage() { grep -qx "$(skey $1)" "$STATE" 2>/dev/null; }
mark() { skey $1 >> "$STATE"; log "stage $(skey $1) done"; }
py() { (cd $CODE && $PY "$@"); }
finished() { grep -q "exit 0" "$RUNS/$1/$2.log" 2>/dev/null && [ -s "$RUNS/$1/$2/Output_Files/Vp_model.dat" ]; }
weights() { awk '/weight1/ {f = 1; next} f && !/^\*/ {print $1, $2, $3; exit}' "$RUNS/$1/$2/tomoDD-SE.inp" 2>/dev/null; }
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
  # the check must be newer than the inputs it ran on (summary.json is written last): after a rebuild, check again
  [ $RUNS/$REGION/check01.log -nt $RUNS/$REGION/inputs/summary.json ] || return 1
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
  [ -d $RT/${SPFX}null ] || { log "spike prep ($SPFX)"; py $W/b2_smooth_spike_test.py --cfg res_tests prep || return 1; }
  local jobs=()
  # a job is done when its manifest exists: it is written last, while absolute.dat comes before dt.ct (and the
  # noise_ploc event.dat), so a job stopped in between left inputs that only look complete
  for d in $RT/${SPFX}*/; do t=$(basename $d); t=${t#$SPFX}
    [ -s $d/inputs/absolute.dat ] && [ -e $d/b2_smooth_spike_syn_$t.manifest.json ] || jobs+=("$W/b2_smooth_spike_test.py --cfg res_tests syn $t > $d/syn.log 2>&1"); done
  [ ${#jobs[@]} -eq 0 ] || { log "spike synthetics: ${#jobs[@]}"; parallel_py 8 "${jobs[@]}" || return 1; }
  [ -n "$SPIKES_ONLY" ] && return 0
  ls -d $RT/chk_L* > /dev/null 2>&1 || { log "checkerboard prep"; py $W/b2_res_checker.py prep || return 1; }
  jobs=()
  for d in $RT/chk_L???_p?/; do p=$(basename $d); p=${p#chk_}
    grep -q "forward done" $d/syn.log 2>/dev/null && [ -s $d/syn/absolute.syn ] && continue
    jobs+=("$W/b2_res_checker.py syn $p > $d/syn.log 2>&1"); done
  [ ${#jobs[@]} -eq 0 ] || { log "checkerboard synthetics: ${#jobs[@]}"; parallel_py 6 "${jobs[@]}" || return 1; }
  jobs=()
  for L in 040 060 100; do
    for pv in p0:clean p0:noise p0:noise_ploc p1:noise; do p=L${L}_${pv%%:*}; v=${pv##*:}
      [ -s $RT/chk_${p}_${v}/inputs/absolute.dat ] && [ -e $RT/chk_${p}_${v}/b2_res_checker_inputs_${p}_${v}.manifest.json ] ||
        jobs+=("$W/b2_res_checker.py inputs $p $v > $RT/inputs_${p}_${v}.log 2>&1"); done; done
  [ ${#jobs[@]} -eq 0 ] || { log "checkerboard inversion inputs: ${#jobs[@]}"; parallel_py 6 "${jobs[@]}" || return 1; }
}

stage_queue() {
  mkdir -p $Q; [ -s $Q/queue.txt ] && { log "queue exists"; return 0; }
  local R=$REGION/res_tests
  {
    if [ -n "$DAMP" ]; then scan="$DTEST 10 10 5|$DTEST 20 20 10|$DTEST 40 40 20"
    else scan="500 20 20 10|1000 10 10 5|1000 20 20 10|1000 40 40 20|2000 20 20 10"; fi
    local IFS_OLD=$IFS; IFS='|'; set -f; local scans=($scan); IFS=$IFS_OLD; set +f
    [ -n "$SPIKES_ONLY$SMOOTH" ] && scans=()
    for ds in "${scans[@]}"; do set -- $ds
      echo "$REGION d$1_s$2 $1 $2 $3 $4"; done
    local smooths=("20 20 10" "10 10 5" "40 40 20"); [ -n "$SMOOTH" ] && smooths=("$SMOOTH")
    for sm in "${smooths[@]}"; do set -- $sm
      for d in $RT/${SPFX}*/; do echo "$R/$(basename $d) inv_d${DTEST}_s$1 $DTEST $1 $2 $3"; done
      [ -n "$SPIKES_ONLY" ] && continue
      for L in 060 040 100; do
        for v in p0_noise p0_clean $( [ $1 = $MS ] && echo p1_noise p0_noise_ploc ); do echo "$R/chk_L${L}_$v inv_d${DTEST}_s$1 $DTEST $1 $2 $3"; done; done
    done
    echo END
  } > $Q/queue.txt
  log "queue: $(grep -vc END $Q/queue.txt) runs"
}

stage_run() {
  local n=0 started=0 bad=0
  # tags carry W1 only: a finished run under a tag but with other weights is not the one queued; stop, do not reuse it
  while read -r line; do [ "$line" = "END" ] && break; set -- $line
    finished $1 $2 && [ "$(weights $1 $2)" != "$4 $5 $6" ] && { bad=1; log "$1/$2 exists with smoothing $(weights $1 $2), queue asks $4 $5 $6"; }
  done < $Q/queue.txt
  [ $bad -eq 0 ] || return 1
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
  while read -r line; do [ "$line" = "END" ] && break; set -- $line; finished $1 $2 || { bad=$((bad + 1)); log "not finished: $1 $2"; }; done < $Q/queue.txt
  log "queue: started $started this time, $bad not finished"
  [ $bad -eq 0 ]
}

stage_analyse() {
  [ -n "$SPIKES_ONLY" ] || { log "checkerboard analysis"; py $W/b2_res_checker.py analyse --mask-smooth $MS || return 1; }
  local w ws
  ws=$(param "' '.join(str(s[0]) for s in T['res_tests']['smooth'])"); [ -n "$SMOOTH" ] && ws=$MS
  for w in $ws; do
    log "spike analysis, damping $DTEST, smoothing $w"; py $W/b2_smooth_spike_analyse.py --cfg res_tests --inv inv_d${DTEST}_s$w || return 1; done
  log "damping/smoothing scan summary"; py $W/b2_tomodd_summary.py $REGION 'd*_s*' || return 1
  mkdir -p $OUT; cp $RUNS/$REGION/summary_runs.csv $OUT/scan_summary.csv 2>/dev/null || true
}

stage_publish() {
  exec 9> $LOCK; flock 9
  cd $CODE || return 1
  local br; br=$(git branch --show-current)
  mkdir -p $RL; cp $Q/queue.txt $Q/runner.log $RL/
  # 5 MB limit in bytes: find rounds -size 5M up to whole MiB, so files of 4-5 MiB were in neither list
  local big files rel=${OUT#$CODE/}
  big=$(cd $OUT && find . -type f ! -size -5242880c -printf '%p (%s B)\n')
  (cd $OUT && find . -type f ! -size -5242880c -printf "$rel/%P\n") > $RL/not_in_git.txt
  mapfile -t files < <(cd $OUT && find . -type f -size -5242880c -printf "$rel/%P\n")
  [ ${#files[@]} -gt 0 ] || { log "no file to publish in $OUT"; return 1; }
  # stop if git add fails (e.g. index.lock held by a session in this worktree), and commit only these files, not
  # whatever else is staged
  git add -- "${files[@]}" || return 1
  local what="checkerboards + spikes, D-B018; damping/smoothing scan"
  [ -n "$SMOOTH" ] && what="checkerboards + spikes, D-B018, smoothing $SMOOTH"
  [ -n "$SPIKES_ONLY" ] && what="spikes at the event clusters, DWS mask"
  if git diff --cached --quiet -- "${files[@]}"; then log "nothing new to commit"; else
    git commit -q -F - -- "${files[@]}" <<EOF || return 1
Track B B2: $REGION resolution tests${DAMP:+ at damping $DTEST} analysed $(date +%F) ($what)

$(grep -c ' done rc=0' $Q/runner.log) inversions done (queue and log in ${RL#$CODE/}/).
Inversion directories stay on $RUNS/$REGION.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  git push -q -u origin $br || return 1
  local summ
  if [ -n "$SPIKES_ONLY" ]; then
    summ="Spikes at the event clusters, DWS mask $(param "T['res_tests'].get('dws_min', 100)") (smoothing 20). Recovery by DWS band:
\`\`\`
$(cat $OUT/${SPFX}inv_d${DTEST}_s20/recovery_by_dws.csv)
\`\`\`
By depth (sampled spikes):
\`\`\`
$(cut -d, -f1,2,14- $OUT/${SPFX}inv_d${DTEST}_s20/by_depth.csv)
\`\`\`"
  else summ=$(sed -n '1,/^## Vs/p' $OUT/summary_tables.md | head -n -1); fi
  local note="**$REGION${DAMP:+, damping $DTEST}**: resolution tests analysed $(date '+%F %H:%M'), $(grep -c ' done rc=0' $Q/runner.log) inversions. Results in \`$rel/\`.${big:+ Files over 5 MB not in git: see \`${RL#$CODE/}/not_in_git.txt\`.}

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
  if stage_$s; then mark $s; else log "stage $(skey $s) FAILED, stopping (rerun the same command to resume)"; exit 1; fi
done
log "finished"
