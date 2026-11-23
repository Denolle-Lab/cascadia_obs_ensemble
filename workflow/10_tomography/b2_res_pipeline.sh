#!/bin/bash
# Track B, B2: resolution tests (checkerboards + spikes, D-B018) run detached, then analysed and published as a PR.
#
#   setsid nohup bash workflow/10_tomography/b2_res_pipeline.sh REGION [STAGES] > LOG 2>&1 < /dev/null &
#
# REGION  tomoDD region under tomodd.runs with a res_tests/ dir (prepared by b2_res_checker.py prep/syn/inputs and
#         b2_smooth_spike_test.py --cfg res_tests prep/syn, and a queue/queue.txt + runner.sh)
# STAGES  any of "run analyse publish" (default: all three, in that order). Every stage skips work already done,
#         so the same command resumes after a crash or a lost session.
#
#   run      start queue/runner.sh unless runner.log says ALL DONE; wait until it does
#   analyse  b2_res_checker.py analyse + b2_smooth_spike_analyse.py for each smoothing in res_tests.smooth
#   publish  copy the code (workflow/10_tomography) and the small outputs (data/tomo_v1/b2_res_tests, < 5 MB each,
#            plus the queue and runner logs) from the code tree into a worktree on branch tomo/res-tests-<region>,
#            commit, push and open (or update) a PR against tomo/track-b. Large files (models, synthetics)
#            stay on /wd1 and are listed in the PR with their paths.
#
# The code tree (CODE) is where the analysis runs and writes; it is never committed to. Only the worktree is.
set -uo pipefail

REGION=${1:?region, e.g. mendocino_gorda_v2}
STAGES=${2:-"run analyse publish"}
CODE=${CODE:-/home/mdenolle/cascadia_obs_ensemble}
ENV=$CODE/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib
RUNS=/wd1/mdenolle_data/tomo_v1/b2_tomodd
RT=$RUNS/$REGION/res_tests
Q=$RT/queue
SHORT=$(echo "$REGION" | sed 's/mendocino_gorda/mg/')
BRANCH=tomo/res-tests-$SHORT
WT=/home/mdenolle/wt_tomo_res
OUT=$CODE/data/tomo_v1/b2_res_tests
STATE=$RT/pipeline.state          # one line per finished stage

log() { echo "$(date '+%F %T') [$REGION] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }

stage_run() {
  if grep -q "^ALL DONE" $Q/runner.log 2>/dev/null; then log "runner already finished"; return 0; fi
  if ! pgrep -f "$Q/runner.sh" > /dev/null; then
    log "starting runner"; (cd $CODE && setsid nohup bash $Q/runner.sh >> $Q/runner.out 2>&1 < /dev/null &)
  fi
  until grep -q "^ALL DONE" $Q/runner.log 2>/dev/null; do sleep 120; done
  nbad=$(grep " done " $Q/runner.log | grep -vc "rc=0")
  log "runner finished: $(grep -c ' done rc=0' $Q/runner.log) ok, $nbad failed"
  [ "$nbad" -eq 0 ]
}

stage_analyse() {
  cd $CODE || return 1
  log "checkerboard analysis"
  $PY workflow/10_tomography/b2_res_checker.py analyse || return 1
  local w
  for w in $($PY -c "import yaml; c=yaml.safe_load(open('workflow/10_tomography/params.yaml'))['tomodd']['res_tests']; print(' '.join(str(w[0]) for w in c['smooth']))"); do
    log "spike analysis, smoothing $w"
    $PY workflow/10_tomography/b2_smooth_spike_analyse.py --cfg res_tests --inv inv_d1000_s$w || return 1
  done
}

stage_publish() {
  if [ ! -d $WT ]; then
    git -C $CODE fetch -q origin && git -C $CODE worktree add -q -B $BRANCH $WT origin/tomo/track-b || return 1
  fi
  cd $WT || return 1
  [ "$(git branch --show-current)" = "$BRANCH" ] || git checkout -q -B $BRANCH || return 1
  # code snapshot (the scripts these results were made with), keeping this driver as committed in the worktree
  rsync -a --exclude __pycache__ --exclude b2_res_pipeline.sh $CODE/workflow/10_tomography/ workflow/10_tomography/
  mkdir -p data/tomo_v1/b2_res_tests/run_log
  (cd $OUT && find . -type f -size -5M) | rsync -a --files-from=- $OUT/ data/tomo_v1/b2_res_tests/
  cp $Q/queue.txt $Q/runner.log data/tomo_v1/b2_res_tests/run_log/
  big=$(cd $OUT && find . -type f -size +5M -printf '%p (%s B)\n')
  git add workflow/10_tomography data/tomo_v1/b2_res_tests
  if git diff --cached --quiet; then log "nothing new to commit"; else
    git commit -q -F - <<EOF || return 1
Track B B2: resolution tests $REGION (checkerboards + spikes, D-B018), analysed $(date +%F)

$(grep -c ' done rc=0' $Q/runner.log) tomoDD inversions (queue in data/tomo_v1/b2_res_tests/run_log/). Code snapshot of
workflow/10_tomography as used. Inversion directories stay on $RT.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  fi
  git push -q -u origin $BRANCH || return 1
  if gh pr view $BRANCH > /dev/null 2>&1; then
    log "PR updated: $(gh pr view $BRANCH --json url -q .url)"
  else
    body=$(cat <<EOF
Resolution tests for tomoDD region \`$REGION\` (D-B018), run detached by \`workflow/10_tomography/b2_res_pipeline.sh\`.

- Inversions: $(grep -c ' done rc=0' $Q/runner.log) runs, all rc=0 (\`data/tomo_v1/b2_res_tests/run_log/\`): checkerboards 40/60/100 km, two phases, clean / pick noise / noise + perturbed locations; spikes at 7 depths; smoothing 10/20/40.
- Results: \`data/tomo_v1/b2_res_tests/\` (tables, masks, figures). Resolved = local correlation >= 0.7 and amplitude >= 0.5.
- Code: snapshot of \`workflow/10_tomography\` as used (includes work not yet committed on \`tomo/track-b\`).
- Not in git (on /wd1): inversion directories under \`$RT\`.${big:+
- Outputs over 5 MB left out of git (in \`$OUT\`):
\`\`\`
$big
\`\`\`}

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)
    gh pr create --base tomo/track-b --head $BRANCH --title "Track B: resolution tests $REGION (checkerboards + spikes)" --body "$body" || return 1
  fi
}

log "start, stages: $STAGES"
for s in $STAGES; do
  if done_stage $s; then log "stage $s already done, skipped"; continue; fi
  log "stage $s"
  if stage_$s; then mark $s; else log "stage $s FAILED, stopping (rerun the same command to resume)"; exit 1; fi
done
log "finished"
