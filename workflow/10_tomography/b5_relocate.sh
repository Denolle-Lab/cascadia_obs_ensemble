#!/bin/bash
# Track B, step B5: double-difference relocation of the whole catalog with tomoDD-SE in relocation-only mode
# (D-B053, D-B060), detached and resumable. Stages of b5_relocate.py (prep split run collect) after a wait for the
# B4 chain of the same model version; notes/b5_relocation.md.
#
#   cd ~/wt_b5_full
#   VERSION=0.3.1 setsid nohup bash workflow/10_tomography/b5_relocate.sh 9>&- \
#       >> /wd1/mdenolle_data/tomo_v1/b5/b5_relocate_v0.3.1.log 2>&1 < /dev/null &
#
# VERSION (default 0.3.1) = model version; RUN (default b5_v$VERSION) = run name; R1, R0, TIME override the B4 run,
# the B4 iter0 run (station terms) and the time-grid folder (defaults: params b5.defaults). NPROC (default params
# b5.nproc) parallel tomoDD runs, NBOOT (default params b5.nboot) bootstrap runs per cluster.
# Stages (state file /wd1/mdenolle_data/tomo_v1/b5/$RUN/b5_relocate.state; rerun the same command to resume, each
# stage skips what is done):
#   wait     the B4 chain of this version has its catalog (line "catalog" in b4_relocate_v$VERSION.state, or
#            b4_relocate.state for v0.3.0) and catalog_b4_$R1.csv exists. Read only: never touches that chain.
#   prep     start (B4 NLLoc, full precision), picks, differential times, components
#   split    clusters (components; larger ones bisected with a 2-hop halo)
#   run      tomoDD per cluster ($NPROC at once, memory guard)
#   collect  centroid restoration, catalog_b5_$RUN.csv, cluster summary, compare.md (repo data/tomo_v1/b5/$RUN)
#   commit   local commit of data/tomo_v1/b5/$RUN on the worktree's branch (no push, no PR)
set -uo pipefail
VERSION=${VERSION:-0.3.1}
RUN=${RUN:-b5_v$VERSION}
OUT=/wd1/mdenolle_data/tomo_v1/b5/$RUN
mkdir -p $OUT
# private copy: bash reads the script while running it
if [ -z "${B5_COPY:-}" ]; then
  cp "$0" $OUT/.b5_relocate_running.sh
  B5_COPY=1 VERSION=$VERSION RUN=$RUN CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash $OUT/.b5_relocate_running.sh "$@"
fi
CODE=${CODE:?}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib
W=$CODE/workflow/10_tomography
R1=${R1:-b4_iter1_v$VERSION}
ARGS="--version $VERSION --run $RUN --b4-run $R1"
[ -n "${R0:-}" ] && ARGS="$ARGS --iter0 $R0"
[ -n "${TIME:-}" ] && ARGS="$ARGS --time $TIME"
NPROC=${NPROC:-$($PY -c "import yaml; print(yaml.safe_load(open('$W/params.yaml'))['b5']['nproc'])")}
NBOOT=${NBOOT:-$($PY -c "import yaml; print(yaml.safe_load(open('$W/params.yaml'))['b5']['nboot'])")}
STATE=$OUT/b5_relocate.state
B4STATE=/wd1/mdenolle_data/tomo_v1/b4_relocate_v$VERSION.state
[ "$VERSION" = 0.3.0 ] && [ ! -e $B4STATE ] && B4STATE=/wd1/mdenolle_data/tomo_v1/b4_relocate.state

log() { echo "$(date '+%F %T') [b5] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }

stage_wait() {
  log "waiting for the B4 catalog of v$VERSION (line 'catalog' in $B4STATE)"
  until grep -qx catalog $B4STATE 2>/dev/null && [ -s /wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_$R1.csv ]; do
    sleep 120
  done
  log "B4 catalog ready: $(ls -l /wd1/mdenolle_data/reloc_v5/catalog/catalog_b4_$R1.csv)"
}
stage_prep() { (cd $W && $PY b5_relocate.py prep $ARGS); }
stage_split() { (cd $W && $PY b5_relocate.py split $ARGS); }
stage_run() { (cd $W && $PY b5_relocate.py run $ARGS --nproc $NPROC --nboot $NBOOT); }
stage_collect() { (cd $W && $PY b5_relocate.py collect $ARGS --nproc $NPROC); }
stage_commit() {
  cd $CODE || return 1
  local p=data/tomo_v1/b5/$RUN
  git add -- $p || return 1
  if git diff --cached --quiet -- $p; then log "nothing new to commit"; return 0; fi
  git commit -q -F - -- $p <<EOF
Track B B5: whole-catalog tomoDD relocation $RUN (model v$VERSION, B4 $R1): compare.md, cluster summary

Catalog and runs: $OUT (not in git).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
}

log "start (code $CODE, version $VERSION, run $RUN, B4 $R1, nproc $NPROC, nboot $NBOOT)"
for s in wait prep split run collect commit; do
  done_stage $s && { log "stage $s already done"; continue; }
  log "stage $s"
  stage_$s || { log "stage $s FAILED"; exit 1; }
  mark $s
done
log "finished: catalog $OUT/catalog_b5_$RUN.csv, summary $CODE/data/tomo_v1/b5/$RUN/compare.md"
