#!/bin/bash
# Track B: merged model and B4 relocation of the whole catalog, detached and resumable (PI 2026-11-23: "go on to the
# next steps ... stop before GrowClust; make the merged model the Zenodo product").
#
#   cd ~/wt_tomo_res
#   setsid nohup bash workflow/10_tomography/b4_relocate.sh 9>&- >> /wd1/mdenolle_data/tomo_v1/b4_relocate.log 2>&1 < /dev/null &
#
# Stages (recorded in /wd1/mdenolle_data/tomo_v1/b4_relocate.state; rerun the same command to resume):
#   wait     the five phase-2 resolution-test pipelines (<tile>_m0, publish_d200_s<W> in their pipeline.state)
#   merge    b2_merge.py: merged model -> v$VERSION (Zarr, netCDF, NonLinLoc velocity grids at cell centres), registry
#   grids    b4_time_grids.py: travel-time grids of the 696 stations (D-B028 P-1/P-5) -> reloc_v5/$TIME
#   iter0    Track A a2_run_nlloc.py (all v4 events, 25 km tiles), no station terms
#   terms    Track A a2_station_terms.py from iter0
#   iter1    a2_run_nlloc.py with the iter0 station terms
#   catalog  a2_build_catalog.py (tiers, ANSS check) and b4_catalog.py (depths back from the EFT frame, comparison
#            with v4-NLL3D iter1)
#   bundle   b2_model_bundle.py $VERSION -> velocity_models.zip, upload to the private Zenodo draft 23084158
#            (PI 2026-11-23 19:30: upload the model as soon as it is built, before relocating)
#   commit_model  commit + push the model registry, PR #33 note
#   gate     waits for /wd1/mdenolle_data/tomo_v1/b4_gate_v$VERSION.release (PI: model on the atlas/dashboard before
#            the next step); create the file (one line saying why) to let the relocation start
#   publish  commit, push, PR #33 comment. Stops here: no GrowClust (B5).
# Stage order: wait merge bundle commit_model gate grids iter0 terms iter1 catalog publish
set -uo pipefail
if [ -z "${B4_COPY:-}" ]; then
  cp "$0" /wd1/mdenolle_data/tomo_v1/.b4_relocate_running.sh
  B4_COPY=1 CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash /wd1/mdenolle_data/tomo_v1/.b4_relocate_running.sh "$@"
fi
CODE=${CODE:?}
VERSION=${VERSION:-0.3.0}
TIME=time_b4_v$VERSION
R0=b4_iter0_v$VERSION
R1=b4_iter1_v$VERSION
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib
MPY="pixi run --manifest-path $CODE/pixi.toml -e models python"
W=$CODE/workflow/10_tomography
TA=/home/mdenolle/wt_reloc/workflow/08_relocation          # Track A scripts (reloc/track-a worktree)
RUNS=/wd1/mdenolle_data/tomo_v1/b2_tomodd
RELOC=/wd1/mdenolle_data/reloc_v5
STATE=/wd1/mdenolle_data/tomo_v1/b4_relocate.state
LOCK=/wd1/mdenolle_data/tomo_v1/publish.lock
TILES="mendocino_gorda_c blanco oregon washington ridges_north"

log() { echo "$(date '+%F %T') [b4] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }

stage_wait() {
  log "waiting for the phase-2 resolution tests of: $TILES"
  while true; do
    local left=0
    for t in $TILES; do
      grep -q "^publish_d200_s" $RUNS/${t}_m0/pipeline.state 2>/dev/null && continue
      grep -q "stage prep FAILED\|stage [a-z]*_d200_s[0-9]* FAILED" $RUNS/${t}_m0/pipeline.log 2>/dev/null &&
        { log "${t}_m0 phase 2 FAILED (see $RUNS/${t}_m0/pipeline.log)"; return 1; }
      left=$((left + 1))
    done
    [ $left -eq 0 ] && return 0
    sleep 120
  done
}

stage_merge() {
  (cd $CODE && $MPY $W/b2_merge.py $VERSION "merged model: the five tiles restarted from M0 (damping 200, smoothing by tile), blended with box taper x resolution window on the M0 reference (D-B046, merge_design M1-M3)" --nll --status candidate)
}

stage_grids() { (cd $CODE && $MPY $W/b4_time_grids.py $VERSION $TIME --nproc ${NPROC_GRIDS:-48}); }

stage_iter0() { (cd $TA && $PY a2_run_nlloc.py $R0 --time $TIME --nproc ${NPROC_NLL:-100}); }

stage_terms() {   # a2_parse_hyp.py writes runs/<run>/locations.csv, which the station terms read
  (cd $TA && $PY a2_parse_hyp.py $R0 && $PY a2_station_terms.py $R0) && [ -s $RELOC/runs/$R0/station_terms.in ]; }

stage_iter1() { (cd $TA && $PY a2_run_nlloc.py $R1 --time $TIME --nproc ${NPROC_NLL:-100} --stat $RELOC/runs/$R0/station_terms.in); }

stage_catalog() {
  (cd $TA && $PY a2_parse_hyp.py $R1 && $PY a2_build_catalog.py $R1) || return 1
  mkdir -p $CODE/data/tomo_v1/b4/$R1
  cp /home/mdenolle/wt_reloc/data/reloc_v5/catalog_v5_trackA_${R1}_tiers.md $CODE/data/tomo_v1/b4/$R1/ 2>/dev/null || true
  (cd $CODE && $PY $W/b4_catalog.py $R1)
}

stage_bundle() {
  local line md5
  line=$(cd $CODE && $MPY $W/b2_model_bundle.py $VERSION | tail -1) || return 1
  md5=$(echo "$line" | awk '{print $3}')
  log "bundle: $line"
  (cd $CODE && $PY utils/zenodo_picks_draft.py upload 23084158 /wd1/mdenolle_data/tomo_v1/models/zenodo/velocity_models.zip) || return 1
  printf '%s,23084158,velocity_models.zip,%s,%s,%s,"private draft (not submitted); product = merged model v%s with NonLinLoc grids; all registered versions (netCDF, Zarr, provenance)"\n' \
    "$(date +%F)" "$md5" "$(stat -c %s /wd1/mdenolle_data/tomo_v1/models/zenodo/velocity_models.zip)" \
    "$(cd $CODE && $PY -c "import pandas as pd; print(' '.join(pd.read_csv('data/tomo_v1/models/registry.csv', dtype=str).version))")" \
    "$VERSION" >> $CODE/data/tomo_v1/models/zenodo_uploads.csv
}

stage_commit_model() {
  ( flock 9
    cd $CODE || exit 1
    git add data/tomo_v1/models && git commit -q -m "Track B: merged model v$VERSION built (b2_merge.py) and uploaded to the private Zenodo draft 23084158

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || true
    git push -q origin $(git branch --show-current) || exit 1
    gh pr comment 33 --body "**Merged model v$VERSION built** ($(date '+%F %H:%M')) and uploaded to the private Zenodo draft (velocity_models.zip). Registry: data/tomo_v1/models/registry.csv. The relocation waits until the model is on the atlas/dashboard (gate file b4_gate_v$VERSION.release)." > /dev/null
  ) 9> $LOCK
}

stage_gate() {
  local f=/wd1/mdenolle_data/tomo_v1/b4_gate_v$VERSION.release
  log "model v$VERSION built and uploaded; relocation waits for $f (created once the model is on the atlas/dashboard)"
  until [ -e $f ]; do sleep 60; done
  log "gate released: $(head -1 $f)"
}

stage_publish() {
  exec 9> $LOCK; flock 9
  cd $CODE || return 1
  git add data/tomo_v1/models data/tomo_v1/b4 2>/dev/null
  git commit -q -F - <<EOF || true
Track B: merged model v$VERSION (Zenodo product) and B4 NLLoc relocation of the catalog in it (stops before GrowClust)

Model store /wd1/mdenolle_data/tomo_v1/models; time grids $RELOC/$TIME; runs $RELOC/runs/$R0, $R1; catalog
$RELOC/catalog/catalog_b4_$R1.csv.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  git push -q origin $(git branch --show-current) || return 1
  gh pr comment 33 --body "**Merged model v$VERSION and B4 relocation done** ($(date '+%F %H:%M')). Stopped before GrowClust.

$(sed -n '1,40p' data/tomo_v1/b4/$R1/compare.md)

Model: data/tomo_v1/models/registry.csv (v$VERSION); Zenodo private draft 23084158 velocity_models.zip updated. Not yet reviewed by Claude in the session." > /dev/null && log "commented on PR #33"
}

log "start (code $CODE, version $VERSION)"
for s in wait merge bundle commit_model gate grids iter0 terms iter1 catalog publish; do
  done_stage $s && { log "stage $s already done"; continue; }
  log "stage $s"
  stage_$s || { log "stage $s FAILED"; exit 1; }
  mark $s
done
log "finished (stopped before GrowClust)"
