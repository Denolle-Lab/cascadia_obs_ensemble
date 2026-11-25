#!/bin/bash
# Track B, catalog v5.1: NLLoc relocation of catalog v4.1 with the round-2 fixes (D-B055..D-B072). Run detached:
#   VERSION=0.3.1 TIME=time_b4_v0.3.1m PICKS=<re-timed picks csv> setsid nohup bash workflow/10_tomography/b4_v51_relocate.sh \
#     9>&- >> /wd1/mdenolle_data/tomo_v1/b4_v51_relocate.log 2>&1 &
# Stages (each marked in the state file; a rerun skips the done ones; delete a line of the state file to redo a stage):
#   surface  b4_surface.py install + floor: merged GMRT/earth_relief surface (D-B068) and the conservative NLLoc floor
#            params b4.surface.floor.file (D-B056)
#   grids    b4_time_grids.py VERSION TIME: fine grids with the merged water/rock boundary (params b4.surface_mask) and
#            any field rule in params b4 (e.g. a shallow-crust rule decided in step 1)
#   wide     b4_wide_grids.py TIME: wide fallback + deep grids to z' ~121 km (D-B067) in reloc_v5/wide_TIME
#   obs      a2_make_obs.py from PICKS (the D-B072 re-timed v4.1 picks) and the v4.1 events into OBS
#   smoke    iter0 settings on SMOKE_TILES into runs/b4_v51_smoke: no refusal, no undocumented arrival drop, every
#            located event below the merged surface; stops the chain otherwise
#   iter0    a2_run_nlloc.py R0 (LOCGAU 0.2 0.0, no LOCGAU2: D-B069; LOCMETH minimum 5: D-B066; merged floor; wide
#            fallback), no station terms
#   terms    a2_parse_hyp.py R0 --no-deep (terms only) + a2_station_terms.py R0 (fine-grid arrivals only)
#   iter1    a2_run_nlloc.py R1 with the R0 station terms
#   deep     a2_run_nlloc.py RD --deep-from R1 (D-B067)
#   catalog  a2_parse_hyp.py R1 --deep RD, a2_build_catalog.py R1, b4_catalog.py R1, b4_acceptance.py R1,
#            b4_lit_paired.py (Mt St Helens / Blanco pairs) -> catalog_b4_R1.csv and data/tomo_v1/b4/R1/
#   publish  commit the reports under the publish lock; DONE marker /wd1/mdenolle_data/tomo_v1/b4_v51.DONE (session -18
#            builds catalog v5.1 with catalog_version.py 5.1 from it)
set -uo pipefail
VERSION=${VERSION:?set VERSION (model version, e.g. 0.3.1)}
if [ -z "${B4_COPY:-}" ]; then
  cp "$0" /wd1/mdenolle_data/tomo_v1/.b4_v51_relocate_running.sh
  B4_COPY=1 VERSION=$VERSION CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash /wd1/mdenolle_data/tomo_v1/.b4_v51_relocate_running.sh "$@"
fi
CODE=${CODE:?}
TIME=${TIME:?set TIME (fine grid folder name, e.g. time_b4_v0.3.1m)}
PICKS=${PICKS:?set PICKS (the D-B072 re-timed v4.1 pick file)}
EVENTS=${EVENTS:-/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/Cascadia_relocated_catalog_v4.csv}
TA=${TA:-/home/mdenolle/wt_reloc/workflow/08_relocation}     # Track A scripts (reloc/track-a with the v5.1 pieces)
R0=${R0:-b4_v51_iter0}; R1=${R1:-b4_v51_iter1}; RD=${RD:-b4_v51_deep}; RS=${RS:-b4_v51_smoke}
SMOKE_TILES=${SMOKE_TILES:-t31_35,t07_33,t09_24}             # Cascades (~950 events), Axial (~9,400), Blanco (~4,100) tiles
NPROC_GRID=${NPROC_GRID:-40}; NPROC_NLL=${NPROC_NLL:-100}
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib
MPY="pixi run --manifest-path $CODE/pixi.toml -e models python"
W=$CODE/workflow/10_tomography
RELOC=/wd1/mdenolle_data/reloc_v5
OBS=${OBS:-$RELOC/obs_tiles_v51}
FLOOR=$(cd $W && $PY -c "from manifest import load_params; print(load_params()['b4']['surface']['floor']['file'])")
STATE=/wd1/mdenolle_data/tomo_v1/b4_v51_relocate.state
LOCK=/wd1/mdenolle_data/tomo_v1/publish.lock
NLL_OPTS=(--time "$TIME" --time-wide "wide_$TIME" --obs-dir "$OBS" --locgau "0.2 0.0" --locgau2 off --topo "$FLOOR")   # D-B069: constant 0.2 s, no LOCGAU2

log() { echo "$(date '+%F %T') [v51] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }

stage_surface() { (cd $W && $PY b4_surface.py install && $PY b4_surface.py floor --nproc 12) && [ -s "$FLOOR" ]; }
stage_grids() {
  # preflight (field verifier should_fix 3): the v5.1 water/rock boundary is params b4.surface_mask merged
  (cd $W && $PY -c "from manifest import load_params; m = load_params()['b4'].get('surface_mask'); assert m == 'merged', m") ||
    { log "params b4.surface_mask is not 'merged'"; return 1; }
  [ -z "${B4_SURFACE_DIR:-}" ] || { log "B4_SURFACE_DIR is set (test override): unset it"; return 1; }
  (cd $CODE && $MPY $W/b4_time_grids.py $VERSION $TIME --nproc $NPROC_GRID) || return 1
  # post-check (should_fix 4): merged, every grid ok, no negative time clamped
  $PY - $RELOC/$TIME/b4_time_grids_$TIME.manifest.json <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1]))
neg = {k: v.get("negative_set_0") for k, v in (m.get("notes", {}).get("fluid_node_times") or {}).items() if v.get("negative_set_0")}
c = m["counts"]
ok = m["params"].get("surface_mask") == "merged" and c.get("ok") == c.get("station_phases") and not neg
print(f"grids: mask {m['params'].get('surface_mask')}, {c.get('ok')}/{c.get('station_phases')} ok, shallow_crust "
      f"{m['params'].get('shallow_crust')}, grids with negatives {len(neg)}")
sys.exit(0 if ok else 1)
EOF2
}
stage_wide() {
  (cd $CODE && $MPY $W/b4_wide_grids.py $TIME --nproc $NPROC_GRID) || return 1
  $PY - $RELOC/wide_$TIME <<'EOF2'
import glob, json, sys
m = json.load(open(glob.glob(sys.argv[1] + "/b4_wide_grids_*.manifest.json")[0]))
c = m["counts"]
print(f"wide: {c.get('ok')}/{c.get('station_phases')} ok (+{c.get('exists')} existing), fine without wide {c.get('fine_without_wide')}")
sys.exit(0 if c.get("fine_without_wide") == 0 and c.get("ok", 0) + c.get("exists", 0) == c.get("station_phases") else 1)
EOF2
}
stage_obs() { (cd $TA && $PY a2_make_obs.py --picks "$PICKS" --events "$EVENTS" --out $OBS --time $TIME); }
stage_smoke() {
  (cd $TA && $PY a2_run_nlloc.py $RS "${NLL_OPTS[@]}" --tiles $SMOKE_TILES --nproc 24) || return 1
  [ ! -e $RELOC/runs/$RS/arrival_problems.csv ] || { log "smoke: undocumented arrival drops ($RELOC/runs/$RS/arrival_problems.csv)"; return 1; }
  (cd $W && $PY - "$RELOC/runs/$RS" <<'EOF'
import glob, sys
import numpy as np
import b4_surface
rows = []
for f in glob.glob(sys.argv[1] + "/*/loc.sum.grid0.loc.hyp"):
    ok = False
    for line in open(f):
        if line.startswith("NLLOC "):
            ok = '"LOCATED"' in line
        elif line.startswith("GEOGRAPHIC") and ok:
            t = line.split(); i = t.index("Lat")
            rows.append((float(t[i + 1]), float(t[i + 3]), float(t[i + 5])))
la, lo, z = map(np.array, zip(*rows))
zt = 6371.0 * (1 - np.exp(-z / 6371.0))
h = b4_surface.surface_m(lo, la) / 1e3
n_above = int((zt < -h).sum())
print(f"smoke: {len(zt)} located, {n_above} above the merged surface")
sys.exit(1 if n_above or len(zt) == 0 else 0)
EOF
  )
}
stage_iter0() { (cd $TA && $PY a2_run_nlloc.py $R0 "${NLL_OPTS[@]}" --nproc $NPROC_NLL); }
stage_terms() { (cd $TA && $PY a2_parse_hyp.py $R0 --no-deep && $PY a2_station_terms.py $R0) && [ -s $RELOC/runs/$R0/station_terms.in ]; }
stage_iter1() { (cd $TA && $PY a2_run_nlloc.py $R1 "${NLL_OPTS[@]}" --stat $RELOC/runs/$R0/station_terms.in --nproc $NPROC_NLL); }
stage_deep() { (cd $TA && $PY a2_run_nlloc.py $RD "${NLL_OPTS[@]}" --stat $RELOC/runs/$R0/station_terms.in --deep-from $R1 --nproc $NPROC_NLL); }
stage_catalog() {
  (cd $TA && $PY a2_parse_hyp.py $R1 --deep $RD && $PY a2_build_catalog.py $R1) || return 1
  mkdir -p $CODE/data/tomo_v1/b4/$R1
  cp $CODE/data/reloc_v5/catalog_v5_trackA_${R1}_tiers.md $CODE/data/tomo_v1/b4/$R1/ 2>/dev/null || true
  (cd $CODE && $PY $W/b4_catalog.py $R1 && $PY $W/b4_acceptance.py $R1 --old b4_iter1_v0.3.1 &&
   $PY $W/b4_lit_paired.py v5.1=$RELOC/catalog/catalog_b4_$R1.csv v0.3.1=$RELOC/catalog/catalog_b4_b4_iter1_v0.3.1.csv)
}
stage_publish() {
  ( flock 9
    cd $CODE || exit 1
    git add -- data/tomo_v1/b4/$R1 data/tomo_v1/depth_literature || exit 1
    if git diff --cached --quiet -- data/tomo_v1/b4/$R1 data/tomo_v1/depth_literature; then log "nothing new to commit"
    else git commit -q -m "Track B: v5.1 NLLoc relocation $R1 (model v$VERSION, grids $TIME, picks $(basename $PICKS)): reports

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- data/tomo_v1/b4/$R1 data/tomo_v1/depth_literature || exit 1; fi
    git push -q origin $(git branch --show-current) || exit 1
  ) 9> $LOCK || return 1
  echo "$(date '+%F %T') catalog $RELOC/catalog/catalog_b4_$R1.csv (model v$VERSION, grids $TIME, picks $PICKS)" > /wd1/mdenolle_data/tomo_v1/b4_v51.DONE
}

log "start (code $CODE, model v$VERSION, grids $TIME, picks $PICKS, Track A $TA, floor $FLOOR)"
for s in surface grids wide obs smoke iter0 terms iter1 deep catalog publish; do
  done_stage $s && { log "stage $s already done"; continue; }
  log "stage $s"
  stage_$s || { log "stage $s FAILED"; exit 1; }
  mark $s
done
log "finished: catalog $RELOC/catalog/catalog_b4_$R1.csv"
