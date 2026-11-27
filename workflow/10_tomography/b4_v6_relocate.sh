#!/bin/bash
# Track B, catalog v6.0 (D-B079, D-B092, D-B095, D-B096, D-B097): NonLinLoc relocation of all 172,342 catalog v4.1 events
# in model v0.4.1 (D-B083) with the clock-corrected picks (D-B090; MH07/MG07 only, D-B097), NLL-SSST (Loc2ssst, CharDist
# 9999 -> 16 -> 8 -> 4 km, D-B096) and the NLL-coherence stage (Lomax & Savvaidis 2022). Built from the v5.1 chain
# (b4_v51_relocate.sh); every output goes to /wd1/mdenolle_data/reloc_v6 (RESUME.md there). One detached, resumable chain:
#   cd /home/mdenolle/wt_v6 && VERSION=0.4.1 setsid nohup nice -n 10 bash workflow/10_tomography/b4_v6_relocate.sh \
#     9>&- >> /wd1/mdenolle_data/reloc_v6/b4_v6_relocate.log 2>&1 < /dev/null &
# Catalog v6.1 is a configuration of the same chain (D-B103, D-B104; RESUME.md in /wd1/mdenolle_data/reloc_v61/run):
#   VERSION=0.4.2 CATVER=6.1 V6_ROOT=/wd1/mdenolle_data/reloc_v61/run V6_SCHEDULE=9999,16 PUBLISH_BRANCH=reloc/v61 \
#   V6_RULECHECK=/wd1/mdenolle_data/reloc_v6/heldout/coh_rule_check.json (the v6.0 R3 partner-rule check, a record)
# Environment: V6_ROOT (run root R6, default reloc_v6), CATVER (catalog version, default 6.0), V6_SCHEDULE (Loc2ssst
# CharDist per SSST pass, default 9999,16,8,4: one stage ssst<CharDist> each), PUBLISH_BRANCH (default reloc/v6-prep).
# Optional catalog inputs (stages catalog and magnitudes pass them to b4_v6_catalog.py; the smoke catalog does not):
# V6_REVIEW_DIR (outputs of review_v51/ingest_review.py -> reviewer columns), V6_EVENT_FLAGS (CSV ID,event_flag merged
# into event_flag, never replacing a flag; catalog v6.1, D-B108: the Mount St Helens analog-signal events,
# analog_test_signal). When V6_EVENT_FLAGS is set and the file is missing, stage catalog waits for it (V6_FLAGS_WAIT_H,
# default 3 h; a log line every 10 min) and then stops: the catalog is never built without it.
# PICKS / PICKS_MD5: another pick file (confirmed clock epochs) -> redo obs.
# STOP_AFTER=<stage> stops cleanly after that stage. Each stage is marked in the state file; a rerun skips the done ones
# (delete a line of the state file to redo a stage). The chain runs from the copy .b4_v6_relocate_running_v6.sh (the grid
# chain started before the SSST stages existed runs .b4_v6_relocate_running.sh; this chain waits for it to exit and never
# overwrites it) and holds reloc_v6/b4_v6_relocate.lock. Processes: reloc_v6/control/nproc ('auto BUSY FREE': BUSY NLLoc
# processes while the tomoDD ensemble runs, FREE after; or one number), new processes only while MemAvailable >= 35 GB.
# The grid builders read R6/control/nproc_grid the same way (b4_time_grids.pool_map, every 30 s; NPROC_GRID without it)
# and start a station-phase job only while MemAvailable stays >= MIN_FREE_GB (40).
#   surface  check only (no rebuild): the D-B068 merged surface sources and the NLLoc floor of v5.1
#            (params b4.surface.floor.file, D-B056) match the md5 of the floor manifest; the floor is shared, not copied
#   grids    b4_time_grids.py VERSION TIME --root reloc_v6: fine 1 km grids, merged surface (params b4.surface_mask),
#            D-B073 field rule (params b4.shallow_crust), fluid nodes rock-continued (FLUID_NODES), Earth-flattened, from
#            the product's Vp_tomography / Vs_tomography (open_model); then: the 1 km fields are md5-identical to the
#            registered v<VERSION>_nll/model.<P|S>.mod.buf, every station-phase has a complete grid (files), no grid had
#            a negative time set to 0
#   wide     b4_wide_grids.py TIME --root reloc_v6: wide fallback + deep grids to z' ~121 km (D-B067, D-B074)
#   manifest grid manifest (md5 of every .hdr and .buf of the fine and wide builds) -> reloc_v6/grids_TIME.md5
#   (the stages below run with b4_v6_reloc.py, b4_v6_catalog.py, b4_v6_acceptance.py, b4_v6_heldout.py; method in
#    b4_v6_reloc.py's docstring)
#   gridcheck the manifest lists every fine and wide P and S grid of every station, buffers complete, no negative or
#            non-finite travel time in any grid -> reloc_v6/grids_TIME.check.json
#   obs      PICKS pinned by md5 (PICKS_MD5); events placed by their catalog v5.1 epicentre (v4 without one:
#            b4_v6_reloc.py tilepos); a2_make_obs.py -> reloc_v6/obs_tiles_v6
#   smoke    the whole recipe below (plan .. acceptance) on SMOKE_TILES into reloc_v6/smoke, with the all-family CC if it
#            is finished, else its test block (cc_dt_all/test: JdF family 1003 is in the smoke tiles); stops the chain on
#            any failure or on an event above the surface
#   heldout  starts b4_v6_heldout.py run in the background (3 processes, nice 12): the v6.0 recipe with 3 station folds on
#            nine CC test clusters (held-out effect of coherence by region, D-B096); acceptance waits for it
#   plan     SSST tiles (25 km, events at their v5.1 epicentre), boxes tile +- 20 km, base grid links, NLLoc jobs -> ROOT/plan
#   iter0    NLLoc it0 without terms in the base grids (fine 1 km grid if it holds the tile box, else wide)
#   ssst9999, ssst16, ssst8, ssst4   Loc2ssst g<k> on it<k> (CharDist 9999, 16, 8, 4 km; halo min(3 CharDist, 50 km);
#            <= 20,000 arrivals per station-phase) and NLLoc it<k+1> in the corrected grids (it4 keeps the oct-trees);
#            one stage per V6_SCHEDULE entry (v6.1: ssst9999, ssst16; the final pass it2 keeps the oct-trees)
#   ccwait   waits for the all-family CC job (cc_dt_all/state/final.json done)
#   coherence NLL-coherence on it4: events with Len3 <= 5 km, partners with pair coherence >= 0.5 (rows straddling a
#            confirmed clock epoch left out)
#   terms    static terms of it0 (fine-grid arrivals, a2_station_terms.py rule) for the deep and edge passes
#   deep     deep pass (D-B067: rejected at the bottom face or z' >= 78 km, not far_only) and edge pass (rejected at a
#            lateral face of the tile box): v5.1 recipe, obs tile +- 75 km box, fine then wide grids, static terms
#   catalog  b4_v6_catalog.py -> reloc_v6/catalog/catalog_b4_v6.0.csv (+ manifest)
#   magnitudes (catalog v6.1 on, D-B105; not in v6.0) converged Route A magnitudes on the catalog's own hypocentres: the
#            catalog of stage catalog is kept as catalog_b4_v<CATVER>_hypo.csv; scale/run_magnitude_v51.sh on it
#            (CATALOG, DATA=MAGDATA, SUFFIX _v61, NAME 6.1), phase19 --solver exact --tag _converged and
#            v51_magnitude_table.py --v4-mags (the converged v4.1 table), as scale/run_magnitude_converged.sh does for
#            v5.1; then the catalog is rebuilt with magnitude_source v<CATVER>_converged (v4.1_converged where the
#            v<CATVER> table has no M) and must equal the _hypo catalog outside the magnitude columns
#   acceptance b4_lit_paired.py (v6.0 and v5.1), b4_v6_heldout.py score, b4_v6_acceptance.py -> data/tomo_v1/b4/v6.0/
#            acceptance.html; stops the chain if any event is above the surface (D-B055)
#   publish  commits the reports, pushes to reloc/v6-prep, writes /wd1/mdenolle_data/tomo_v1/b4_v6.DONE and
#            reloc_v6/v6.0_for_session18.md (session -18 builds the catalog version bundle, atlas and Zenodo draft)
set -uo pipefail
VERSION=${VERSION:?set VERSION (model version, e.g. 0.4.1)}
R6=${V6_ROOT:-/wd1/mdenolle_data/reloc_v6}
CATVER=${CATVER:-6.0}
V6_SCHEDULE=${V6_SCHEDULE:-9999,16,8,4}
RUNCOPY=$R6/.b4_v6_relocate_running_v6.sh
if [ -z "${B4_COPY:-}" ]; then
  mkdir -p $R6
  cp "$0" $RUNCOPY.tmp && mv -f $RUNCOPY.tmp $RUNCOPY      # rename: a copy being executed keeps its own file
  B4_COPY=1 VERSION=$VERSION V6_ROOT=$R6 CATVER=$CATVER V6_SCHEDULE=$V6_SCHEDULE CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash $RUNCOPY "$@"
fi
exec 8> $R6/b4_v6_relocate.lock
flock -n 8 || { echo "$(date '+%F %T') [v6] another v6 chain holds $R6/b4_v6_relocate.lock: not starting"; exit 1; }
CODE=${CODE:?}
TIME=${TIME:-time_v6_v$VERSION}
PICKS=${PICKS:-/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv}
PICKS_MD5=${PICKS_MD5:-1b21ab1c2cf5aba8b48c64a58fc37356}
EVENTS=${EVENTS:-/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/Cascadia_relocated_catalog_v4.csv}
TA=$CODE/workflow/08_relocation                                 # Track A scripts (main since PR #38)
NPROC_GRID=${NPROC_GRID:-12}                                    # machine shared: at most 16 processes for this session
GRID_CTRL=$R6/control/nproc_grid                                # 'auto BUSY FREE' or a number (NPROC_GRID without it)
ENV=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default
PY=$ENV/bin/python
export LD_LIBRARY_PATH=$ENV/lib
MENV=${MENV:-/home/mdenolle/wt_tomo_res/.pixi/envs/models}       # existing pixi env "models" (a new worktree has none)
MPY="env LD_LIBRARY_PATH=$MENV/lib $MENV/bin/python"
W=$CODE/workflow/10_tomography
STORE=/wd1/mdenolle_data/tomo_v1/models/cascadia_obs_tomo
GTSRCE=/wd1/mdenolle_data/reloc_v5/model_vpvs176/stations_GTSRCE.in
OBS=${OBS:-$R6/obs_tiles_v6}
FLOOR=$(cd $W && $PY -c "from manifest import load_params; print(load_params()['b4']['surface']['floor']['file'])")
STATE=$R6/b4_v6_relocate.state
MIN_FREE_GB=${MIN_FREE_GB:-40}
ROOT=${ROOT:-$R6/v${CATVER//./}}                               # catalog run (b4_v6_reloc.py ROOT): v60, v61
# the coherence stage's R3 partner-rule check is v6.0's decision record (b4_v6_reloc.RULECHECK); later versions read it
[ "$CATVER" = 6.0 ] || export V6_RULECHECK=${V6_RULECHECK:-/wd1/mdenolle_data/reloc_v6/heldout/coh_rule_check.json}
SMOKE=$R6/smoke
SMOKE_TILES=${SMOKE_TILES:-t10_40,t10_39,t28_41}               # JdF (CC test family 1003; 578 events), Puget (281)
CC=${CC:-/wd1/mdenolle_data/tomo_v1/cc_dt_all}                 # all-family CC (tomo/cc-dt-all, ~/wt_ccall)
CAT=$R6/catalog/catalog_b4_v$CATVER.csv
V51CAT=/wd1/mdenolle_data/catalog_v5/v5.1/cascadia_catalog_v5.1.csv
V60CAT=/wd1/mdenolle_data/reloc_v6/catalog/catalog_b4_v6.0.csv        # comparison for later versions (acceptance)
V60HELDOUT=/wd1/mdenolle_data/reloc_v6/heldout
MAGDATA=${MAGDATA:-/wd1/mdenolle_data/magnitude_converged/v$CATVER}  # magnitudes stage (D-B105)
MAG41=/wd1/mdenolle_data/magnitude_converged/v4.1/cascadia_catalog_M_routeA_converged.csv
if [ "$CATVER" = 6.0 ]; then MAG_STAGE=""; else MAG_STAGE=magnitudes; fi   # v6.0 kept the v5.1 / v4.1 tables
OUTREP=$CODE/data/tomo_v1/b4/v$CATVER                           # reports committed by publish
if [ "$CATVER" = 6.0 ]; then DONE=/wd1/mdenolle_data/tomo_v1/b4_v6.DONE; else DONE=/wd1/mdenolle_data/tomo_v1/b4_v$CATVER.DONE; fi
PUBLISH_BRANCH=${PUBLISH_BRANCH:-reloc/v6-prep}
SSST_STAGES=$(echo "$V6_SCHEDULE" | tr ',' '\n' | sed 's/^/ssst/' | tr '\n' ' ')
NPROC_NLL=${NPROC_NLL:-40}
export VERSION TIME NPROC_NLL V6_ROOT=$R6 V6_SCHEDULE V6_PICKS=$PICKS
RELOC="$PY $W/b4_v6_reloc.py"

log() { echo "$(date '+%F %T') [v6] $*"; }
done_stage() { grep -qx "$1" "$STATE" 2>/dev/null; }
mark() { echo "$1" >> "$STATE"; log "stage $1 done"; }
mem_ok() {   # the machine is shared: do not start a stage below MIN_FREE_GB of available memory
  local f; f=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  [ "$f" -ge "$MIN_FREE_GB" ] || { log "only $f GB available (< $MIN_FREE_GB): not starting"; return 1; }
}

stage_surface() {
  # the floor and surface sources of v5.1 are reused unchanged: check their md5 against the floor manifest
  $PY - "$FLOOR" <<'EOF2'
import hashlib, json, sys
from pathlib import Path
f = Path(sys.argv[1]); m = json.loads(f.with_name(f.name[:-4] + ".manifest.json").read_text())
bad = []
for o in m["outputs"]:
    h = hashlib.md5(Path(o["path"]).read_bytes()).hexdigest()
    if h != o["md5"]:
        bad.append(o["path"])
    print(f"surface: {o['path']} md5 {h} {'ok' if h == o['md5'] else 'DIFFERS from manifest ' + o['md5']}")
sys.exit(1 if bad else 0)
EOF2
}
stage_grids() {
  (cd $W && $PY -c "from manifest import load_params; b = load_params()['b4']; assert b.get('surface_mask') == 'merged', b.get('surface_mask'); assert b.get('shallow_crust'), 'no b4.shallow_crust'") ||
    { log "params b4.surface_mask is not 'merged' or b4.shallow_crust is unset"; return 1; }
  [ -z "${B4_SURFACE_DIR:-}" ] || { log "B4_SURFACE_DIR is set (test override): unset it"; return 1; }
  mem_ok || return 1
  # identity reference: the registered model's NLL fields, or (model registered without them: v0.4.2) the same fields
  # built from the inversion's merged nodes named in its provenance (b4_time_grids.reference_field)
  local ref=$STORE/v${VERSION}_nll
  if [ ! -e $ref/model.P.mod.buf ]; then
    ref=$R6/field_ref_v$VERSION
    log "no registered $STORE/v${VERSION}_nll: identity reference $ref (fields from the provenance's merged netcdf)"
    (cd $CODE && $MPY $W/b4_time_grids.py $VERSION $TIME --reference-field $ref) || return 1
  fi
  (cd $CODE && $MPY $W/b4_time_grids.py $VERSION $TIME --nproc $NPROC_GRID --root $R6 --nproc-ctrl $GRID_CTRL \
     --mem-min-gb $MIN_FREE_GB) || return 1
  $PY - $R6/$TIME/b4_time_grids_$TIME.manifest.json $R6/work_$TIME $ref $GTSRCE <<'EOF2'
import hashlib, json, sys
from pathlib import Path
m = json.load(open(sys.argv[1])); work, nll = Path(sys.argv[2]), Path(sys.argv[3])
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
ident = {}
for ph in ("P", "S"):                                # the fields are the registered model's NLL fields (or reference)
    ref = nll / f"model.{ph}.mod.buf"
    ident[ph] = (md5(work / f"field_1.{ph}.buf"), md5(ref) if ref.exists() else None)
neg = {k: v.get("negative_set_0") for k, v in (m.get("notes", {}).get("fluid_node_times") or {}).items() if v.get("negative_set_0")}
d = Path(sys.argv[1]).parent
st = [l.split()[1] for l in open(sys.argv[4]) if l.startswith("GTSRCE")]
bad = []
for sta in st:                                       # files, not manifest counts (a resumed build counts only new grids)
    for ph in ("P", "S"):
        h = d / f"cascadia3d.{ph}.{sta}.time.hdr"; b = d / f"cascadia3d.{ph}.{sta}.time.buf"
        n = [int(v) for v in h.read_text().split()[:3]] if h.exists() else None
        if not (n and b.exists() and b.stat().st_size == 4 * n[0] * n[1] * n[2]):
            bad.append(f"{ph}.{sta}")
same = all(a == b for a, b in ident.values())
ok = m["params"].get("surface_mask") == "merged" and not bad and not neg and same
print(f"grids: mask {m['params'].get('surface_mask')}, model_vars {m['params'].get('model_vars')}, "
      f"{2 * len(st) - len(bad)}/{2 * len(st)} complete, shallow_crust {m['params'].get('shallow_crust')}, "
      f"grids with negatives set to 0: {len(neg)}, field_1 md5 (built, registered) {ident}: {'identical' if same else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
EOF2
}
stage_wide() {
  mem_ok || return 1
  (cd $CODE && $MPY $W/b4_wide_grids.py $TIME --nproc $NPROC_GRID --root $R6 --nproc-ctrl $GRID_CTRL \
     --mem-min-gb $MIN_FREE_GB) || return 1
  $PY - $R6/wide_$TIME <<'EOF2'
import glob, json, sys
m = json.load(open(glob.glob(sys.argv[1] + "/b4_wide_grids_*.manifest.json")[0]))
c = m["counts"]
print(f"wide: {c.get('ok')}/{c.get('station_phases')} ok (+{c.get('exists')} existing), fine without wide {c.get('fine_without_wide')}")
sys.exit(0 if c.get("fine_without_wide") == 0 and c.get("ok", 0) + c.get("exists", 0) == c.get("station_phases") else 1)
EOF2
}
stage_manifest() {
  # md5 of every grid file of the build (fine + wide), sorted; its own md5 identifies the grid set of the v6.0 run
  ( cd $R6 && find $TIME wide_$TIME -maxdepth 1 -name 'cascadia3d.*.time.*' -print0 | sort -z |
      xargs -0 -n 64 -P 4 md5sum | sort -k2 > grids_$TIME.md5.partial ) || return 1
  local n; n=$(wc -l < $R6/grids_$TIME.md5.partial)
  local want; want=$(( 4 * $(grep -c '^GTSRCE' $GTSRCE) * 2 ))   # fine + wide, P + S, hdr + buf
  [ "$n" -eq "$want" ] || { log "grid manifest: $n files, expected $want"; return 1; }
  mv $R6/grids_$TIME.md5.partial $R6/grids_$TIME.md5
  log "grid manifest $R6/grids_$TIME.md5: $n files, md5 $(md5sum < $R6/grids_$TIME.md5 | cut -c1-32)"
}
stage_gridcheck() { $PY $W/b4_v6_reloc.py gridcheck; }
stage_obs() {
  local h; h=$(md5sum < "$PICKS" | cut -c1-32)
  [ "$h" = "$PICKS_MD5" ] || { log "picks $PICKS md5 $h, expected $PICKS_MD5"; return 1; }
  $RELOC tilepos $R6/inputs/events_tilepos_v51.csv || return 1
  if [ -d $OBS ] && [ ! -e $OBS/obs_meta.json ]; then log "removing the incomplete $OBS"; rm -rf $OBS; fi
  [ -e $OBS/obs_meta.json ] || (cd $TA && $PY a2_make_obs.py --picks "$PICKS" --events $R6/inputs/events_tilepos_v51.csv \
                                  --out $OBS --time $R6/$TIME) || return 1
  $PY - $OBS/obs_meta.json "$PICKS_MD5" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1]))
ok = m["inputs"]["picks"]["md5"] == sys.argv[2] and m["counts"]["events_in_catalog"] == 172342
print(f"obs: picks md5 {m['inputs']['picks']['md5']}, {m['counts']}")
sys.exit(0 if ok else 1)
EOF2
}
cc_done() { $PY -c "import json,sys; sys.exit(0 if json.load(open('$1/state/final.json')).get('done') else 1)" 2>/dev/null; }
relocate() {   # relocate ROOT [CCDIR]: plan .. catalog (b4_v6_reloc.py, b4_v6_catalog.py), each step resumable
  local R=$1 C=${2:-$CC}
  $RELOC plan $R ${TILES_ARG:-} || return 1
  local k n; n=$(echo "$V6_SCHEDULE" | tr ',' '\n' | wc -l)
  for ((k = 0; k < n; k++)); do $RELOC nll $R $k || return 1; $RELOC lss $R $k || return 1; done
  $RELOC nll $R $n || return 1
  $RELOC coh $R --cc $C && $RELOC terms $R && $RELOC fallback $R
}
stage_smoke() {
  local C=$CC
  cc_done $CC || C=$CC/test
  log "smoke: tiles $SMOKE_TILES, CC $C"
  ( export V6_CHUNK=40 TILES_ARG="--tiles $SMOKE_TILES"; relocate $SMOKE $C ) || return 1
  $PY $W/b4_v6_catalog.py $SMOKE --out $SMOKE/catalog/catalog_smoke.csv || return 1
  $PY $W/b4_v6_acceptance.py $SMOKE/catalog/catalog_smoke.csv --outdir $SMOKE/accept --smoke --label v$CATVER || return 1
  [ ! -e $SMOKE/it4/arrival_problems.csv ] || { log "smoke: undocumented arrival drops"; return 1; }
  log "smoke report: $SMOKE/accept/acceptance.html"
}
HG=("axial,jdf,mtj" "msh,puget,gorda" "blanco,blancow,georgia")
heldout_launch() {   # (re)start every held-out group that is not finished and not running
  mkdir -p $R6/heldout/logs
  local g n s todo
  for g in "${HG[@]}"; do
    n=${g//,/_}; todo=0
    for s in ${g//,/ }; do [ -e $R6/heldout/$s/DONE ] || todo=1; done
    [ $todo = 1 ] || continue
    pgrep -f "b4_v6_heldout.py run --sets $g " > /dev/null && continue
    log "held-out group $g started"
    setsid nohup nice -n 12 env LD_LIBRARY_PATH=$ENV/lib $PY $W/b4_v6_heldout.py run --sets $g --nproc 6 \
      >> $R6/heldout/logs/run_$n.log 2>&1 < /dev/null 8>&- &
  done
}
stage_heldout() { cc_done $CC || true; heldout_launch; }
stage_plan() { $RELOC plan $ROOT; }
stage_iter0() { $RELOC nll $ROOT 0; }
stage_ssst() {   # stage ssst<CharDist>: Loc2ssst g<k> on it<k>, NLLoc it<k+1> (k = the CharDist's place in V6_SCHEDULE)
  local k; k=$(echo "$V6_SCHEDULE" | tr ',' '\n' | grep -nx "$1" | cut -d: -f1)
  [ -n "$k" ] || { log "ssst$1: $1 is not in V6_SCHEDULE $V6_SCHEDULE"; return 1; }
  $RELOC lss $ROOT $((k - 1)) && $RELOC nll $ROOT $k
}
stage_ccwait() {
  local n=0
  until cc_done $CC; do
    [ $((n % 6)) = 0 ] && log "waiting for the CC job ($CC/state: $(ls $CC/state 2>/dev/null | tr '\n' ' '); running: \
$(pgrep -fc '[b]5_ccall_') processes)"
    n=$((n + 1)); sleep 300
  done
}
stage_coherence() { $RELOC coh $ROOT --cc $CC; }
stage_terms() { $RELOC terms $ROOT; }
stage_deep() { $RELOC fallback $ROOT; }
cat_inputs() {   # catalog options from the environment (D-B108 event flags, reviewer verdicts)
  echo ${V6_EVENT_FLAGS:+--event-flags $V6_EVENT_FLAGS} ${V6_REVIEW_DIR:+--review-dir $V6_REVIEW_DIR}
}
stage_catalog() {
  if [ -n "${V6_EVENT_FLAGS:-}" ]; then   # D-B108: wait for the flag file, never build the catalog without it
    local t0 tl m; t0=$(date +%s); tl=0
    until [ -s "$V6_EVENT_FLAGS" ]; do
      m=$(( ($(date +%s) - t0) / 60 ))
      if [ $(( $(date +%s) - tl )) -ge 600 ]; then log "catalog: waiting for the event flags $V6_EVENT_FLAGS (${m} min)"; tl=$(date +%s); fi
      if [ $m -ge $(( ${V6_FLAGS_WAIT_H:-3} * 60 )) ]; then
        log "catalog: STOP - event flag file $V6_EVENT_FLAGS (Mount St Helens analog-signal events, D-B108) still missing \
after ${V6_FLAGS_WAIT_H:-3} h. Put it there (ID,event_flag) and relaunch: the chain resumes at stage catalog."
        return 1
      fi
      sleep 60
    done
    sleep 30                               # a file still being written: settle
    log "catalog: event flags $V6_EVENT_FLAGS ($(($(wc -l < "$V6_EVENT_FLAGS") - 1)) rows, md5 $(md5sum < "$V6_EVENT_FLAGS" | cut -c1-32))"
  fi
  $PY $W/b4_v6_catalog.py $ROOT --out $CAT $(cat_inputs)
}
stage_magnitudes() {   # D-B105 (see the header)
  local pre=${CAT%.csv}_hypo.csv sfx=_v${CATVER//./} tab=$MAGDATA/cascadia_catalog_v${CATVER}_magnitudes_converged.csv
  if [ ! -e $pre ]; then cp $CAT $pre.tmp && cp ${CAT%.csv}.manifest.json ${pre%.csv}.manifest.json && mv $pre.tmp $pre || return 1; fi
  mkdir -p $MAGDATA
  log "magnitudes: Route A chain on $pre -> $MAGDATA (log $MAGDATA/driver.log)"
  ( export OMP_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12 MKL_NUM_THREADS=12
    cd $CODE && CATALOG=$pre DATA=$MAGDATA SUFFIX=$sfx NAME=$CATVER bash scale/run_magnitude_v51.sh >> $MAGDATA/driver.log 2>&1 &&
    cd $CODE/workflow/05_magnitude && export PYTHONPATH=$PWD &&
    { [ -s $MAGDATA/cascadia_catalog_M${sfx}_converged.csv ] ||
      $PY phase19_preferred_magnitude.py --data $MAGDATA --suffix $sfx --matched $MAGDATA/comcat_mt_matched$sfx.csv \
        --solver exact --outdir $MAGDATA --tag _converged > $MAGDATA/phase19_exact.log 2>&1; } &&
    { [ -s $tab ] ||
      $PY v51_magnitude_table.py --data $MAGDATA --suffix $sfx --catalog $pre --name $CATVER \
        --pm $MAGDATA/cascadia_catalog_M${sfx}_converged.csv --out $MAGDATA --tag _converged --solver exact \
        --v4-mags $MAG41 > $MAGDATA/table_converged.log 2>&1; } ) || { log "magnitudes: Route A chain failed (logs in $MAGDATA)"; return 1; }
  [ -z "${V6_EVENT_FLAGS:-}" ] || [ -s "$V6_EVENT_FLAGS" ] || { log "magnitudes: event flag file $V6_EVENT_FLAGS missing"; return 1; }
  $PY $W/b4_v6_catalog.py $ROOT --out $CAT --mag-tables v${CATVER}_converged=$tab,v4.1_converged=$MAG41 $(cat_inputs) || return 1
  $PY - $pre $CAT <<'EOF2'
import sys
import pandas as pd
a, b = (pd.read_csv(f, low_memory=False) for f in sys.argv[1:3])
mag = ["M", "M_unc", "M_type", "ML", "ML_unc", "n_ML", "ML_method", "Mw_mt", "comcat_id", "magnitude_source"]
keep = [c for c in a.columns if c not in mag]
same = list(a.columns) == list(b.columns) and a[keep].equals(b[keep])
print(f"magnitudes: catalog rebuilt, non-magnitude columns {'identical' if same else 'DIFFERENT'} to {sys.argv[1]}; "
      f"magnitude_source {b.magnitude_source.fillna('').value_counts().to_dict()}")
sys.exit(0 if same else 1)
EOF2
}
stage_acceptance() {
  heldout_launch
  while pgrep -f "[b]4_v6_heldout.py run" > /dev/null; do sleep 300; done
  $PY $W/b4_v6_heldout.py score || log "held-out scoring failed (the report says so)"
  # later versions are compared with v6.0 too (plan targets: literature depths no worse than v5.1, held-out CC fit no
  # worse than v6.0 beyond 5 %)
  local ref=() href=()
  [ "$CATVER" = 6.0 ] || { ref=(v6.0=$V60CAT); href=(--heldout-ref $V60HELDOUT); }
  (cd $CODE && $PY $W/b4_lit_paired.py v$CATVER=$CAT "${ref[@]}" v5.1=$V51CAT) || return 1
  $PY $W/b4_v6_acceptance.py $CAT --paired $CODE/data/tomo_v1/depth_literature/paired_v$CATVER.md --heldout $R6/heldout \
    --outdir $OUTREP --label v$CATVER "${href[@]}" || return 1
  cp $OUTREP/acceptance.html $R6/catalog/acceptance_v$CATVER.html
}
stage_publish() {
  local md5c; md5c=$(md5sum < $CAT | cut -c1-32)
  mkdir -p $OUTREP
  cp ${CAT%.csv}.manifest.json $OUTREP/catalog_b4_v$CATVER.manifest.json
  cp $R6/heldout/heldout.csv $R6/heldout/scores.json $OUTREP/ 2>/dev/null
  local sched=${V6_SCHEDULE//,/-} npass; npass=$(echo "$V6_SCHEDULE" | tr ',' '\n' | wc -l)
  cat > $R6/v${CATVER}_for_session18.md <<EOF2
# Catalog v$CATVER relocation: ready for the version bundle ($(date '+%F %T'))

- Catalog: $CAT (md5 $md5c), one row per catalog v4.1 event (v4.1 additions not located are left out for
  catalog_version.py, as b4_catalog.py does); manifest ${CAT%.csv}.manifest.json.
- Model v$VERSION; grids $R6/$TIME and $R6/wide_$TIME, grid manifest $R6/grids_$TIME.md5
  (md5 $(md5sum < $R6/grids_$TIME.md5 | cut -c1-32)); floor $FLOOR; picks $PICKS (md5 $PICKS_MD5, D-B090: MH07/MG07 clock
  corrections only, D-B097).
- Station terms: NLL-SSST grids of the run ($ROOT/g$((npass - 1)): CharDist schedule $sched km, final pass it$npass,
  loc_pass ssst_it$npass), static terms for the deep and edge passes ($ROOT/terms/station_terms.in).
- New columns (b4_v6_catalog.py docstring): model_version, loc_pass, station_terms, ssst_char_km, rms_unweighted,
  n_picks_retimed, n_picks_clock_corrected, shift_h_v51_km, shift_z_v51_km, coherence_relocated, coherence_partners,
  edge_pass, review_problems, review_auto_indicators, review_not_earthquake/review_depth_unreliable/review_relocate/
  review_depth_km (empty, D-B097), magnitude_source (short labels v5.1_converged / v4.1_converged). Magnitude columns
  M ... comcat_id are the converged D-B091 values (v5.1 table, else v4.1 table): the catalog_version.py entry must keep
  the catalog's magnitude columns (magnitudes: catalog).
- Acceptance: $OUTREP/acceptance.html (copy $R6/catalog/acceptance_v$CATVER.html).
EOF2
  echo "$(date '+%F %T') catalog $CAT md5 $md5c (model v$VERSION, grids $TIME, picks $PICKS md5 $PICKS_MD5); summary $R6/v${CATVER}_for_session18.md" > $DONE
  ( cd $CODE || exit 1
    git add -- data/tomo_v1/b4/v$CATVER data/tomo_v1/depth_literature/paired_v$CATVER.md \
      data/tomo_v1/depth_literature/paired_v$CATVER.csv data/tomo_v1/depth_literature/b4_lit_paired_v$CATVER.manifest.json 2>/dev/null
    if git diff --cached --quiet; then log "nothing new to commit"
    else git commit -q -m "Track B v$CATVER: NonLinLoc relocation $(basename $ROOT) (model v$VERSION, NLL-SSST $sched km + coherence): acceptance report, literature pairs, held-out coherence scores

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || exit 1; fi
    git push -q origin HEAD:refs/heads/$PUBLISH_BRANCH || { log "PUSH REFUSED: reports committed locally only"; exit 1; }
  ) || return 1
}

log "start (code $CODE, catalog v$CATVER, model v$VERSION, grids $TIME, SSST CharDist $V6_SCHEDULE, picks $PICKS, floor $FLOOR, root $ROOT)"
# the grid chain started before the SSST stages (copy .b4_v6_relocate_running.sh) must finish first
while pgrep -f '^bash /wd1/mdenolle_data/reloc_v6/\.b4_v6_relocate_running\.sh' > /dev/null; do
  log "the grid chain (.b4_v6_relocate_running.sh) is still running: waiting"; sleep 120
done
cc_done $CC && log "CC job: done" || log "CC job: not done yet (the coherence stage waits for it: stage ccwait)"
for s in surface grids wide manifest gridcheck obs smoke heldout plan iter0 $SSST_STAGES ccwait coherence \
         terms deep catalog $MAG_STAGE acceptance publish; do
  done_stage $s && { log "stage $s already done"; continue; }
  log "stage $s"
  case $s in ssst*) stage_ssst ${s#ssst} ;; *) stage_$s ;; esac || { log "stage $s FAILED"; exit 1; }
  mark $s
  [ "$s" = "${STOP_AFTER:-}" ] && { log "stopping after $s (STOP_AFTER); rerun without it to continue"; exit 0; }
done
log "finished: catalog $CAT"
