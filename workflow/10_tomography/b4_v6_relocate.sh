#!/bin/bash
# Track B, catalog v6.0 (D-B079, D-B092): NonLinLoc relocation of all catalog v4.1 events in model v0.4.1 (D-B083) with
# the clock-corrected picks (D-B090) and NonLinLoc's source-specific station terms (NLL-SSST, Lomax & Savvaidis 2022).
# Built from the v5.1 chain (b4_v51_relocate.sh); every output goes to /wd1/mdenolle_data/reloc_v6 (RESUME.md there).
# Run detached:
#   STOP_AFTER=wide VERSION=0.4.1 setsid nohup nice -n 10 bash workflow/10_tomography/b4_v6_relocate.sh \
#     9>&- >> /wd1/mdenolle_data/reloc_v6/b4_v6_relocate.log 2>&1 &
# STOP_AFTER=<stage> stops cleanly after that stage. Stages (each marked in the state file; a rerun skips the done ones;
# delete a line of the state file to redo a stage):
#   surface  check only (no rebuild): the D-B068 merged surface sources and the NLLoc floor of v5.1
#            (params b4.surface.floor.file, D-B056) match the md5 of the floor manifest; the floor is shared, not copied
#   grids    b4_time_grids.py VERSION TIME --root reloc_v6: fine 1 km grids, merged surface (params b4.surface_mask),
#            D-B073 field rule (params b4.shallow_crust), fluid nodes rock-continued (FLUID_NODES), Earth-flattened, from
#            the product's Vp_tomography / Vs_tomography (open_model); then: the 1 km fields are md5-identical to the
#            registered v<VERSION>_nll/model.<P|S>.mod.buf, every station-phase has a complete grid (files), no grid had
#            a negative time set to 0
#   wide     b4_wide_grids.py TIME --root reloc_v6: wide fallback + deep grids to z' ~121 km (D-B067, D-B074)
#   manifest grid manifest (md5 of every .hdr and .buf of the fine and wide builds) -> reloc_v6/grids_TIME.md5
#   obs      a2_make_obs.py from PICKS (D-B090 clock-corrected picks) and the v4.1 events into reloc_v6/obs_tiles_v6
#   (later stages: smoke, iter0, static terms, SSST iterations, deep pass, catalog, publish; see notes/v6_plan.md.
#    They are added after the PI's review of the SSST test and of the clock-candidate epochs: the chain refuses to go on.)
set -uo pipefail
VERSION=${VERSION:?set VERSION (model version, e.g. 0.4.1)}
R6=/wd1/mdenolle_data/reloc_v6
if [ -z "${B4_COPY:-}" ]; then
  mkdir -p $R6
  cp "$0" $R6/.b4_v6_relocate_running.sh
  B4_COPY=1 VERSION=$VERSION CODE=${CODE:-$(cd "$(dirname "$0")/../.." && pwd)} exec bash $R6/.b4_v6_relocate_running.sh "$@"
fi
CODE=${CODE:?}
TIME=${TIME:-time_v6_v$VERSION}
PICKS=${PICKS:-/wd1/mdenolle_data/tomo_v1/checks/pick_timing_v6/Cascadia_picks_catalog_v4.1tc.csv}
PICKS_MD5=${PICKS_MD5:-1b21ab1c2cf5aba8b48c64a58fc37356}
EVENTS=${EVENTS:-/home/mdenolle/cascadia_obs_ensemble/data/catalog_v4/Cascadia_relocated_catalog_v4.csv}
TA=$CODE/workflow/08_relocation                                 # Track A scripts (main since PR #38)
NPROC_GRID=${NPROC_GRID:-12}                                    # machine shared: at most 16 processes for this session
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
  (cd $CODE && $MPY $W/b4_time_grids.py $VERSION $TIME --nproc $NPROC_GRID --root $R6) || return 1
  $PY - $R6/$TIME/b4_time_grids_$TIME.manifest.json $R6/work_$TIME $STORE/v${VERSION}_nll $GTSRCE <<'EOF2'
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
for ph in ("P", "S"):                                # the fields are the registered model's NLL fields
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
  (cd $CODE && $MPY $W/b4_wide_grids.py $TIME --nproc $NPROC_GRID --root $R6) || return 1
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
stage_obs() {
  local h; h=$(md5sum < "$PICKS" | cut -c1-32)
  [ "$h" = "$PICKS_MD5" ] || { log "picks $PICKS md5 $h, expected $PICKS_MD5"; return 1; }
  (cd $TA && $PY a2_make_obs.py --picks "$PICKS" --events "$EVENTS" --out $OBS --time $R6/$TIME)
}
stage_pending() {
  log "the next stages (smoke, iter0, terms, SSST iterations, deep, catalog, publish) wait for the PI's review of the "
  log "SSST test and of the clock-candidate epochs (D-B092); see notes/v6_plan.md"
  return 1
}

log "start (code $CODE, model v$VERSION, grids $TIME, picks $PICKS, floor $FLOOR, grid processes $NPROC_GRID)"
for s in surface grids wide manifest obs pending; do
  done_stage $s && { log "stage $s already done"; continue; }
  log "stage $s"
  stage_$s || { log "stage $s FAILED"; exit 1; }
  mark $s
  [ "$s" = "${STOP_AFTER:-}" ] && { log "stopping after $s (STOP_AFTER); rerun without it to continue"; exit 0; }
done
