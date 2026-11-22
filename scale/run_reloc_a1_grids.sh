#!/bin/bash
# Track A, A1: travel-time + take-off-angle grids for every station and phase (NonLinLoc Grid2Time,
# Podvin-Lecomte) on +-250 km station-centred cuts of the 3D model built by workflow/08_relocation/a1_build_model.py.
# Usage: [MODEL=model_vpvs176 TIME=time_vpvs176] scale/run_reloc_a1_grids.sh [NPROC]     (~9 s and ~1 GB RAM per station-phase)
set -euo pipefail
B=/wd1/mdenolle_data/reloc_v5
MODEL=${MODEL:-model_vpvs176}; TIME=${TIME:-time_vpvs176}
export B MODEL TIME NLL=$B/src/NonLinLoc/src/bin
NPROC=${1:-48}
mkdir -p "$B/$TIME" "$B/work_$TIME" "$B/logs_$TIME"

one() {  # $1 station code, $2 phase
    local sta=$1 ph=$2 w="$B/work_$TIME/$1.$2"
    [ -s "$B/$TIME/cascadia3d.$ph.$sta.time.buf" ] && [ -s "$B/$TIME/cascadia3d.$ph.$sta.angle.buf" ] && return 0
    mkdir -p "$w"; cd "$w"
    local src; src=$(grep " $sta " "$B/$MODEL/stations_GTSRCE.in")
    "$NLL/GridCut" "$B/$MODEL/cascadia3d.$ph.mod" "cut.$ph.mod" "501 501 86  0 0 -5.0  1.0 1.0 1.0" "$src" > cut.log 2>&1
    printf "CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\nGTFILES cut cascadia3d %s\nGTMODE GRID3D ANGLES_YES\n%s\nGT_PLFD 1.0e-3 0\n" "$ph" "$src" > g2t.in
    "$NLL/Grid2Time" g2t.in > g2t.log 2>&1 || true
    cp g2t.log "$B/logs_$TIME/$sta.$ph.g2t.log"
    if grep -qi "error" g2t.log || [ ! -s "cascadia3d.$ph.$sta.angle.buf" ]; then return 1; fi
    for k in time angle; do for e in hdr buf; do mv "cascadia3d.$ph.$sta.$k.$e" "$B/$TIME/"; done; done
    rm -f "cut.$ph.mod.buf" "cascadia3d.$ph.mod.buf"
}
export -f one
awk '{print $2}' "$B/$MODEL/stations_GTSRCE.in" | while read -r s; do echo "$s P"; echo "$s S"; done \
  | xargs -P "$NPROC" -n 2 bash -c 'one "$0" "$1" || echo "FAILED $0 $1"' 2>&1 | tee "$B/logs_$TIME/run_grids.log"
echo "grids: $(ls $B/$TIME/*.time.buf | wc -l) time, $(ls $B/$TIME/*.angle.buf | wc -l) angle"
