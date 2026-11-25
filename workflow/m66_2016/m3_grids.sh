#!/bin/bash
# M66 fix, step 3: travel-time grids for the stations used here.
#  3D: Track A model (reloc_v5/model_vpvs176, read-only) cut +-500 km around each station (Track A used
#      +-250 km, too small for these offshore events), Grid2Time Podvin-Lecomte, no angles.
#  1D: iasp91 crust/upper mantle (model1d/iasp91, Vel2Grid), GRID2D.
# Usage: m3_grids.sh GTSRCE_FILE [NPROC]     outputs in /wd1/mdenolle_data/m66_2016/{time3d,time1d}
set -euo pipefail
B=/wd1/mdenolle_data/reloc_v5; O=/wd1/mdenolle_data/m66_2016; NLL=$B/src/NonLinLoc/src/bin
export B O NLL SRCFILE=$(readlink -f "$1")
mkdir -p $O/time3d $O/time1d $O/work_time
one() {
    local sta=$1 ph=$2 src; src=$(grep " $sta " "$SRCFILE")
    if [ ! -s "$O/time3d/cascadia3d.$ph.$sta.time.buf" ]; then
        local w="$O/work_time/$sta.$ph"; mkdir -p "$w"; cd "$w"
        "$NLL/GridCut" "$B/model_vpvs176/cascadia3d.$ph.mod" "cut.$ph.mod" "1001 1001 86  0 0 -5.0  1.0 1.0 1.0" "$src" > cut.log 2>&1
        printf "CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\nGTFILES cut %s/time3d/cascadia3d %s\nGTMODE GRID3D ANGLES_NO\n%s\nGT_PLFD 1.0e-3 0\n" "$O" "$ph" "$src" > g2t.in
        "$NLL/Grid2Time" g2t.in > g2t.log 2>&1 || echo "FAILED 3D $sta $ph"
        rm -f cut.$ph.mod.buf
    fi
    if [ ! -s "$O/time1d/iasp91.$ph.$sta.time.buf" ]; then
        local w="$O/work_time/$sta.$ph.1d"; mkdir -p "$w"; cd "$w"
        printf "CONTROL 1 54321\nTRANS LAMBERT WGS-84 45.0 -125.8 41.0 49.0 0.0\nGTFILES %s/model1d/iasp91 %s/time1d/iasp91 %s\nGTMODE GRID2D ANGLES_NO\n%s\nGT_PLFD 1.0e-3 0\n" "$O" "$O" "$ph" "$src" > g2t.in
        "$NLL/Grid2Time" g2t.in > g2t.log 2>&1 || echo "FAILED 1D $sta $ph"
    fi
}
export -f one
awk '{print $2}' "$SRCFILE" | while read -r s; do echo "$s P"; echo "$s S"; done | xargs -P "${2:-24}" -n 2 bash -c 'one "$0" "$1"'
echo "3D: $(ls $O/time3d/*.time.buf | wc -l)  1D: $(ls $O/time1d/*.time.buf | wc -l)"
