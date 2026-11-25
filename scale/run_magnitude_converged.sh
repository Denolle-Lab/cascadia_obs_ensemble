#!/usr/bin/env bash
# Converged Route A magnitudes (decision D-B091) for catalog v4.1 (the paper's magnitudes)
# and catalog v5.1. Only the phase19 solver changes (phase19_solver.py, method "exact");
# amplitudes, SNR, epochs, Hutton-Boore, anchors, the Mw switch at 4.5 and the Mw_cal rule
# are those of the stored runs, whose inputs are read in place and never written:
#   v4.1: /home/mdenolle/cascadia_obs_ensemble/data/magnitude_v4 (phase2/3/18 outputs)
#   v5.1: /wd1/mdenolle_data/catalog_v5/v5.1_analysis/mag
# Steps (each skipped when its output exists; FORCE=1 reruns all):
#   1. phase19 v4.1, solvers exact / lsqr (column-scaled check) / legacy3000 (reproduction)
#   2. v4.1 patch (172113 removed, 900001-900003 = ComCat Mw)  -> cascadia_catalog_M_routeA_converged.csv
#   3. phase19 v5.1, exact                                   -> cascadia_catalog_M_v51_converged.csv
#   4. v51_magnitude_table.py with the converged v4.1 as M_v4 -> cascadia_catalog_v5.1_magnitudes_converged.csv
#   5. near-pick tables v4.1 (exact, legacy3000)
#   6. thread test (1 vs 12 BLAS threads, exact and legacy) and LSQR convergence curve
#   7. gauge (deployment ties), gauge-design test, comparisons (region, magnitude, distance, ComCat,
#      tensors) and the reconciliation with D-B091
#   8. Mc / b: v51_mc_b.py on the converged v5.1 table; v4_mc_maps.py on the converged v4.1 M;
#      clustering v4_clusters_zbz.py on the converged v4.1 M (main run + 3 shuffles)
#   9. Mc/b comparison and report (HTML) -> data/magnitude_converged/
# Large outputs in $OUT (default /wd1/mdenolle_data/magnitude_converged), small tables,
# figures and the report in data/magnitude_converged/ of this repository.
#   setsid nohup nice -n 10 bash scale/run_magnitude_converged.sh > $OUT/driver.log 2>&1 &
# Paths as in scale/run_magnitude_v51.sh: the pixi env and the stored v4 run (gitignored CSVs) come
# from the main checkout (CASCADIA_MAIN, found from the git common directory), the data disk is
# CASCADIA_WD1 (default /wd1/mdenolle_data); the Python side reads the same from v51_paths.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/workflow/05_magnitude"
if [ -z "${CASCADIA_MAIN:-}" ]; then
  COMMON="$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$ROOT/.git")"
  CASCADIA_MAIN="$(dirname "$COMMON")"
fi
export CASCADIA_MAIN
export CASCADIA_WD1="${CASCADIA_WD1:-/wd1/mdenolle_data}"
ENV="${ENV:-$CASCADIA_MAIN/.pixi/envs/default}"
PY="$ENV/bin/python"
export LD_LIBRARY_PATH="$ENV/lib"
export PYTHONPATH="$PWD:$ROOT/workflow/06_analysis"
export OMP_NUM_THREADS="${THREADS:-12}" OPENBLAS_NUM_THREADS="${THREADS:-12}" MKL_NUM_THREADS="${THREADS:-12}"
NPROC="${NPROC:-12}"
OUT="${OUT:-$CASCADIA_WD1/magnitude_converged}"
SMALL="$ROOT/data/magnitude_converged"
V4="${V4:-$CASCADIA_MAIN/data/magnitude_v4}"
V51="${V51:-$CASCADIA_WD1/catalog_v5/v5.1_analysis/mag}"
mkdir -p "$OUT/v4.1" "$OUT/v5.1" "$SMALL"
run() { echo "[$(date +%F' '%T)] $*"; "$@"; }
need() { [ -n "${FORCE:-}" ] || [ ! -s "$1" ]; }

for s in exact lsqr legacy3000; do
  tag="_$s"; [ "$s" = exact ] && tag="_converged_raw"
  need "$OUT/v4.1/cascadia_catalog_M_routeA$tag.csv" && run "$PY" phase19_preferred_magnitude.py \
      --data "$V4" --suffix _routeA --matched "$V4/comcat_mt_matched_v4.csv" --solver "$s" \
      --outdir "$OUT/v4.1" --tag "$tag" > "$OUT/v4.1/phase19_$s.log" 2>&1
done
need "$OUT/v4.1/cascadia_catalog_M_routeA_converged.csv" && run "$PY" mag_converged.py patch
need "$OUT/v5.1/cascadia_catalog_M_v51_converged.csv" && run "$PY" phase19_preferred_magnitude.py \
    --data "$V51" --suffix _v51 --matched "$V51/comcat_mt_matched_v51.csv" --solver exact \
    --outdir "$OUT/v5.1" --tag _converged > "$OUT/v5.1/phase19_exact.log" 2>&1
need "$OUT/v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv" && run "$PY" v51_magnitude_table.py \
    --data "$V51" --suffix _v51 --pm "$OUT/v5.1/cascadia_catalog_M_v51_converged.csv" --out "$OUT/v5.1" \
    --tag _converged --solver exact --v4-mags "$OUT/v4.1/cascadia_catalog_M_routeA_converged.csv" \
    > "$OUT/v5.1/table.log" 2>&1
need "$OUT/v4.1/phase19_near_picks_routeA_converged.parquet" && run "$PY" mag_converged.py near \
    > "$OUT/v4.1/near.log" 2>&1
need "$SMALL/convergence_curve.csv" && run "$PY" mag_converged.py threads > "$OUT/threads.log" 2>&1
need "$SMALL/gauge_deployments.csv" && run "$PY" mag_converged.py gauge > "$OUT/gauge.log" 2>&1
need "$SMALL/gauge_design_test.csv" && run "$PY" mag_converged.py gauge-test > "$OUT/gauge_test.log" 2>&1
need "$SMALL/phase19_runs.csv" && run "$PY" mag_converged.py compare > "$OUT/compare.log" 2>&1
need "$SMALL/reconcile_db091.csv" && run "$PY" mag_converged.py reconcile > "$OUT/reconcile.log" 2>&1

# Mc / b, same estimators as the v5.1 piece (v51_mc_b.py) and as the paper (v4_mc_maps.py), then the
# paper's clustering (v4_clusters_zbz.py main run + 3 shuffles, no sensitivity variants). The module
# globals MAG / OUT are overridden so that nothing under data/catalog_v4 is written.
cd "$ROOT/workflow/06_analysis"
M51="$OUT/v5.1/cascadia_catalog_v5.1_magnitudes_converged.csv"
M41="$OUT/v4.1/cascadia_catalog_M_routeA_converged.csv"
if need "$OUT/mc_b/paired_b.csv"; then
  for st in compute paired plot; do
    run "$PY" v51_mc_b.py --mags "$M51" --out "$OUT/mc_b" --stage $st --nproc "$NPROC" >> "$OUT/mc_b.log" 2>&1
  done
fi
need "$OUT/mc_maps_v41/mc_regional.csv" && run "$PY" -c "
import sys; from pathlib import Path
sys.argv = ['v4_mc_maps.py', '--stage', 'all', '--nproc', '$NPROC']
import v4_mc_maps as mm
mm.MAG = Path('$M41'); mm.OUT = Path('$OUT/mc_maps_v41')
mm.main()" > "$OUT/mc_maps_v41.log" 2>&1
if need "$OUT/clusters_zbz_v41/setting_stats.csv"; then
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 run "$PY" -c "
import sys; from pathlib import Path
sys.argv = ['v4_clusters_zbz.py', '--out', '$OUT/clusters_zbz_v41', '--workers', '$NPROC', '--variants', '']
import v4_clusters_zbz as z
z.MAG = Path('$M41')
z.main()" > "$OUT/clusters.log" 2>&1
  # the report stops at the (absent) sensitivity variants after writing setting_stats.csv
  run "$PY" -c "
import sys; from pathlib import Path
sys.argv = ['v4_clusters_zbz_report.py', '--no-map']
import v4_clusters_zbz_report as r
r.OUT = Path('$OUT/clusters_zbz_v41')
r.main()" >> "$OUT/clusters.log" 2>&1 || true
fi
cd "$ROOT/workflow/05_magnitude"
run "$PY" mag_converged.py mcb > "$OUT/mcb_compare.log" 2>&1
run "$PY" mag_converged.py release > "$OUT/release.log" 2>&1
run "$PY" mag_converged_report.py > "$OUT/report.log" 2>&1
echo "[$(date +%F' '%T)] done -> $SMALL/magnitude_converged_report.html"
