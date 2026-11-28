#!/bin/bash
# Plate-boundary paper plan on catalog v6.1 (D-B111): every pb_* step through the PB_* variables of pb_common.py.
# Output: /wd1/mdenolle_data/paper_plan/v6.1/ (page, figures, large tables) and its tables/ folder (small tables).
# Usage (from the repository root): bash workflow/11_paper_plan/run_v61.sh ["time label"]
set -euo pipefail
W=/wd1/mdenolle_data
M=/home/mdenolle/cascadia_obs_ensemble
export LD_LIBRARY_PATH=$M/.pixi/envs/default/lib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=$M/.pixi/envs/default/bin/python
export PB_CATALOG=$W/catalog_v6/v6.1/cascadia_catalog_v6.1.csv PB_TAG=v61 PB_LABEL=v6.1
export PB_CLUSTERS=$W/catalog_v6/v6.1_analysis/clusters PB_MCMAP=$W/magnitude_converged/mc_maps_v61
export PB_FAULTS=$W/catalog_v6/v6.1_analysis/faults PB_FAULTS_BIG=$W/catalog_v6/v6.1_analysis/faults
export PB_MODEL=$W/tomo_v1/models/cascadia_obs_tomo/v0.4.2.nc PB_ENS=$W/tomo_v1/models/cascadia_obs_tomo/v0.4.2+ens.100.nc
export PB_MEMBERS=$W/tomo_v1/ensemble_v0.4.2/real PB_SYNTH=$W/tomo_v1/ensemble_v0.4.2/synth
export PB_OUT=$W/paper_plan/v6.1/tables PB_BIG=$W/paper_plan/v6.1
mkdir -p "$PB_OUT"
cd workflow/11_paper_plan
for s in pb_axes pb_structure pb_circularity pb_struct_seis pb_seismicity pb_sensitivity pb_figures pb_axes_compare; do
    echo "== $s $(date)"; nice -n 10 $PY $s.py
done
# before/after metrics against the previous axes (PB_AXES_OLD, default gmrt; tables kept in $PB_BIG/tables_<old>_axes)
export PB_AXES_OLD=${PB_AXES_OLD:-gmrt}
[ -d "$PB_BIG/tables_${PB_AXES_OLD}_axes" ] && nice -n 10 $PY pb_axes_changes.py
# null-truth control (D-B112): score the finished null members on the current node sets, compare, zone check
if ls /wd1/mdenolle_data/tomo_v1/ensemble_v0.4.2_null/synth/s0*/MERGED > /dev/null 2>&1; then
    echo "== pb_null_score $(date)"; nice -n 10 $PY pb_null_score.py
    [ -f "$PB_OUT/null_truth_contrasts_${PB_AXES_OLD}.csv" ] && nice -n 10 $PY pb_null_compare.py
    nice -n 10 $PY pb_null_zones.py
fi
nice -n 10 $PY pb_page.py "${1:-$(date)}"
