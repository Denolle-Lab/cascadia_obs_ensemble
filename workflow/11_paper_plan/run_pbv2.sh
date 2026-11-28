#!/bin/bash
# Plate boundaries v2 (D-B116): proposal from the v6.1 relocated seismicity and the bathymetry.
# Outputs: /wd1/mdenolle_data/paper_plan/plate_boundaries_v2/ (GeoJSON, figures, report) and
# data/paper_plan/plate_boundaries_v2/ (small tables). Rules: data/paper_plan/plate_boundaries_v2/RULES.md.
# Usage (from the repository root): bash workflow/11_paper_plan/run_pbv2.sh
set -euo pipefail
M=/home/mdenolle/cascadia_obs_ensemble
export LD_LIBRARY_PATH=$M/.pixi/envs/default/lib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=$M/.pixi/envs/default/bin/python
cd "$(dirname "$0")"
for s in pbv2_fit pbv2_bias pbv2_adopt pbv2_figures; do     # pbv2_adopt: D-B117 adopted version
    echo "== $s $(date)"; nice -n 10 $PY -I $s.py
done
nice -n 10 $PY -I pbv2_report.py "$(date)"
