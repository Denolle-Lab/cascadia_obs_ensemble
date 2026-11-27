#!/bin/bash
# Paper figures v2 (D-B114): every figure and the review page in one command.
# Usage (from any checkout): bash workflow/08_figures/run_figures_v2.sh [config.yaml] [figure names, default all]
# Before: the plate-boundary analysis on the final segment rule (D-B113: workflow/11_paper_plan/run_v61.sh) and the
# null-truth score (D-B112: pb_null_score.py) have written their tables into the folders the config names.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
M=/home/mdenolle/cascadia_obs_ensemble
CFG=${1:-$HERE/paper_figures_v2.yaml}
shift || true
export LD_LIBRARY_PATH=$M/.pixi/envs/default/lib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$HERE/../.."
nice -n 10 $M/.pixi/envs/default/bin/python workflow/08_figures/paper_figures_v2.py --config "$CFG" ${@:-all}
