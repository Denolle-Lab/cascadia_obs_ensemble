#!/bin/bash
# Q1 slip style (D-B115): every step on catalog v6.1, in dependency order. The segment join (pb_common.assign_segments,
# D-B113 axes; PB_AXES=pb2002 for the old rule) is recomputed inside q1_metrics.py, so a new axis file or segment rule
# only needs this script rerun. Pre-registration: data/paper_plan/q1_slip_style/RESUME.md.
# Output: /wd1/mdenolle_data/paper_plan/q1_slip_style/ (report_q1.html, figures/, tables/, nulls/, vpvs/, covariates/)
# and the small tables in data/paper_plan/q1_slip_style/.
# Resources: nice 10; the clustering nulls fork 5 workers (the tomoDD null-truth run holds 6); MemAvailable > 40 GB.
# Usage (from the repository root): bash workflow/11_paper_plan/run_q1.sh [steps]
#   steps (default all): covariates metrics recluster vpvs metrics2 tests figures report
set -euo pipefail
M=/home/mdenolle/cascadia_obs_ensemble
export LD_LIBRARY_PATH=$M/.pixi/envs/default/lib OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="nice -n 10 $M/.pixi/envs/default/bin/python"
STEPS=${1:-"covariates metrics recluster vpvs metrics2 tests figures report"}
cd "$(dirname "$0")"
for s in $STEPS; do
    echo "== $s $(date)"
    while [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -lt 40 ]; do sleep 60; done
    case $s in
        covariates) $PY q1_covariates.py --tag v61; $PY q1_covariates.py --tag v60 ;;
        metrics)    $PY q1_metrics.py --runs main ;;                       # writes the event table the nulls read
        recluster)  $PY q1_recluster.py --steps mcu,tshuf,mshuf --workers 5 ;;
        vpvs)       $PY q1_vpvs.py ;;
        metrics2)   $PY q1_metrics.py --runs mcu_count,mc_local,ci_window,variants,v60,mcu_recluster ;;
        tests)      $PY q1_tests.py ;;
        figures)    $PY q1_figures.py ;;
        report)     $PY q1_report.py "$(date)" ;;
        *) echo "unknown step $s"; exit 1 ;;
    esac
done
