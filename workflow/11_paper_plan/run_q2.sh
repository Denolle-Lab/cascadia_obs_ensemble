#!/usr/bin/env bash
# Q2 (D-B115) driver: depth limit of offshore seismicity, temperature vs composition.
# Pre-registration: data/paper_plan/q2_depth_limit/RESUME.md (2026-11-27 13:27:22 PST, commit e6bab1ac).
# usage: bash workflow/11_paper_plan/run_q2.sh [all|circ|report]
#   all     every configuration (q2_analysis.py), circularity, extras, figures, report  (~10 min, 1 process)
#   circ    rescore the circularity controls (e.g. after more D-B112 null members merge), then figures and report
#   seeds   Monte Carlo precision: main configuration with 20 independent seeds (4 processes), then figures and report
#   report  figures and report only
# Single process, nice 10 (a tomoDD null-truth run may hold 6 processes; keep MemAvailable > 40 GB).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib
PY=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python
BIG=${Q2_BIG:-/wd1/mdenolle_data/paper_plan/q2_depth_limit}
mkdir -p "$BIG"
run() { nice -n 10 "$PY" -W ignore "$HERE/$1" "${@:2}"; }
what=${1:-all}
if [[ $what == all ]]; then
  [[ -s $BIG/coast_i.txt ]] || /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/gmt coast \
      -R-133/-119/38/51.5 -Di -W -M > "$BIG/coast_i.txt"
  run q2_analysis.py > "$BIG/q2_analysis.log" 2>&1
fi
if [[ $what == seeds ]]; then
  run q2_seeds.py 20 4 > "$BIG/q2_seeds.log" 2>&1
fi
if [[ $what == all || $what == circ ]]; then
  run q2_circularity.py > "$BIG/q2_circularity.log" 2>&1
  run q2_extras.py > "$BIG/q2_extras.log" 2>&1
fi
run q2_figures.py
run q2_report.py
