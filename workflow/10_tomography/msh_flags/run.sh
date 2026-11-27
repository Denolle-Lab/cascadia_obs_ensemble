#!/bin/bash
# Mount St Helens analog test-signal flags for catalog v6.1 (D-B108). Steps in order; each writes under
# /wd1/mdenolle_data/reloc_v61/flags (work/, fig/). Limits: nice 10, at most 6 processes.
set -euo pipefail
cd "$(dirname "$0")"
PY="env LD_LIBRARY_PATH=/home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/lib OMP_NUM_THREADS=1 nice -n 10 /home/mdenolle/cascadia_obs_ensemble/.pixi/envs/default/bin/python"
$PY maf_hist.py                     # 0 pick timing histograms (window definition)
$PY maf_template.py                 # 1 pulse template
$PY maf_scan.py --nproc 6           # 2 continuous-data scan (resumable)
$PY maf_scan.py --confirm --nproc 6 #   second station on days with an EDM-only detection
$PY maf_screen.py                   # 3 signal channels
$PY maf_calendar.py                 # 4 pulse calendar and windows
# RULE.md registered here (2026-11-26 16:27:27 PST), before scoring
$PY maf_score.py --nproc 6          # 5 classes, event_flags_v61.csv, candidates
$PY maf_sweep.py                    # 5b post-hoc diagnostic (no flag)
$PY maf_figs.py --examples 28237,26754,17439,33276 --real 27434,63183,17438,24181,11422
$PY maf_html.py --examples 28237,26754,17439,33276 --real 27434,63183 --probable 17438,24181,11422 \
    --notes notes_findings.html
